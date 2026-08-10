#!/usr/bin/env python
"""
prepare_libraryM.py
===================
Convert a Desai-lab "Library M" phenotype CSV (the ``MABA*_..._phenotype_data.csv``
files from background_dependence_SARS-CoV-2) into the plain ``seq,y,y_var`` CSV that
``analysis.py`` consumes -- no changes to the analysis code, no forked script.

Why a converter (and not a new analysis script)
-----------------------------------------------
The Library M files are ALREADY close to the GB1 shape: ``ph_mean`` is the
phenotype, ``ph_sd`` its noise, and the varied RBD positions are given as one
column each (``pos_332``, ``pos_339``, ...). The only real work is:

    1. keep just the VARYING positions (the pos_* columns) as an aligned "seq",
       instead of the full 176-residue ``aa_geno`` string (176 positions would blow
       up one-hot encoding and the interaction combinatorics; only ~36 vary),
    2. drop rows whose called residues are biologically unusable -- a stop codon
       ('*') or an uncalled/ambiguous residue ('X') -- so the alphabet stays the
       20 amino acids plus '-' (deletion, which IS meaningful here),
    3. rename ph_mean -> y, ph_sd -> y_var.

The column order of the pos_* fields is preserved (genomic order), and a sidecar
``*_positions.csv`` records seq-index -> RBD position so SHAP indices remain
interpretable (index 0 = pos_332, etc.), the way GB1's indices mean 39/40/41/54.

Backgrounds
-----------
Library M spans four genetic backgrounds, but in these per-file tables the
background differences are encoded IN the pos_* columns themselves (the ``sort``
and ``dataset`` columns are single-valued within a file). So there is no separate
background stratifier to split on here: all varying pos_* loci are the feature set,
and any background x mutation structure shows up directly in the SHAP interactions.
If you later use a file that DOES carry a background/lineage column, stratify with
``--filter-col``/``--filter-val`` before analysis.

Usage
-----
    python prepare_libraryM.py \
        --in  data/library_M/MABAa_ACE2_phenotype_data.csv \
        --out data/library_M/MABAa_ACE2.seqy.csv

    # then feed the result straight to the existing analysis tool:
    python scripts/analysis.py --data data/library_M/MABAa_ACE2.seqy.csv \
        --kernel jenga --gpu --n-condition 5000 \
        --interaction-order 2 --order-max 2 \
        --out-dir output/libraryM_ACE2

Notes
-----
* L for these files is ~36. Keep ``--interaction-order 2`` for analysis: order 2
  enumerates 36*35/2 = 630 pairwise terms (fine); order 3 would be ~7000 (don't).
* '*' (stop) and 'X' (uncalled) rows are dropped by default. Pass ``--keep-bad``
  to retain them (not recommended for a fitness model).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# residues that make a row unusable for a fitness model
BAD_RESIDUES = {"*", "X"}


def convert(in_path: Path, out_path: Path, *, keep_bad: bool = False,
            filter_col: str | None = None, filter_val: str | None = None,
            drop_constant: bool = True) -> None:
    df = pd.read_csv(in_path, index_col=0)

    # optional background/lineage stratification (not needed for the MABA* files,
    # where such columns are single-valued -- see the module docstring)
    if filter_col:
        if filter_col not in df.columns:
            raise ValueError(f"--filter-col '{filter_col}' not in {in_path.name} "
                             f"(have: {list(df.columns)}).")
        before = len(df)
        df = df[df[filter_col].astype(str) == str(filter_val)]
        print(f"[prep] filtered {filter_col}=={filter_val}: {before} -> {len(df)} rows")

    for col in ("ph_mean", "ph_sd"):
        if col not in df.columns:
            raise ValueError(f"expected column '{col}' not in {in_path.name}.")

    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    if not pos_cols:
        raise ValueError(f"no 'pos_*' columns found in {in_path.name}.")
    # preserve the file's genomic column order
    print(f"[prep] {len(pos_cols)} pos_ columns; {len(df)} rows in")

    # drop rows with missing phenotype
    n0 = len(df)
    df = df[df["ph_mean"].notna()].copy()
    if len(df) < n0:
        print(f"[prep] dropped {n0 - len(df)} rows with missing ph_mean")

    # build the aligned seq from the varying positions
    pos = df[pos_cols].astype(str)

    # drop rows containing stop ('*') or uncalled ('X') residues
    if not keep_bad:
        bad_mask = pos.isin(BAD_RESIDUES).any(axis=1)
        n_bad = int(bad_mask.sum())
        if n_bad:
            print(f"[prep] dropped {n_bad} rows containing {sorted(BAD_RESIDUES)} "
                  "(stop / uncalled residues)")
        df = df[~bad_mask].copy()
        pos = pos[~bad_mask]

    # optionally drop positions that are constant AFTER filtering (no information)
    if drop_constant:
        keep = [c for c in pos_cols if pos[c].nunique() > 1]
        dropped = [c for c in pos_cols if c not in keep]
        if dropped:
            print(f"[prep] dropped {len(dropped)} now-constant positions: "
                  f"{', '.join(dropped)}")
        pos_cols = keep
        pos = pos[pos_cols]

    # sanity: every cell must be a single character
    lens = pos.apply(lambda s: s.str.len()).to_numpy()
    if not np.all(lens == 1):
        bad_cols = [pos_cols[j] for j in range(len(pos_cols)) if not np.all(lens[:, j] == 1)]
        raise ValueError(f"non-single-character residues in columns {bad_cols}; "
                         "unexpected encoding.")

    seq = pos.agg("".join, axis=1)
    L = len(pos_cols)
    alphabet = "".join(sorted(set("".join(seq.values))))

    out = pd.DataFrame({
        "seq": seq.values,
        "y": df["ph_mean"].to_numpy(dtype=float),
        "y_var": df["ph_sd"].to_numpy(dtype=float),
    })

    # collapse exact-duplicate genotypes (same seq measured more than once):
    # inverse-variance-ish average via simple mean of y, mean of y_var. Report only.
    n_dupe = len(out) - out["seq"].nunique()
    if n_dupe:
        print(f"[prep] note: {n_dupe} rows share a seq with another row "
              "(kept as-is; deduplicate downstream if desired)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # sidecar: seq index -> RBD position, so SHAP indices are interpretable
    sidecar = out_path.with_name(out_path.stem + "_positions.csv")
    pd.DataFrame({
        "seq_index": list(range(L)),
        "rbd_position": [c.replace("pos_", "") for c in pos_cols],
    }).to_csv(sidecar, index=False)

    print(f"[prep] wrote {out_path}  ({len(out)} sequences, L={L}, "
          f"alphabet='{alphabet}' A={len(alphabet)})")
    print(f"[prep] wrote position map {sidecar}")
    print(f"[prep] y (ph_mean): min={out['y'].min():.3f} max={out['y'].max():.3f} "
          f"mean={out['y'].mean():.3f}")
    print("[prep] NEXT: python scripts/analysis.py "
          f"--data {out_path} --kernel jenga --gpu --n-condition 5000 "
          "--interaction-order 2 --order-max 2 --out-dir output/libraryM")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="input MABA*_phenotype_data.csv")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="output seq,y,y_var CSV for analysis.py")
    ap.add_argument("--keep-bad", action="store_true",
                    help="keep rows with '*' (stop) or 'X' (uncalled) residues "
                         "(default: drop them)")
    ap.add_argument("--keep-constant", action="store_true",
                    help="keep pos_ columns that are constant after filtering "
                         "(default: drop them, since they carry no information)")
    ap.add_argument("--filter-col", default=None,
                    help="optional column to stratify on (e.g. a background/lineage "
                         "column, if present) before converting")
    ap.add_argument("--filter-val", default=None,
                    help="value to keep for --filter-col")
    args = ap.parse_args()

    if bool(args.filter_col) != bool(args.filter_val):
        ap.error("--filter-col and --filter-val must be given together.")

    convert(Path(args.in_path), Path(args.out_path),
            keep_bad=args.keep_bad,
            filter_col=args.filter_col, filter_val=args.filter_val,
            drop_constant=(not args.keep_constant))


if __name__ == "__main__":
    main()