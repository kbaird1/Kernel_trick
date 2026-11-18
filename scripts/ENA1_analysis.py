"""
ENA1_analysis.py

Find and characterize "landmark" segregants: yeast segregants with GLOBALLY high
fitness that nevertheless carry the deleterious ENA1(RM) allele. These are the
input set for local SHAP attribution under the generalized kernel-trick model --
the question being asked of the model is "why does it predict high fitness for
this genotype despite the single largest-effect locus being in its bad state?"

DATA LAYOUT (confirmed empirically)
-----------------------------------
  qtls_li_hq.csv          19,833 measured segregants.
                            seq   - 83-char A/B string, one char per focal QTL
                            y     - relative fitness under lithium exposure
                            y_var - measurement variance of y
                          The experimental dataset. Self-contained.

  qtls_li_hq.seqs.txt     131,072 = 2^17 genotypes x 83 chars. NOT measured
                          data: the enumerated Figure 4D landscape (16 top QTLs
                          + 1 background pseudo-locus).

  qtls_li_hq.seqs_key.txt 131,072 17-char strings, plain binary enumeration.
                          Row k is the 17-bit index of row k of seqs.txt.
                          Diffing the two recovers which of the 83 positions the
                          16 top QTLs occupy.

GENE LABELS
-----------
No annotation file ships with the data. The 16 recovered loci come out strictly
ascending (7, 8, 12, 33, ... 77), so the key bits are ordered by GENOMIC
COORDINATE, not by decay factor as the paper's prose lists them. Assigning the
paper's 16 genes by chromosome and within-chromosome coordinate yields
GENE_BY_LOCUS below, corroborated by four independent phenotypic predictions
from the paper (checked in test_ena1_interactions):

    HAL9  (74) neutral with ENA1(BY), beneficial with ENA1(RM)
    PHO84 (59) beneficial ONLY in the ENA1(RM) background
    MKT1  (69) detrimental with ENA1(BY), beneficial with ENA1(RM)
    HAP1  (54) deleterious with BOTH ENA1 alleles

ENA1 itself is called independently, on effect size, so it does not depend on
this mapping being right.

SELECTION CRITERIA
------------------
Three filters, all reported and drawn in panel C of the figure:
  1. ENA1 state    -- deleterious (RM) allele at the ENA1 locus
  2. Global cutoff -- y >= quantile(y, --high-quantile) over ALL segregants
  3. Precision     -- P(true fitness > cutoff) >= --min-post-prob, from y_var

Criterion 2 is deliberately GLOBAL rather than relative to the ENA1(RM)
subgroup. The point is to hand SHAP genotypes that are fit on the absolute
scale, so the explanation addresses "high fitness despite ENA1(RM)" rather than
"merely above-average for a crippled background."

  Alternative, if a larger set is ever needed (--rm-relative): threshold at the
  same quantile computed WITHIN the ENA1(RM) subgroup. On this dataset that
  yields ~985 segregants instead of ~7, and it removes the ENA1 main effect from
  the contrast, so SHAP would explain compensation rather than absolute fitness.
  Different question -- kept available but off by default.

Usage:
    python scripts/ENA1_analysis.py
    python scripts/ENA1_analysis.py --high-quantile 0.85 --min-post-prob 0.75
    python scripts/ENA1_analysis.py --rm-relative        # the alternative above
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: safe on a login/compute node
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "qtls_li_hq"
DEFAULT_OUT_DIR = REPO_ROOT / "output"

# ---- project modules ------------------------------------------------------ #
# analysis.py is deliberately NOT used. Its _build_model_from_params picks the
# kernel from --kernel and always constructs a GeneralProductKernel, so a
# Connectedness checkpoint (theta shape (L, 1)) is loaded into a kernel that
# expects (L, A+1) = (L, 3). That does not raise; it produces a meaningless
# kernel and the model collapses to its zero prior mean (observed R^2 = -4.29,
# exactly what a constant-zero predictor scores on this data).
#
# Instead we load the checkpoint here and infer the kernel family from the
# parameter shapes, so the file itself decides which kernel it is.
_HAVE_EPIK = True
_EPIK_IMPORT_ERR = None


def bootstrap_epik(project_root: Path | None = None):
    """Put the project root, modules/ and epik/ on sys.path, then import epik.

    Mirrors analysis.py's _bootstrap_paths (~line 143): epik is vendored in the
    repo and needs epik_loader.ensure_epik_loaded before its submodules resolve.
    Returns a namespace of the classes and helpers this script needs.
    """
    root = Path(project_root) if project_root else REPO_ROOT

    # NOTE: root/epik must NOT go on sys.path. The package lives at
    # root/epik/epik, so `from epik.epik.kernel import ...` needs ROOT on the
    # path. Adding root/epik makes `import epik` resolve to the inner package
    # and `epik.epik` then looks for a third level that does not exist
    # (ModuleNotFoundError: No module named 'epik.epik').
    for sub_p in (root, root / "modules", root / "scripts"):
        sp = str(sub_p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    try:
        import epik_loader
        epik_loader.ensure_epik_loaded(epik_code_root=str(root))
        print("  epik_loader.ensure_epik_loaded OK")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [note] epik_loader unavailable or failed ({exc}); "
              "relying on sys.path alone")

    def _import_epik():
        """Try the known module layouts, then report what is actually on disk."""
        import importlib
        attempts = [("epik.epik.kernel", "epik.epik.model", "epik.epik.utils"),
                    ("epik.kernel", "epik.model", "epik.utils")]
        errors = []
        for kpath, mpath, upath in attempts:
            try:
                km = importlib.import_module(kpath)
                mm = importlib.import_module(mpath)
                um = importlib.import_module(upath)
                print(f"  imported epik via {kpath}")
                return km, mm.EpiK, um.encode_seqs
            except Exception as exc:                           # noqa: BLE001
                errors.append(f"{kpath}: {type(exc).__name__}: {exc}")
        print("  !! could not import epik. Attempts:")
        for e in errors:
            print(f"     {e}")
        edir = root / "epik"
        print(f"  contents of {edir}:")
        if edir.exists():
            for item in sorted(edir.iterdir())[:20]:
                print(f"     {'d' if item.is_dir() else 'f'} {item.name}")
        else:
            print("     (does not exist)")
        raise ImportError("epik not importable; see the layout above")

    kmod, EpiK, encode_seqs = _import_epik()
    from kernel_trick_computer_refactored import KernelTrickComputer
    from shap_kernel_trick_computer import ShapKernelTrickComputer, make_background

    def pick(*names):
        for n in names:
            if hasattr(kmod, n):
                return getattr(kmod, n)
        return None

    return SimpleNamespace(
        EpiK=EpiK,
        encode_seqs=encode_seqs,
        KernelTrickComputer=KernelTrickComputer,  # noqa: F405
        ShapKernelTrickComputer=ShapKernelTrickComputer,
        make_background=make_background,
        Connectedness=pick("ConnectednessKernel", "Connectedness"),
        Jenga=pick("JengaKernel", "Jenga"),
        GeneralProduct=pick("GeneralProductKernel", "GeneralProduct"),
        kernel_module=kmod,
    )


def _to_np(t):
    """Detach a torch tensor to numpy; pass arrays and scalars through."""
    import torch
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


N_LOCI = 83
N_LANDSCAPE_BITS = 17          # 16 top QTLs + background pseudo-locus

# Locus index (0-based, into the 83-char seq) -> gene. See GENE LABELS above.
GENE_BY_LOCUS = {
    7: "SUS1", 8: "SUP45", 12: "ENA1", 33: "TAO3", 39: "HAL5", 47: "PTR2",
    51: "DPH5", 54: "HAP1", 57: "VIP1", 59: "PHO84", 60: "BUL2", 69: "MKT1",
    70: "MLF3", 74: "HAL9", 75: "IRA2", 77: "NRT1",
}
CHR_BY_LOCUS = {
    7: "II", 8: "II", 12: "IV", 33: "IX", 39: "X", 47: "XI",
    51: "XII", 54: "XII", 57: "XII", 59: "XIII", 60: "XIII", 69: "XIV",
    70: "XIV", 74: "XV", 75: "XV", 77: "XV",
}
ENA1_LOCUS_EXPECTED = 12

# Encoding polarity, established in this dataset and cross-checked three ways:
#   ENA1 (locus 12): B is beneficial -> B = BY, per the paper's Figure 4C.
#   HAP1 (54) and MKT1 (69): B is deleterious, which the paper also reports for
#   the BY allele at both -- so B = BY *uniformly*, not just at ENA1.
# Hence 'A'*83 is the RM parental genotype and 'B'*83 the BY parental genotype,
# and the paper gives their relative fitnesses directly. Using the RM genotype as
# the SHAP background makes phi_0 == f(RM wild type), which should land on -0.30.
RM_LETTER, BY_LETTER = "A", "B"
PAPER_WT_FITNESS = {"RM": -0.30, "BY": -0.18, "best_predicted_combo": -0.08}

# Paper's Figure 4E findings, used as validation not as input.
PAPER_CONDITIONAL = {
    "HAL9": "neutral with BY, beneficial with RM",
    "PHO84": "beneficial only in RM background",
    "MKT1": "detrimental with BY, beneficial with RM",
    "HAP1": "deleterious with both alleles",
}

# Colours
C_BY = "#3B7EA1"        # ENA1 BY (beneficial)
C_RM = "#C1553B"        # ENA1 RM (deleterious)
C_MARK = "#1A1A1A"      # landmark outline
C_BG = "#CCCCCC"


def banner(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def sub(t: str) -> None:
    print(f"\n--- {t} ---")


def gene_label(locus: int) -> str:
    g = GENE_BY_LOCUS.get(locus)
    return f"{g}" if g else f"L{locus}"


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_segregants(path: Path) -> pd.DataFrame:
    sub(f"load {path.name}")
    df = pd.read_csv(path)

    missing = {"seq", "y", "y_var"} - set(df.columns)
    if missing:
        raise ValueError(f"expected columns seq/y/y_var, missing {missing}")

    df["seq"] = df["seq"].astype(str).str.strip()
    lengths = df["seq"].str.len()
    if lengths.nunique() != 1 or lengths.iloc[0] != N_LOCI:
        raise ValueError(f"expected all seqs length {N_LOCI}, got "
                         f"{lengths.value_counts().to_dict()}")
    if not df["seq"].str.fullmatch(r"[AB]+").all():
        raise ValueError("seqs contain characters outside {A,B}")

    sd = np.sqrt(df["y_var"])
    print(f"  {len(df):,} segregants x {N_LOCI} loci")
    print(f"  y    : mean={df['y'].mean():.4f} sd={df['y'].std():.4f} "
          f"range=[{df['y'].min():.4f}, {df['y'].max():.4f}]")
    print(f"  y sd : median={sd.median():.4f} "
          f"range=[{sd.min():.4f}, {sd.max():.4f}] "
          f"({sd.max() / sd.min():.1f}x spread)")
    return df


def genotype_matrix(seqs: pd.Series) -> np.ndarray:
    """(n, 83) uint8, 1 = 'B' allele."""
    arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), N_LOCI)
    return (arr == b"B").astype(np.uint8)


# --------------------------------------------------------------------------- #
# Recover the Figure 4D locus mapping
# --------------------------------------------------------------------------- #

def recover_landscape_mapping(seqs_path: Path, key_path: Path) -> pd.DataFrame | None:
    sub("recover Figure 4D locus mapping")
    if not (seqs_path.exists() and key_path.exists()):
        print("  enumeration files absent -- skipping (not required)")
        return None

    seqs = [ln.strip() for ln in seqs_path.read_text().splitlines() if ln.strip()]
    keys = [ln.strip() for ln in key_path.read_text().splitlines() if ln.strip()]
    if len(seqs) != len(keys):
        print(f"  !! row count mismatch {len(seqs)} vs {len(keys)} -- skipping")
        return None
    if set(seqs[0]) != {"A"}:
        print(f"  !! row 0 is not all-A -- skipping")
        return None

    baseline = np.frombuffer(seqs[0].encode(), dtype="S1")
    rows = []
    for bit in range(N_LANDSCAPE_BITS):
        k = 1 << (N_LANDSCAPE_BITS - 1 - bit)      # key is plain binary, bit 0 = MSB
        expected = "A" * bit + "B" + "A" * (N_LANDSCAPE_BITS - bit - 1)
        if keys[k] != expected:
            print(f"  !! key row {k} is {keys[k]!r}, expected {expected!r} "
                  "-- not plain binary; mapping unreliable")
            return None
        flipped = np.flatnonzero(
            np.frombuffer(seqs[k].encode(), dtype="S1") != baseline)
        rows.append({
            "key_bit": bit,
            "n_loci_flipped": len(flipped),
            "locus": int(flipped[0]) if len(flipped) == 1 else -1,
            "kind": "single_qtl" if len(flipped) == 1 else "background_pseudo_locus",
        })

    mapping = pd.DataFrame(rows)
    singles = sorted(mapping.loc[mapping["kind"] == "single_qtl", "locus"])
    n_bg = (mapping["kind"] != "single_qtl").sum()
    print(f"  {len(singles)} single-QTL bits, {n_bg} multi-locus bit(s)")

    if len(singles) == 16 and n_bg == 1:
        bg = mapping.loc[mapping["kind"] != "single_qtl"].iloc[0]
        print(f"  matches the paper: 16 QTLs + background bit "
              f"(bit {bg['key_bit']}, flips {bg['n_loci_flipped']} loci)")
    if singles == sorted(GENE_BY_LOCUS):
        print("  recovered loci match GENE_BY_LOCUS exactly -> gene labels valid")
    else:
        print(f"  !! recovered loci {singles}")
        print(f"     differ from GENE_BY_LOCUS {sorted(GENE_BY_LOCUS)}")
        print("     -> gene labels are NOT valid for this run; treat as L<index>")

    mapping["gene"] = mapping["locus"].map(GENE_BY_LOCUS)
    mapping["chrom"] = mapping["locus"].map(CHR_BY_LOCUS)
    print(f"\n{mapping[mapping.kind == 'single_qtl'][['key_bit', 'locus', 'gene', 'chrom']].to_string(index=False)}")
    return mapping


# --------------------------------------------------------------------------- #
# Marginal effects, polarity, ENA1
# --------------------------------------------------------------------------- #

def marginal_effects(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-locus mean fitness difference (B minus A), Welch SE."""
    sub("per-locus marginal effects")
    out = []
    for j in range(X.shape[1]):
        b = X[:, j] == 1
        ya, yb = y[~b], y[b]
        if len(ya) < 2 or len(yb) < 2:
            out.append({"locus": j, "n_A": len(ya), "n_B": len(yb),
                        "mean_A": np.nan, "mean_B": np.nan,
                        "delta_B_minus_A": np.nan, "se": np.nan, "t": np.nan})
            continue
        se = math.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
        d = yb.mean() - ya.mean()
        out.append({"locus": j, "n_A": len(ya), "n_B": len(yb),
                    "mean_A": ya.mean(), "mean_B": yb.mean(),
                    "delta_B_minus_A": d, "se": se,
                    "t": d / se if se else np.nan})

    eff = pd.DataFrame(out)
    eff["gene"] = eff["locus"].map(GENE_BY_LOCUS)
    eff["abs_delta"] = eff["delta_B_minus_A"].abs()
    # 'B' is the beneficial letter wherever the effect of B is positive.
    eff["beneficial_letter"] = np.where(eff["delta_B_minus_A"] > 0, "B", "A")
    eff = eff.sort_values("abs_delta", ascending=False).reset_index(drop=True)

    cols = ["locus", "gene", "n_A", "n_B", "delta_B_minus_A", "t"]
    print(f"  top 10 by |effect|:\n{eff.head(10)[cols].to_string(index=False)}")
    top, second = eff.iloc[0], eff.iloc[1]
    print(f"\n  leader: locus {int(top['locus'])} ({gene_label(int(top['locus']))})  "
          f"delta={top['delta_B_minus_A']:.4f}  t={top['t']:.1f}  "
          f"({top['abs_delta'] / second['abs_delta']:.2f}x runner-up)")
    return eff


def beneficial_allele_count(X: np.ndarray, eff: pd.DataFrame,
                            loci: list[int] | None = None,
                            min_abs_t: float = 4.0) -> np.ndarray:
    """
    Count beneficial alleles per genotype, using each locus's OWN effect
    direction. A raw B-count is nearly useless here: B is beneficial at some
    loci and deleterious at others (genome-wide r with fitness is only ~0.03).
    Loci with |t| below min_abs_t are skipped as directionally unresolved.
    """
    e = eff.set_index("locus")
    use = loci if loci is not None else list(range(N_LOCI))
    use = [j for j in use if abs(e.loc[j, "t"]) >= min_abs_t]
    cnt = np.zeros(X.shape[0], dtype=int)
    for j in use:
        want_b = e.loc[j, "beneficial_letter"] == "B"
        cnt += (X[:, j] == 1) if want_b else (X[:, j] == 0)
    return cnt, use


def identify_ena1(eff: pd.DataFrame, mapping: pd.DataFrame | None) -> int | None:
    banner("identify ENA1")
    top = eff.iloc[0]
    locus = int(top["locus"])
    checks = [
        ("dominant main effect",
         top["abs_delta"] / eff.iloc[1]["abs_delta"] > 1.5,
         f"{top['abs_delta'] / eff.iloc[1]['abs_delta']:.2f}x runner-up"),
        ("clear allele separation", top["abs_delta"] > 0.02,
         f"|delta|={top['abs_delta']:.4f}"),
        ("matches expected ENA1 locus", locus == ENA1_LOCUS_EXPECTED,
         f"locus {locus} vs expected {ENA1_LOCUS_EXPECTED}"),
    ]
    if mapping is not None:
        singles = set(mapping.loc[mapping["kind"] == "single_qtl", "locus"])
        checks.insert(2, ("in Figure 4D top-16", locus in singles,
                          f"locus {locus} "
                          f"{'in' if locus in singles else 'NOT in'} landscape"))
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    if n_fail == 0:
        print(f"\n  ACCEPTED: locus {locus} is ENA1")
    elif n_fail == 1:
        print(f"\n  PROVISIONAL: locus {locus}, 1 failed check -- inspect")
    else:
        print(f"\n  REJECTED: {n_fail} failed checks")
        return None
    return locus


def test_ena1_interactions(X: np.ndarray, y: np.ndarray, ena1: int,
                           beneficial_is_b: bool,
                           mapping: pd.DataFrame | None) -> pd.DataFrame:
    """Figure 4E: per-QTL effect conditional on ENA1 state. Free validation."""
    banner("conditional effects (Figure 4E validation)")
    ben = (X[:, ena1] == 1) if beneficial_is_b else (X[:, ena1] == 0)
    print(f"  ENA1 beneficial n={ben.sum():,}  deleterious n={(~ben).sum():,}")

    loci = (sorted(mapping.loc[mapping["kind"] == "single_qtl", "locus"])
            if mapping is not None else list(range(N_LOCI)))
    rows = []
    for j in loci:
        if j == ena1:
            continue
        r = {"locus": j, "gene": GENE_BY_LOCUS.get(j)}
        for lab, m in [("ben", ben), ("del", ~ben)]:
            b = X[m, j] == 1
            ya, yb = y[m][~b], y[m][b]
            r[f"eff_{lab}"] = (yb.mean() - ya.mean()
                               if len(ya) > 1 and len(yb) > 1 else np.nan)
        r["diff"] = r["eff_del"] - r["eff_ben"]
        rows.append(r)

    df = pd.DataFrame(rows)
    df = df.reindex(df["diff"].abs().sort_values(ascending=False).index)
    print(f"\n{df.to_string(index=False)}")

    # Qualitative checks against the paper's prose.
    print("\n  validation against Figure 4E prose:")
    d = df.set_index("gene")
    tests = {
        "HAL9": lambda r: abs(r.eff_ben) < 0.015 and r.eff_del > 0.02,
        "PHO84": lambda r: r.eff_ben <= 0.005 and r.eff_del > 0.01,
        "MKT1": lambda r: r.eff_ben < -0.02 and r.eff_del > -0.005,
        "HAP1": lambda r: r.eff_ben < 0 and r.eff_del < 0,
    }
    n_ok = 0
    for g, fn in tests.items():
        if g not in d.index:
            print(f"    [ -- ] {g}: not in landscape set")
            continue
        r = d.loc[g]
        ok = bool(fn(r))
        n_ok += ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {g}: "
              f"BY={r.eff_ben:+.4f} RM={r.eff_del:+.4f}  "
              f"expected: {PAPER_CONDITIONAL[g]}")
    print(f"  {n_ok}/{len(tests)} qualitative predictions reproduced")
    if n_ok >= 3:
        print("  -> gene mapping and ENA1 call both corroborated")
    else:
        print("  -> weak corroboration; re-examine GENE_BY_LOCUS")
    return df


def ld_check(X: np.ndarray, eff: pd.DataFrame, thresh: float = 0.2) -> pd.DataFrame:
    """
    Linked markers break the feature-independence assumption behind
    interventional SHAP; attribution smears across a block. Report any pairs
    above |r| = thresh so they can be grouped or handled with a conditional
    SHAP variant downstream.
    """
    sub(f"LD check (|r| > {thresh})")
    R = np.corrcoef(X.T)
    iu = np.triu_indices(N_LOCI, k=1)
    r = R[iu]
    keep = np.abs(r) > thresh
    pairs = pd.DataFrame({
        "locus_i": iu[0][keep], "locus_j": iu[1][keep], "r": r[keep],
    }).sort_values("r", key=np.abs, ascending=False)
    pairs["gene_i"] = pairs["locus_i"].map(GENE_BY_LOCUS)
    pairs["gene_j"] = pairs["locus_j"].map(GENE_BY_LOCUS)
    pairs["gap"] = pairs["locus_j"] - pairs["locus_i"]
    print(f"  {len(pairs)} of {len(r):,} locus pairs exceed |r|={thresh}")
    if len(pairs):
        print(f"  max |r| = {pairs['r'].abs().max():.3f}")
        print(f"{pairs.head(15).to_string(index=False)}")
        print("  -> group these or use a conditional/on-manifold SHAP variant")
    else:
        print("  -> no strong LD; interventional SHAP assumptions are reasonable")
    return pairs


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def prob_above(y: np.ndarray, y_var: np.ndarray, thr: float) -> np.ndarray:
    """P(true fitness > thr) under a normal measurement model."""
    sd = np.sqrt(np.maximum(y_var, 1e-12))
    return 0.5 * np.array([math.erfc((thr - yi) / (s * math.sqrt(2)))
                           for yi, s in zip(y, sd)])


def select_landmarks(df: pd.DataFrame, X: np.ndarray, ena1: int,
                     beneficial_is_b: bool, eff: pd.DataFrame,
                     high_q: float, min_post: float,
                     rm_relative: bool) -> tuple[pd.DataFrame, dict]:
    banner("landmark selection")

    y = df["y"].to_numpy()
    ena1_b = X[:, ena1] == 1
    ben = ena1_b if beneficial_is_b else ~ena1_b

    out = df.copy()
    out["ena1_allele"] = np.where(ena1_b, "B", "A")
    out["ena1_state"] = np.where(ben, "BY", "RM")
    out["y_sd"] = np.sqrt(out["y_var"])

    qtl_loci = [j for j in GENE_BY_LOCUS if j != ena1]
    cnt_all, used_all = beneficial_allele_count(X, eff)
    cnt_qtl, used_qtl = beneficial_allele_count(X, eff, loci=qtl_loci)
    out["n_beneficial"] = cnt_all
    out["n_beneficial_qtl"] = cnt_qtl
    print(f"  beneficial-allele count uses {len(used_all)}/{N_LOCI} loci "
          f"(|t| >= 4) and {len(used_qtl)}/{len(qtl_loci)} of the non-ENA1 QTLs")

    g = out.groupby("ena1_state")["y"].agg(["count", "mean", "median", "std"])
    print(f"\n  fitness by ENA1 state:\n{g.to_string()}")
    sep = g.loc["BY", "mean"] - g.loc["RM", "mean"]
    print(f"  mean separation: {sep:.4f}   "
          f"(paper: BY WT ~ -0.18, RM WT ~ -0.30)")

    if rm_relative:
        thr = float(np.quantile(y[~ben], high_q))
        scope = f"within ENA1(RM), q={high_q:.2f}"
    else:
        thr = float(np.quantile(y, high_q))
        scope = f"global, q={high_q:.2f}"
    print(f"\n  fitness threshold ({scope}): y >= {thr:.4f}")

    out["p_above_thr"] = prob_above(y, df["y_var"].to_numpy(), thr)

    n_rm = int((~ben).sum())
    step1 = out[out["ena1_state"] == "RM"]
    step2 = step1[step1["y"] >= thr]
    step3 = step2[step2["p_above_thr"] >= min_post]

    print(f"\n  selection funnel:")
    print(f"    all segregants            {len(out):>7,}")
    print(f"    ENA1(RM) deleterious      {len(step1):>7,}")
    print(f"    ... and y >= {thr:.4f}      {len(step2):>7,}")
    print(f"    ... and P(y>thr) >= {min_post:.2f}  {len(step3):>7,}   <- landmarks")
    dropped = len(step2) - len(step3)
    if len(step2):
        print(f"    ({dropped} of {len(step2)} = {dropped / len(step2):.0%} of the "
              "point-estimate hits were noise-driven)")

    # Rate comparison: how much does ENA1 state gate access to high fitness?
    hi_by = int(((out["ena1_state"] == "BY") & (out["y"] >= thr)).sum())
    n_by = int(ben.sum())
    r_by = hi_by / n_by
    r_rm = len(step2) / n_rm
    print(f"\n  P(y >= thr | ENA1 BY) = {hi_by:,}/{n_by:,} = {r_by:.4f}")
    print(f"  P(y >= thr | ENA1 RM) = {len(step2):,}/{n_rm:,} = {r_rm:.4f}")
    if r_rm > 0:
        print(f"  -> ENA1(BY) raises the odds of reaching this threshold "
              f"{r_by / r_rm:.0f}x")

    cand = step3.sort_values("y", ascending=False).copy()
    print(f"\n  LANDMARK SET: {len(cand)} segregants")
    if len(cand):
        show = ["y", "y_sd", "p_above_thr", "n_beneficial", "n_beneficial_qtl"]
        print(cand[show].to_string())
        print(f"\n  beneficial alleles (all resolved loci): landmarks median "
              f"{cand['n_beneficial'].median():.0f} vs "
              f"{out['n_beneficial'].median():.0f} dataset-wide")
        print(f"  beneficial alleles (15 non-ENA1 QTLs):  landmarks median "
              f"{cand['n_beneficial_qtl'].median():.0f} vs "
              f"{out['n_beneficial_qtl'].median():.0f} dataset-wide")
        print("  -> enrichment here is partly real compensation and partly an "
              "artifact of selecting on the outcome; do not over-read it")

    meta = {"thr": thr, "scope": scope, "min_post": min_post, "high_q": high_q,
            "n_total": len(out), "n_rm": n_rm, "n_by": n_by,
            "n_step2": len(step2), "n_step3": len(step3),
            "rate_by": r_by, "rate_rm": r_rm, "sep": sep,
            "rm_relative": rm_relative}
    return out, cand, meta


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def make_figure(out: pd.DataFrame, cand: pd.DataFrame, X: np.ndarray,
                eff: pd.DataFrame, ena1: int, beneficial_is_b: bool,
                meta: dict, out_dir: Path) -> Path:
    banner("figure")

    rm = out["ena1_state"] == "RM"
    by = ~rm
    thr, min_post = meta["thr"], meta["min_post"]
    z = math.sqrt(2) * _erfcinv(2 * min_post)     # P(y>thr)=min_post <=> y = thr + z*sd

    fig = plt.figure(figsize=(17.5, 10.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28,
                          left=0.055, right=0.985, top=0.86, bottom=0.075)

    # ---- A: fitness distributions by ENA1 state (Figure 4C analogue) ------- #
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(out["y"].min(), out["y"].max(), 70)
    ax.hist(out.loc[by, "y"], bins=bins, color=C_BY, alpha=0.75,
            label=f"ENA1$^{{BY}}$  n={meta['n_by']:,}")
    ax.hist(out.loc[rm, "y"], bins=bins, color=C_RM, alpha=0.75,
            label=f"ENA1$^{{RM}}$  n={meta['n_rm']:,}")
    ax.axvline(thr, color="k", ls="--", lw=1.2)
    ax.text(thr, ax.get_ylim()[1] * 0.97, f" y={thr:.3f}", fontsize=8,
            va="top", ha="left")
    for v in cand["y"]:
        ax.plot([v], [ax.get_ylim()[1] * 0.045], marker="v", ms=7,
                color=C_MARK, mec="w", mew=0.6, zorder=5, clip_on=False)
    ax.set_xlabel("measured fitness $y$")
    ax.set_ylabel("# segregants")
    ax.set_title("A  ENA1 splits the fitness distribution", loc="left",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.annotate("landmarks sit in the far RM tail,\ninside the BY bulk",
                xy=(cand["y"].median(), ax.get_ylim()[1] * 0.06),
                xytext=(0.42, 0.55), textcoords="axes fraction", fontsize=8,
                ha="left", arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))

    # ---- B: box + whisker, RM vs BY (requested) ---------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    data = [out.loc[rm, "y"].values, out.loc[by, "y"].values]
    bp = ax.boxplot(data, tick_labels=["ENA1$^{RM}$", "ENA1$^{BY}$"],
                    widths=0.5, patch_artist=True, showmeans=True, notch=True,
                    meanprops=dict(marker="D", mfc="w", mec="k", ms=6),
                    medianprops=dict(color="k", lw=1.4),
                    flierprops=dict(marker=".", ms=2, mfc=C_BG, mec="none",
                                    alpha=0.5))
    for patch, c in zip(bp["boxes"], [C_RM, C_BY]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.axhline(thr, color="k", ls="--", lw=1.2)
    ax.text(2.42, thr, f"threshold\n{thr:.3f}", fontsize=8, va="center")
    ax.scatter(np.full(len(cand), 1) + np.random.default_rng(0).normal(0, 0.045, len(cand)),
               cand["y"], s=44, marker="*", color="#F2C14E", ec=C_MARK,
               lw=0.7, zorder=6, label="landmarks")
    for i, (lab, arr) in enumerate(zip(["RM", "BY"], data), start=1):
        ax.text(i, arr.min() - 0.025, f"$\\mu$={arr.mean():.3f}\n$\\sigma$={arr.std():.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel("measured fitness $y$")
    ax.set_title(f"B  RM vs BY: $\\Delta\\mu$ = {meta['sep']:.3f}", loc="left",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- C: the three criteria, drawn geometrically ------------------------ #
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(out.loc[by, "y_sd"], out.loc[by, "y"], s=3, c=C_BG,
               alpha=0.35, lw=0, label="ENA1$^{BY}$")
    ax.scatter(out.loc[rm, "y_sd"], out.loc[rm, "y"], s=4, c=C_RM,
               alpha=0.45, lw=0, label="ENA1$^{RM}$")
    ax.axhline(thr, color="k", ls="--", lw=1.2)
    sd_grid = np.linspace(out["y_sd"].min(), out["y_sd"].max(), 100)
    ax.plot(sd_grid, thr + z * sd_grid, color="k", lw=1.4,
            label=f"P(y>thr)={min_post:.2f}")
    ax.scatter(cand["y_sd"], cand["y"], s=95, marker="*", color="#F2C14E",
               ec=C_MARK, lw=0.8, zorder=6, label="landmarks")
    ax.set_xlabel("measurement s.d.  $\\sqrt{y\\_var}$")
    ax.set_ylabel("measured fitness $y$")
    ax.set_title("C  how the landmarks were found", loc="left",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.text(0.03, 0.97,
            f"1. ENA1$^{{RM}}$ allele\n"
            f"2. $y \\geq$ {thr:.3f} ({meta['scope']})\n"
            f"3. P(true $y >$ thr) $\\geq$ {min_post:.2f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="w", ec="0.6", lw=0.6, alpha=0.9, pad=4))

    # ---- D: genotype sequence space (PCA) --------------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    Xc = X.astype(float) - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var = (S ** 2) / (S ** 2).sum()
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=out["y"], s=3, cmap="viridis",
                    alpha=0.5, lw=0)
    idx = out.index.get_indexer(cand.index)
    ax.scatter(pcs[idx, 0], pcs[idx, 1], s=110, marker="*", color="#F2C14E",
               ec=C_MARK, lw=0.9, zorder=6, label="landmarks")
    plt.colorbar(sc, ax=ax, label="fitness $y$", pad=0.02)
    ax.set_xlabel(f"PC1 ({var[0]:.1%} of genotype variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("D  genotype space (PCA)", loc="left",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(0.03, 0.03,
            "segregants assort near-randomly, so PCs\n"
            "carry little structure: landmarks are NOT\n"
            "a genotype cluster, they are scattered",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(fc="w", ec="0.6", lw=0.6, alpha=0.9, pad=3))

    # ---- E: fitness vs beneficial-allele dosage --------------------------- #
    ax = fig.add_subplot(gs[1, 1])
    for m, c, lab in [(by, C_BY, "ENA1$^{BY}$"), (rm, C_RM, "ENA1$^{RM}$")]:
        ax.scatter(out.loc[m, "n_beneficial"]
                   + np.random.default_rng(1).normal(0, 0.16, int(m.sum())),
                   out.loc[m, "y"], s=3, c=c, alpha=0.3, lw=0, label=lab)
        grp = out.loc[m].groupby("n_beneficial")["y"].mean()
        grp = grp[grp.index.isin(
            out.loc[m, "n_beneficial"].value_counts().loc[lambda s: s >= 20].index)]
        ax.plot(grp.index, grp.values, color=c, lw=2, zorder=4)
    ax.scatter(cand["n_beneficial"], cand["y"], s=110, marker="*",
               color="#F2C14E", ec=C_MARK, lw=0.9, zorder=6, label="landmarks")
    ax.axhline(thr, color="k", ls="--", lw=1.2)
    ax.set_xlabel("# beneficial alleles (locus-specific direction, $|t|\\geq4$)")
    ax.set_ylabel("measured fitness $y$")
    ax.set_title("E  landmarks are dosage outliers, not ENA1 escapees",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- F: landmark genotypes at the 16 focal QTLs ----------------------- #
    # Row labels only stay legible for a modest number of landmarks. The global
    # threshold normally yields ~7; cap so --rm-relative runs stay readable.
    MAX_ROWS = 24
    ax = fig.add_subplot(gs[1, 2])
    qtl = sorted(GENE_BY_LOCUS)
    e = eff.set_index("locus")
    shown = cand.head(MAX_ROWS)
    truncated = len(cand) > MAX_ROWS

    M = np.zeros((len(shown), len(qtl)))
    for r, (_, row) in enumerate(shown.iterrows()):
        for c_, j in enumerate(qtl):
            M[r, c_] = 1.0 if row["seq"][j] == e.loc[j, "beneficial_letter"] else 0.0
    ax.imshow(M, cmap="RdBu", vmin=-0.15, vmax=1.15, aspect="auto")

    # Column annotation uses the FULL landmark set, not just the shown rows.
    Mfull = np.zeros((len(cand), len(qtl)))
    for r, (_, row) in enumerate(cand.iterrows()):
        for c_, j in enumerate(qtl):
            Mfull[r, c_] = 1.0 if row["seq"][j] == e.loc[j, "beneficial_letter"] else 0.0
    freq = Mfull.mean(axis=0)

    ax.set_xticks(range(len(qtl)))
    ax.set_xticklabels([f"{GENE_BY_LOCUS[j]}  {f:.2f}"
                        for j, f in zip(qtl, freq)], rotation=90, fontsize=8)
    for tick, j in zip(ax.get_xticklabels(), qtl):
        if j == ena1:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    if len(shown) <= 14:
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([f"row {i}   y={v:.3f}"
                            for i, v in zip(shown.index, shown["y"])], fontsize=8)
    else:
        ax.set_yticks([0, len(shown) - 1])
        ax.set_yticklabels([f"y={shown['y'].iloc[0]:.3f}",
                            f"y={shown['y'].iloc[-1]:.3f}"], fontsize=8)
    ax.set_ylabel("landmark (sorted by fitness)", fontsize=9)
    title = "F  landmark alleles at the 16 focal QTLs"
    if truncated:
        title += f" (top {MAX_ROWS} of {len(cand)})"
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("blue = beneficial allele, red = deleterious;  number after "
                  f"each gene\nis its beneficial-allele frequency across all "
                  f"{len(cand)} landmarks", fontsize=8)

    # ---- provenance line -------------------------------------------------- #
    fig.suptitle(
        "Landmark segregants: globally high fitness despite the deleterious "
        "ENA1$^{RM}$ allele\n"
        f"{meta['n_total']:,} segregants  \u2192  "
        f"{meta['n_rm']:,} carry ENA1$^{{RM}}$ (locus {ena1})  \u2192  "
        f"{meta['n_step2']:,} also reach y \u2265 {thr:.3f} ({meta['scope']})  \u2192  "
        f"{meta['n_step3']:,} survive P(true y > thr) \u2265 {min_post:.2f}"
        f"      |      P(high | BY) = {meta['rate_by']:.3f} vs "
        f"P(high | RM) = {meta['rate_rm']:.4f}",
        fontsize=11.5, y=0.975, ha="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "ena1_landmarks.png"
    fig.savefig(png, dpi=200)
    fig.savefig(out_dir / "ena1_landmarks.pdf")
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {out_dir / 'ena1_landmarks.pdf'}")
    return png


def _erfcinv(v: float) -> float:
    """Inverse complementary error function, bisection. Avoids a scipy dep."""
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid) > v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =========================================================================== #
# GP fit + first-order SHAP on the landmarks
# =========================================================================== #

# Cohort definitions. Three groups of 7, all attributed the same way, so the
# comparison between them is the actual result.
COHORTS = {
    "rm_high": dict(
        label="ENA1$^{RM}$, globally high fitness",
        short="RM-high",
        color="#C1553B",
        note="the landmark set: deleterious ENA1 yet above the global threshold"),
    "by_matched": dict(
        label="ENA1$^{BY}$, fitness-matched",
        short="BY-matched",
        color="#3B7EA1",
        note="beneficial ENA1, matched one-to-one on measured fitness to RM-high"),
    "rm_mean": dict(
        label="ENA1$^{RM}$, near the RM mean",
        short="RM-typical",
        color="#7A7A7A",
        note="deleterious ENA1 at typical RM fitness -- the baseline both "
             "high-fitness cohorts are measured against"),
}


# Parameter count per locus identifies the kernel family. This is the crux of
# doing the load ourselves: the CHECKPOINT decides which kernel to build, not a
# command-line flag.
#   Connectedness          theta (L, 1)      one decay parameter per locus
#   Jenga / GeneralProduct theta (L, A+1)    = (L, 3) at A=2
THETA_WIDTH_TO_FAMILY = {1: "Connectedness", 3: "GeneralProduct"}

CKPT_REQUIRED = ["covar_module.theta", "covar_module.log_var"]
CKPT_NOISE_KEY = "likelihood.second_noise_covar.raw_noise"


def load_checkpoint(path: Path | None) -> tuple[dict, str]:
    """Read the checkpoint and infer its kernel family. -> (params, family).

    Raises FileNotFoundError / KeyError with a usable message; the caller turns
    that into 'NO MODEL FOUND' and stops rather than proceeding with a model
    that silently predicts a constant.
    """
    import torch
    if path is None:
        raise FileNotFoundError(
            "no --params-path given. Point it at a checkpoint, e.g.\n"
            "  --params-path models/qtls.li.hq/"
            "qtls_li_hq.full.1.Connectedness.model_params.pth")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NO MODEL FOUND at {path}")

    sub(f"checkpoint {path.name}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    params = blob if isinstance(blob, dict) else blob.state_dict()
    print(f"  {len(params)} keys:")
    for k, v in params.items():
        print(f"    {k:<58} "
              f"{tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")

    missing = [k for k in CKPT_REQUIRED if k not in params]
    if missing:
        raise KeyError(f"checkpoint lacks {missing}")

    theta = params["covar_module.theta"]
    shp = tuple(theta.shape)
    if len(shp) != 2 or shp[0] != N_LOCI:
        raise KeyError(f"theta is {shp}; expected ({N_LOCI}, k) for "
                       f"{N_LOCI} loci -- wrong dataset?")
    family = THETA_WIDTH_TO_FAMILY.get(shp[1])
    if family is None:
        raise KeyError(f"theta width {shp[1]} matches no known kernel family "
                       f"{THETA_WIDTH_TO_FAMILY}")
    print(f"\n  theta {shp} -> kernel family: {family}")
    print(f"  (inferred from the parameter shape, not from --kernel; this is why "
          "the load is done here rather than through analysis.py)")
    if CKPT_NOISE_KEY not in params:
        print(f"  [note] no {CKPT_NOISE_KEY}; the learned extra noise term will "
              "be left at its default")
    return params, family


def _load_into_kernel(kernel, params: dict) -> None:
    """Copy theta / log_var into the kernel and VERIFY they landed.

    Uses copy_ on the live parameters rather than a constructor argument, so we
    do not have to guess whether this kernel class takes theta0=. Shapes are
    reconciled when the element count matches -- the checkpoint stores log_var
    as a 0-d scalar while the kernel may hold it as (1,), and a raw
    load_state_dict would reject that.
    """
    import torch
    for ck, attr in [("covar_module.theta", "theta"),
                     ("covar_module.log_var", "log_var")]:
        src = params[ck]
        dst = getattr(kernel, attr, None)
        if dst is None:
            raise AttributeError(f"kernel has no attribute '{attr}'; available: "
                                 f"{[n for n, _ in kernel.named_parameters()]}")
        src_t = torch.as_tensor(src, dtype=dst.dtype, device=dst.device)
        if tuple(src_t.shape) != tuple(dst.shape):
            if src_t.numel() != dst.numel():
                raise ValueError(f"{ck}: checkpoint {tuple(src_t.shape)} vs "
                                 f"kernel {tuple(dst.shape)} -- incompatible")
            print(f"  reshaped {attr}: {tuple(src_t.shape)} -> {tuple(dst.shape)}")
            src_t = src_t.reshape(dst.shape)
        with torch.no_grad():
            dst.copy_(src_t)
        back = _to_np(dst).ravel()
        ref = _to_np(src_t).ravel()
        if not np.allclose(back, ref, rtol=1e-6, atol=1e-9):
            raise ValueError(f"{attr} did not load correctly")
        print(f"  loaded {attr:<8} {tuple(dst.shape)}  "
              f"range [{back.min():+.4f}, {back.max():+.4f}]")


def warm_prediction_strategy(model, X_t, n_probe: int = 8) -> bool:
    """Force GPyTorch to build model.gp.prediction_strategy.

    KernelTrickComputer.__init__ reads
        model.gp.prediction_strategy.covar_cache
    but GPyTorch creates prediction_strategy LAZILY, on the first eval-mode
    call. Straight after set_data it is still None, so constructing the
    computer raises AttributeError on NoneType. analysis.py never hit this
    because run_fit_only predicts on a held-out split first, which warms it as a
    side effect.

    fast_pred_var is enabled deliberately: covar_cache under LOVE is the
    low-rank (n, k) form -- the earlier working run had ktc.S of shape
    (2000, 100) -- whereas without it the cache is the full (n, n) root, which
    at n=19,833 is ~1.6 GB and a different object than the SHAP path expects.
    """
    import torch
    import gpytorch

    gp = model.gp
    gp.eval()
    lik = getattr(model, "likelihood", None)
    if lik is not None:
        lik.eval()

    try:
        dev = next(gp.parameters()).device
    except StopIteration:
        dev = X_t.device
    probe = X_t[:n_probe].to(dev)

    print(f"  warming prediction_strategy with {n_probe} probe sequences "
          f"on {dev} ...")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _ = gp(probe)

    ps = getattr(gp, "prediction_strategy", None)
    if ps is None:
        print("  !! prediction_strategy is still None after an eval-mode call")
        return False
    cache = getattr(ps, "covar_cache", None)
    if cache is None:
        print("  !! prediction_strategy exists but covar_cache is None")
        return False
    print(f"  prediction_strategy ready; covar_cache {tuple(cache.shape)} "
          f"on {cache.device}")
    return True


def load_model_and_shap(args, df: pd.DataFrame, X_geno: np.ndarray):
    """Build the model from a checkpoint and wrap it in the GKT SHAP computer.

    Returns a namespace exposing the same attributes the rest of this script
    uses (shap, L, A, alphabet, model, ...), so nothing downstream changes.

    Deliberate choices:
      * Kernel family from the checkpoint's theta shape, not from a flag.
      * ALL segregants are conditioned on. The landmarks must be IN the
        conditioning set or the posterior cannot use their own precise
        measurements -- with a 4,000-point subset the chance of containing all
        seven was ~1 in 74,000, and they were predicted near the RM group mean.
      * y is CENTERED before conditioning. EpiK defaults to a zero prior mean
        and these checkpoints carry no mean parameter, so an uncentered y of
        mean -0.197 leaves the model predicting ~0 for everything. The offset is
        added back when reporting f(x) and phi_0; the phi_i themselves are
        differences and are unaffected by a constant shift.
    """
    banner("model")

    # Path check FIRST, before importing torch: "no checkpoint" is the most
    # common failure and should not be masked by an environment problem.
    ppath = getattr(args, "params_path", None)
    if ppath is None or not Path(ppath).exists():
        print(f"\n  NO MODEL FOUND"
              f"{'' if ppath is None else f' at {ppath}'}"
              f"{' (--params-path not given)' if ppath is None else ''}")
        print("\n  Checkpoints present:")
        mdir = REPO_ROOT / "models" / "qtls.li.hq"
        found = sorted(mdir.glob("*.pth")) if mdir.exists() else []
        for f in found:
            print(f"    {f.relative_to(REPO_ROOT)}")
        if not found:
            print(f"    (nothing at {mdir})")
        return None

    try:
        import torch
    except ImportError as exc:                                 # noqa: BLE001
        print(f"\n  !! torch not importable: {exc}")
        print("     activate the project environment first (activate_env)")
        return None

    try:
        params, family = load_checkpoint(ppath)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\n  {exc}")
        print("\n  NO MODEL FOUND -- stopping. Checkpoints present:")
        mdir = REPO_ROOT / "models" / "qtls.li.hq"
        found = sorted(mdir.glob("*.pth")) if mdir.exists() else []
        for f in found:
            print(f"    {f.relative_to(REPO_ROOT)}")
        if not found:
            print(f"    (nothing at {mdir})")
        return None

    try:
        ep = bootstrap_epik(getattr(args, "project_root", None))
    except ImportError as exc:                                 # noqa: BLE001
        print(f"\n  {exc}")
        print("  Run from the project root, or pass --project-root explicitly.")
        return None
    kernel_cls = {"Connectedness": ep.Connectedness,
                  "GeneralProduct": ep.GeneralProduct}[family]
    if kernel_cls is None:
        print(f"  !! {family} kernel class not found in "
              f"{ep.kernel_module.__name__}; available: "
              f"{[n for n in dir(ep.kernel_module) if 'Kernel' in n]}")
        return None

    dev = "cuda" if (getattr(args, "gpu", False) and torch.cuda.is_available()) else "cpu"
    if getattr(args, "threads", 0):
        torch.set_num_threads(int(args.threads))
    print(f"\n  device {dev}   torch threads {torch.get_num_threads()}")

    alphabet = "".join(sorted({c for s in df["seq"] for c in s}))
    A, L = len(alphabet), N_LOCI
    print(f"  alphabet {alphabet!r}  L={L}  A={A}")
    if A != 2:
        print(f"  !! expected a bi-allelic alphabet, got A={A}")

    # ---- encode + center -------------------------------------------------- #
    Xenc = ep.encode_seqs(list(df["seq"]), alphabet=alphabet)
    Xt = torch.as_tensor(np.asarray(Xenc), dtype=torch.float32)
    if Xt.shape != (len(df), L * A):
        print(f"  !! encoded X is {tuple(Xt.shape)}, expected "
              f"({len(df)}, {L * A})")
    y_raw = df["y"].to_numpy(dtype=float)
    y_offset = float(y_raw.mean()) if not args.no_center_y else 0.0
    y_t = torch.as_tensor(y_raw - y_offset, dtype=torch.float32)
    yv_t = torch.as_tensor(df["y_var"].to_numpy(dtype=float) * args.yvar_scale,
                           dtype=torch.float32)
    print(f"  conditioning on ALL {len(df):,} segregants")
    print(f"  y centering: offset {y_offset:+.4f} "
          f"({'disabled' if args.no_center_y else 'subtracted before conditioning'})")
    if args.yvar_scale != 1.0:
        print(f"  y_var scaled by {args.yvar_scale}: mean "
              f"{float(yv_t.mean()):.5f} vs Var(y) {y_raw.var():.5f}")

    # ---- build ------------------------------------------------------------ #
    kernel = kernel_cls(n_alleles=A, seq_length=L)
    print(f"\n  built {type(kernel).__name__}; loading checkpoint parameters:")
    _load_into_kernel(kernel, params)

    model = ep.EpiK(kernel, train_noise=(CKPT_NOISE_KEY in params),
                    track_progress=False, device=dev)
    model.set_data(Xt, y_t, y_var=yv_t)
    if CKPT_NOISE_KEY in params:
        with torch.no_grad():
            model.likelihood.second_noise_covar.noise = torch.exp(
                torch.as_tensor(params[CKPT_NOISE_KEY]).reshape(-1).to(
                    model.likelihood.second_noise_covar.noise.device))
        print(f"  extra noise from checkpoint: "
              f"{float(_to_np(model.likelihood.second_noise_covar.noise).ravel()[0]):.6f}")
    print("  no fitting performed -- parameters come entirely from the checkpoint")

    if not warm_prediction_strategy(model, Xt):
        print("  !! cannot build the kernel-trick computer without a populated "
              "prediction_strategy")
        return None

    ktc = ep.KernelTrickComputer(model, alphabet, L)
    shap = ep.ShapKernelTrickComputer(ktc=ktc)
    print(f"  KernelTrickComputer built; S {tuple(ktc.S.shape)}  "
          f"Qy {tuple(ktc.Qy.shape)}")

    n_gpu, n_cpu = audit_model_device(model, dev)
    if dev == "cuda" and n_cpu and not n_gpu:
        print("  !! requested cuda but every model tensor is on cpu")
        if getattr(args, "require_gpu", False):
            return None

    return SimpleNamespace(
        model=model, ktc=ktc, shap=shap, make_background=ep.make_background,
        alphabet=alphabet, L=L, A=A, y_offset=y_offset,
        n_condition_used=len(df), kernel_family=family, device=dev,
    )


def _precision_mask(sub_df: pd.DataFrame, q: float) -> pd.Series:
    """Keep the most precisely measured fraction ``q`` of a subgroup."""
    cut = sub_df["y_var"].quantile(q)
    return sub_df["y_var"] <= cut


def select_cohorts(out: pd.DataFrame, cand: pd.DataFrame, meta: dict,
                   precision_q: float = 0.25) -> dict:
    """Build the three 7-sequence cohorts.

    rm_high     the landmarks already found (unchanged).
    by_matched  ENA1(BY) matched ONE-TO-ONE on measured fitness to rm_high, by
                greedy nearest-neighbour without replacement. Restricted to the
                same posterior-confidence filter as rm_high so the comparison is
                not precise-RM against noisy-BY.
    rm_mean     ENA1(RM) closest to the RM group mean.

    NOTE ON THE DIFFERING PRECISION CRITERIA: rm_high and by_matched both use
    P(true y > threshold) >= min_post, which is meaningful because both sit
    above that threshold. For rm_mean that probability is ~0 by construction
    (the threshold is the global 90th percentile), so precision there is imposed
    as a y_var quantile instead. The cohort table below prints y_var for all
    three so any residual imbalance is visible rather than hidden.
    """
    banner("cohort selection")
    n = len(cand)
    min_post = meta["min_post"]
    cohorts = {"rm_high": cand.copy()}

    # ---- by_matched: greedy nearest-fitness match, no replacement --------- #
    pool = out[(out["ena1_state"] == "BY") & (out["p_above_thr"] >= min_post)].copy()
    print(f"  BY pool with P(y>thr) >= {min_post:.2f}: {len(pool):,} candidates")
    picks, deltas = [], []
    available = pool.copy()
    for target in cand["y"].to_numpy():
        if available.empty:
            break
        j = (available["y"] - target).abs().idxmin()
        picks.append(j)
        deltas.append(float(available.loc[j, "y"] - target))
        available = available.drop(index=j)
    cohorts["by_matched"] = out.loc[picks].copy()

    print(f"  matched {len(picks)}/{n}; |dy| max={max(abs(d) for d in deltas):.4f} "
          f"median={np.median(np.abs(deltas)):.4f}")
    if max(abs(d) for d in deltas) > 0.01:
        print("  !! at least one match differs by >0.01 in fitness; the cohorts are "
              "not tightly matched and part of any SHAP difference is a fitness "
              "difference rather than an ENA1-state difference")

    # ---- rm_mean: closest to the RM group mean --------------------------- #
    rm = out[out["ena1_state"] == "RM"]
    mu = float(rm["y"].mean())
    rm_prec = rm[_precision_mask(rm, precision_q)]
    print(f"\n  RM mean fitness {mu:+.4f}; "
          f"{len(rm_prec):,} RM segregants in the most precise "
          f"{precision_q:.0%} (y_var <= {rm['y_var'].quantile(precision_q):.5f})")
    cohorts["rm_mean"] = rm_prec.assign(
        _d=(rm_prec["y"] - mu).abs()).nsmallest(n, "_d").drop(columns="_d")

    # ---- report ----------------------------------------------------------- #
    print(f"\n  {'cohort':<12} {'n':>3} {'y median':>10} {'y range':>19} "
          f"{'y_var median':>13}")
    for k, df_c in cohorts.items():
        print(f"  {COHORTS[k]['short']:<12} {len(df_c):>3} "
              f"{df_c['y'].median():>+10.4f} "
              f"[{df_c['y'].min():+.4f}, {df_c['y'].max():+.4f}] "
              f"{df_c['y_var'].median():>13.5f}")
    for k, df_c in cohorts.items():
        print(f"\n  {COHORTS[k]['short']} rows: {list(df_c.index)}")
        print(f"    {COHORTS[k]['note']}")
    return cohorts


def audit_model_device(model, want: str) -> tuple[int, int]:
    """Report where the EpiK model's tensors actually live. -> (n_gpu, n_cpu).

    EpiK is not an nn.Module, so named_parameters() does not exist and the real
    GPyTorch model sits at model.gp. Walk attributes instead.

    This exists because analysis.py moves ONLY the KernelTrickComputer to the
    device (_move_to_device, ~line 641) while the model is constructed without
    one at line 457 -- so the SHAP path lands on GPU and the fit silently runs
    on CPU. Diagnosed via nvidia-smi (0% utilization during a fit); the torch
    allocator is useless here because KeOps allocates outside it.

    Fix, in scripts/analysis.py:
        line 457  self._EpiK(kernel_cls(...), track_progress=True,
                             device=cfg.device)
        line 620  same addition
    """
    import torch
    import torch.nn as nn

    found, seen = [], set()

    def walk(obj, prefix, depth=0):
        if id(obj) in seen or depth > 3:
            return
        seen.add(id(obj))
        if isinstance(obj, nn.Module):
            for n, q in obj.named_parameters(recurse=True):
                found.append((f"{prefix}.{n}", str(q.device)))
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:                                  # noqa: BLE001
                continue
            if torch.is_tensor(v):
                found.append((f"{prefix}.{name}", str(v.device)))
            elif isinstance(v, nn.Module):
                walk(v, f"{prefix}.{name}", depth + 1)
            elif isinstance(v, (tuple, list)) and v and torch.is_tensor(v[0]):
                found.append((f"{prefix}.{name}[0]", str(v[0].device)))

    walk(model, "model")
    n_gpu = sum(1 for _, d in found if d.startswith("cuda"))
    n_cpu = sum(1 for _, d in found if d.startswith("cpu"))

    sub(f"model device audit (requested: {want})")
    print(f"  model tensors: {n_gpu} on gpu, {n_cpu} on cpu")
    if n_cpu:
        for path, d in [f for f in found if f[1].startswith("cpu")][:8]:
            print(f"    {path:<42} {d}")
    return n_gpu, n_cpu



def make_pi(sfa, mode: str, empirical_counts=None):
    """Background distribution pi for the attribution reference."""
    wt = None
    if mode == "rm_wildtype":
        wt = RM_LETTER * sfa.L
        mode_arg = "wildtype"
    elif mode == "by_wildtype":
        wt = BY_LETTER * sfa.L
        mode_arg = "wildtype"
    else:
        mode_arg = mode
    return sfa.make_background(mode_arg, sfa.L, sfa.alphabet, wildtype=wt,
                               empirical_counts=empirical_counts,
                               device=sfa.shap.device)


def predict(sfa, seq: str) -> float:
    """Model prediction on the ORIGINAL fitness scale.

    f(x) == E_MAP at a delta distribution on x (fixing every position leaves
    nothing to average). y was centered before conditioning, so the offset is
    added back here -- otherwise every prediction sits ~0.197 low and looks
    like the degenerate-model failure.
    """
    return float(_to_np(sfa.ktc.E_MAP(sfa.shap.target_to_onehot(seq)))) + sfa.y_offset


def check_base_values(sfa) -> dict:
    """phi_0 under the two parental backgrounds vs the paper's wild-type fitnesses.

    f(x) == E_MAP at a delta background, so phi_0 under the RM-wildtype pi IS the
    model's prediction for the RM parental genotype. The paper gives -0.30 (RM) and
    -0.18 (BY), so this is a direct, unfitted check on the GP and the encoding at
    once. A large miss means something upstream is wrong -- do not proceed.
    """
    banner("phi_0 sanity check vs the paper's parental fitnesses")
    out = {}
    for key, mode in [("RM", "rm_wildtype"), ("BY", "by_wildtype")]:
        try:
            v = float(_to_np(sfa.shap.base_value(make_pi(sfa, mode)))) + sfa.y_offset
        except Exception as exc:                            # noqa: BLE001
            print(f"  !! {key} background failed: {exc}")
            continue
        exp = PAPER_WT_FITNESS[key]
        out[key] = v
        ok = abs(v - exp) < 0.06
        print(f"  [{'PASS' if ok else 'FAIL'}] f({key} wild type) = {v:+.4f}  "
              f"(paper {exp:+.2f}, diff {v - exp:+.4f})")
    if len(out) == 2:
        gap = out["BY"] - out["RM"]
        exp_gap = PAPER_WT_FITNESS["BY"] - PAPER_WT_FITNESS["RM"]
        frac = gap / exp_gap if exp_gap else float("nan")
        print(f"\n  PRIMARY CHECK -- parental gap: BY - RM = {gap:+.4f} vs the "
              f"paper's {exp_gap:+.2f}  ({frac:.0%} recovered)")
        print("  The gap matters more than either parent alone: a model that shrinks "
              "predictions toward the dataset mean (-0.197) will land near the BY "
              "value (-0.18) by accident while missing RM badly.")
        if frac < 0.7:
            print(f"  [FAIL] only {frac:.0%} of the parental spread recovered -- the "
                  "model is over-shrinking. Both parents are ~41 mutations from a "
                  "typical segregant, so this is the extrapolation the landmarks "
                  "also live in.")
        else:
            print(f"  [PASS] parental spread largely recovered")
    return out


def check_background_informative(sfa, cand: pd.DataFrame, pi, ena1: int) -> None:
    """Warn if the chosen background zeroes out attributions on these targets.

    With a delta (one-hot) pi, ``f_x(S)`` substitutes the reference allele
    everywhere outside ``S``, so at any locus where the target already matches the
    reference the two terms of ``f_x(S u i) - f_x(S)`` are identical and phi_i is
    EXACTLY zero. The landmarks are all ENA1(RM), which is what the RM parental
    background carries at ENA1 -- so an RM reference makes phi_ENA1 vanish and
    silently changes the question to "which substitutions lift this above the RM
    parent". A frequency background (empirical / uniform) keeps ENA1 informative.
    """
    pi_np = _to_np(pi)
    delta_rows = int((pi_np.max(axis=1) > 1.0 - 1e-9).sum())
    if delta_rows == 0:
        return

    letters = list(sfa.alphabet)
    ref = "".join(letters[int(i)] for i in pi_np.argmax(axis=1))
    n_dead = [sum(1 for j, ch in enumerate(str(s)) if ch == ref[j])
              for s in cand["seq"]]
    print(f"\n  !! background is a delta at {delta_rows}/{sfa.L} loci")
    print(f"     phi_i == 0 exactly wherever the target matches it: "
          f"{min(n_dead)}-{max(n_dead)} of {sfa.L} loci per landmark")
    if all(str(s)[ena1] == ref[ena1] for s in cand["seq"]):
        print(f"     including ENA1 (locus {ena1}) in EVERY landmark -- phi_ENA1 "
              "will be identically zero and the effect under study is invisible.")
        print("     use --shap-background empirical unless that is intended.")


def landmark_shap(sfa, cand: pd.DataFrame, pi, eta: float = 1.0,
                  background: str = "") -> dict:
    """First-order SHAP for each landmark: phi (n,L), posterior sd, local accuracy.

    ``covariance='diag'`` gives the per-locus posterior VARIANCES in O(L) instead
    of building the (L,L) grid and keeping its diagonal -- the same numbers, and
    the only thing needed for error bars.

    NOTE ON INTERPRETATION: these are first-order Shapley values, NOT ANOVA main
    effects. If the fitted kernel is epistatic (jenga / connectedness), each phi_i
    already contains locus i's share of every interaction it takes part in. That
    is the desired reading here -- a compensating locus gets credited for the
    compensation -- but it is not "epistasis-free".
    """
    banner("first-order SHAP on the landmarks")
    shap = sfa.shap
    seqs = [str(s) for s in cand["seq"]]
    n, L = len(seqs), sfa.L

    phi = np.full((n, L), np.nan)
    phi_sd = np.full((n, L), np.nan)
    fx = np.full(n, np.nan)
    resid = np.full(n, np.nan)

    phi0 = float(_to_np(shap.base_value(pi, eta=eta))) + sfa.y_offset
    print(f"  phi_0 (background expectation) = {phi0:+.4f}")

    for r, s in enumerate(seqs):
        p, var = shap.shap_values(s, pi, eta=eta, calc_covar=True,
                                  covariance="diag", use_double=True)
        phi[r] = _to_np(p)
        phi_sd[r] = np.sqrt(np.clip(_to_np(var), 0.0, None))
        fx[r] = predict(sfa, s)
        resid[r] = fx[r] - phi0 - phi[r].sum()
        print(f"  [{r + 1}/{n}] row {cand.index[r]}  y={cand['y'].iloc[r]:+.4f}  "
              f"f(x)={fx[r]:+.4f}  sum_phi={phi[r].sum():+.4f}  "
              f"local-accuracy residual={resid[r]:+.2e}", flush=True)

    worst = np.nanmax(np.abs(resid))
    print(f"\n  worst local-accuracy residual: {worst:.2e}")
    if worst > 1e-6:
        print("  !! local accuracy broken (phi_0 + sum phi_i should equal f(x)). "
              "Check eta == 1 and that pi is a valid product background.")
    else:
        print("  -> local accuracy holds; the decomposition is exact")

    return {"phi": phi, "phi_sd": phi_sd, "fx": fx, "resid": resid,
            "phi0": phi0, "rows": list(cand.index), "seqs": seqs,
            "y": cand["y"].to_numpy(), "background": background}


def check_model_sanity(sfa, df: pd.DataFrame, n_sample: int = 400,
                       strict: bool = True) -> bool:
    """Is the model a real function of the sequence, or a constant?

    A degenerate model passes every other check in this script: predictions near
    zero clear a high-fitness threshold of -0.08 trivially, local accuracy holds
    exactly (0 == 0 + sum of zeros), and the SHAP values look like a tidy tiny
    distribution. The failure mode seen in practice was a constant-zero
    predictor -- R^2 -4.29, which is exactly what a zero predictor scores here
    (MSE = Var(y) + mean(y)^2 = 0.0479 against Var(y) = 0.0091) -- caused by a
    checkpoint whose theta shape did not match the kernel the loader built.

    So compare predictions against measurements directly: spread, correlation,
    and offset. Cheap, and it catches the whole class of "model never loaded".
    """
    banner("model sanity: is f(x) actually a function of the sequence?")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    seqs = df["seq"].to_numpy()[idx]
    y = df["y"].to_numpy()[idx]

    pred = np.array([predict(sfa, str(sq)) for sq in seqs])

    sd_p, sd_y = float(pred.std()), float(y.std())
    r = float(np.corrcoef(pred, y)[0, 1]) if sd_p > 0 else 0.0
    ss_res = float(((y - pred) ** 2).mean())
    r2 = 1.0 - ss_res / float(y.var())

    print(f"  {len(idx)} random segregants")
    print(f"  measured y  : mean={y.mean():+.4f}  sd={sd_y:.4f}")
    print(f"  predicted   : mean={pred.mean():+.4f}  sd={sd_p:.4f}")
    print(f"  sd ratio    : {sd_p / max(sd_y, 1e-12):.4f}  (want ~1, not ~0)")
    print(f"  correlation : r={r:+.4f}")
    print(f"  R^2         : {r2:+.4f}")

    checks = [
        ("predictions vary with sequence", sd_p > 0.1 * sd_y,
         f"pred sd {sd_p:.4f} vs y sd {sd_y:.4f}"),
        ("predictions track measurements", r > 0.5, f"r={r:+.4f}"),
        ("prediction offset is sane", abs(pred.mean() - y.mean()) < 0.5 * sd_y,
         f"offset {pred.mean() - y.mean():+.4f}"),
    ]
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    if n_fail:
        print(f"\n  >>> MODEL IS NOT USABLE ({n_fail} failed checks).")
        if sd_p < 0.1 * sd_y:
            print("      Predictions are near-constant -- the model is returning its")
            print("      prior mean. Likely causes:")
            print("        * checkpoint/kernel mismatch: a Connectedness checkpoint has")
            print("          theta of shape (L, 1), but _build_model_from_params builds a")
            print("          GeneralProductKernel expecting (L, A+1). It does not raise,")
            print("          it just produces a meaningless kernel.")
            print("        * zero prior mean against non-zero y: the checkpoint carries")
            print("          no mean parameter, so y probably needs centering (or")
            print("          train_mean=True) before set_data.")
        if abs(pred.mean() - y.mean()) >= 0.5 * sd_y:
            print(f"      Predicted mean {pred.mean():+.4f} vs measured {y.mean():+.4f}:")
            print("      a mean/centering problem specifically.")
        print("\n      Every downstream check passes vacuously on a constant model:")
        print("      local accuracy holds (0 = 0 + sum of zeros), and the landmark")
        print("      gate passes because ~0 clears a threshold of -0.08. Do not read")
        print("      the attribution figures.")
        if strict:
            return False
    else:
        print("\n  -> model is a genuine function of the sequence; proceed")
    return True


def gate_landmark_predictions(sfa, cand: pd.DataFrame, meta: dict,
                              strict: bool = False) -> bool:
    """Check the model predicts the landmarks ABOVE the selection threshold.

    SHAP explains a PREDICTION, not a measurement. If f(x) sits near the
    ENA1(RM) group mean, the model considers the segregant unremarkable and the
    attribution is faithfully explaining something that is not the phenomenon
    under study. That produces plausible-looking panels with no bearing on why
    the measured fitness was high, which is worse than no panels.

    Returns True if the set is usable. With ``strict`` a failure is fatal.
    """
    banner("gate: does the model predict the landmarks as high-fitness?")
    thr = meta["thr"]
    rows = list(cand.index)
    fx = np.array([predict(sfa, str(sq)) for sq in cand["seq"]])

    print(f"  selection threshold: y >= {thr:+.4f}")
    print(f"  {'row':>8} {'measured y':>11} {'f(x)':>9} {'gap':>8}  verdict")
    for i, ymeas, p in zip(rows, cand["y"], fx):
        gap = ymeas - p
        print(f"  {i:>8} {ymeas:>+11.4f} {p:>+9.4f} {gap:>+8.4f}  "
              f"{'ok' if p >= thr else 'BELOW THRESHOLD'}")

    n_bad = int((fx < thr).sum())
    print(f"\n  {n_bad}/{len(fx)} landmarks predicted below the threshold that "
          "selected them")
    if n_bad:
        print("  !! for those rows SHAP will explain a prediction that is NOT high "
              "fitness. Do not read their panels as 'why this segregant escaped "
              "ENA1(RM)'.")
        print("     fix the model first (see the R^2 and parental-gap checks above) "
              "rather than interpreting these attributions.")
        if strict:
            print("     --strict-gate set -> stopping.")
            return False
    else:
        print("  -> all landmarks are predicted high; attributions address the "
              "intended question")
    return True


def _qtl_tick_labels(ena1: int):
    """(positions, labels) for the 16 focal QTLs, for use as sparse x-ticks."""
    qtl = sorted(GENE_BY_LOCUS)
    return qtl, [GENE_BY_LOCUS[j] for j in qtl]


def make_shap_panels(res: dict, eff: pd.DataFrame, ena1: int,
                     out_dir: Path, tag: str = "rm_high") -> Path:
    """Figure 2: one stacked panel per landmark, phi_i across all 83 loci.

    Stacked rather than gridded, sharing the x-axis, so a locus that contributes
    consistently across landmarks reads as a vertical stripe -- the comparison
    that matters when asking what substitutes for ENA1.
    """
    banner("figure: per-landmark SHAP panels")
    phi, sd = res["phi"], res["phi_sd"]
    n, L = phi.shape
    qtl_pos, qtl_lab = _qtl_tick_labels(ena1)
    e = eff.set_index("locus")

    fig, axes = plt.subplots(n, 1, figsize=(15.5, 1.75 * n + 2.2), sharex=True)
    if n == 1:
        axes = [axes]
    vmax = np.nanmax(np.abs(phi) + np.nan_to_num(sd)) * 1.15

    for r, ax in enumerate(axes):
        colors = np.where(phi[r] >= 0, C_BY, C_RM)
        ax.bar(np.arange(L), phi[r], yerr=sd[r], color=colors, width=0.78,
               error_kw=dict(lw=0.6, ecolor="0.35"))
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(ena1, color=C_RM, lw=1.0, ls=":", alpha=0.8)
        ax.set_ylim(-vmax, vmax)
        ax.set_ylabel("$\\phi_i$", fontsize=9)

        # Label the few loci doing the most work in THIS landmark.
        top = np.argsort(-np.abs(phi[r]))[:4]
        for j in top:
            ax.annotate(gene_label(int(j)), xy=(j, phi[r][j]),
                        xytext=(0, 9 if phi[r][j] >= 0 else -13),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color="0.15")

        ax.text(0.004, 0.94,
                f"row {res['rows'][r]}   measured y={res['y'][r]:+.4f}   "
                f"f(x)={res['fx'][r]:+.4f}   "
                f"$\\phi_0$+$\\Sigma\\phi_i$={res['phi0'] + phi[r].sum():+.4f}   "
                f"$\\phi_{{ENA1}}$={phi[r][ena1]:+.4f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(fc="w", ec="0.7", lw=0.5, alpha=0.9, pad=2.5))

    axes[-1].set_xticks(qtl_pos)
    axes[-1].set_xticklabels(qtl_lab, rotation=90, fontsize=8)
    for tick, j in zip(axes[-1].get_xticklabels(), qtl_pos):
        if j == ena1:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    axes[-1].set_xlim(-1, L)
    axes[-1].set_xlabel("locus (ticks = the 16 focal QTLs; unlabelled bars are the "
                        "other 67 loci)", fontsize=9)

    fig.suptitle(
        f"First-order SHAP per sequence -- cohort: {COHORTS[tag]['label']}\n"
        f"background = {res.get('background', 'n/a')}, so $\\phi_0$ = "
        f"{res['phi0']:+.4f} and $\\phi_0 + \\Sigma_i \\phi_i$ = f(x).  "
        "Error bars = GP posterior s.d.  Blue = raises fitness, red = lowers it.",
        fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"shap_panels_{tag}.png"
    fig.savefig(p, dpi=190)
    fig.savefig(out_dir / f"shap_panels_{tag}.pdf")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def make_global_shap(res: dict, ena1: int, out_dir: Path,
                     top_k: int = 25, tag: str = "rm_high") -> tuple[Path, pd.DataFrame]:
    """Figure 3: mean |phi_i| across the landmarks, plus the signed counterpart.

    Two panels on purpose. ``mean |phi_i|`` is the requested global importance --
    it survives sign cancellation, so a locus that helps some landmarks and hurts
    others still registers. But that is exactly why it cannot tell you DIRECTION,
    so the lower panel shows mean signed phi_i with SEM across the landmarks and
    the individual landmark values overlaid. Read together: the top panel ranks
    what matters, the bottom says which way and whether the landmarks agree.
    """
    banner("figure: global SHAP importance over the landmarks")
    phi = res["phi"]
    n, L = phi.shape

    mean_abs = np.nanmean(np.abs(phi), axis=0)
    mean_signed = np.nanmean(phi, axis=0)
    sem = np.nanstd(phi, axis=0) / max(1.0, math.sqrt(n))
    # Sign agreement: 1.0 = every landmark pushed the same way at this locus.
    with np.errstate(invalid="ignore"):
        agree = np.nanmax(np.stack([(phi > 0).mean(axis=0),
                                    (phi < 0).mean(axis=0)]), axis=0)

    tbl = pd.DataFrame({
        "locus": np.arange(L),
        "gene": [GENE_BY_LOCUS.get(j) for j in range(L)],
        "mean_abs_phi": mean_abs,
        "mean_signed_phi": mean_signed,
        "sem_phi": sem,
        "sign_agreement": agree,
        "is_focal_qtl": [j in GENE_BY_LOCUS for j in range(L)],
    }).sort_values("mean_abs_phi", ascending=False).reset_index(drop=True)

    print(f"  top 15 loci by mean |phi| over {n} landmarks:")
    print(tbl.head(15).to_string(index=False))

    sel = tbl.head(top_k)
    xs = np.arange(len(sel))
    # pd.isna, not truthiness: a NaN gene is a float and renders as "nan".
    labels = [f"L{int(l)}" if pd.isna(g) else str(g)
              for g, l in zip(sel["gene"], sel["locus"])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(11, 0.52 * len(sel)), 9),
                                   sharex=True)

    ax1.bar(xs, sel["mean_abs_phi"], color="#5B6C8F", width=0.72)
    ax1.set_ylabel("mean $|\\phi_i|$ over landmarks", fontsize=10)
    ax1.set_title(f"A  global importance across the {n} {COHORTS[tag]['short']} "
                  "sequences (magnitude only -- direction not implied)",
                  loc="left", fontsize=11, fontweight="bold")
    for i, (v, a) in enumerate(zip(sel["mean_abs_phi"], sel["sign_agreement"])):
        ax1.text(i, v, f"{a:.0%}", ha="center", va="bottom", fontsize=6.5,
                 color="0.35", rotation=90)
    ax1.set_ylim(0, float(sel["mean_abs_phi"].max()) * 1.30)   # headroom for labels
    ax1.text(0.995, 0.95, "% above each bar = sign agreement across landmarks",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8,
             bbox=dict(fc="w", ec="0.7", lw=0.5, alpha=0.9, pad=3))

    colors = np.where(sel["mean_signed_phi"] >= 0, C_BY, C_RM)
    ax2.bar(xs, sel["mean_signed_phi"], yerr=sel["sem_phi"], color=colors,
            width=0.72, error_kw=dict(lw=0.8, ecolor="0.3"))
    for i, locus in enumerate(sel["locus"]):
        vals = phi[:, int(locus)]
        ax2.scatter(np.full(len(vals), i)
                    + np.random.default_rng(3).normal(0, 0.07, len(vals)),
                    vals, s=13, color="0.2", alpha=0.65, zorder=5, lw=0)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("mean signed $\\phi_i$  (SEM; dots = landmarks)", fontsize=10)
    ax2.set_title("B  direction and agreement (same loci, same order)",
                  loc="left", fontsize=11, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=90, fontsize=8.5)
    for tick, locus in zip(ax2.get_xticklabels(), sel["locus"]):
        if int(locus) == ena1:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    ax2.set_xlabel(f"locus, ranked by mean $|\\phi_i|$ (top {len(sel)} of {L})",
                   fontsize=9)

    fig.suptitle(f"Global first-order SHAP importance -- {COHORTS[tag]['label']} "
                 f"({n} sequences)", fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"shap_global_{tag}.png"
    fig.savefig(p, dpi=190)
    fig.savefig(out_dir / f"shap_global_{tag}.pdf")
    plt.close(fig)
    print(f"  wrote {p}")
    return p, tbl


def make_cohort_comparison(results: dict, ena1: int, out_dir: Path,
                           top_k: int = 20) -> tuple[Path, pd.DataFrame]:
    """Compare per-locus attribution between the cohorts.

    The question: two groups reach the SAME measured fitness, one carrying the
    beneficial ENA1 allele and one the deleterious one. Where does the model put
    the credit differently?

    Two panels:

      A  per-locus dot plot, one dot per sequence, coloured by cohort. No bars --
         the raw spread is the point. Cohort means as black ticks.
      B  Delta = mean(phi | RM-high) - mean(phi | BY-matched), with combined SEM
         and a z label. ENA1's delta is an order of magnitude larger than
         everything else, so the y-axis is scaled to the OTHER loci and ENA1 is
         drawn clipped at the boundary with its true value printed -- otherwise
         it flattens every locus that actually answers the question.

    RM-typical is still computed and plotted in panel A, and its per-locus means
    stay in the CSV (mean_phi_rm_mean, delta_high_minus_typical) so the control
    contrast is available numerically. Earlier versions drew two phi-vs-phi
    scatters against the y=x diagonal; they were dropped because every locus
    except ENA1 sits at the origin at this scale, so the panels carried almost
    no information for the space they took.
    """
    banner("figure: cohort comparison")
    A_tag, B_tag, C_tag = "rm_high", "by_matched", "rm_mean"
    if A_tag not in results or B_tag not in results:
        print("  !! need both rm_high and by_matched; skipping")
        return None, pd.DataFrame()

    phiA, phiB = results[A_tag]["phi"], results[B_tag]["phi"]
    L = phiA.shape[1]
    nA, nB = phiA.shape[0], phiB.shape[0]
    has_C = C_tag in results

    def ms(phi):
        n = phi.shape[0]
        return np.nanmean(phi, axis=0), np.nanstd(phi, axis=0) / max(1.0, math.sqrt(n))

    mA, sA = ms(phiA)
    mB, sB = ms(phiB)
    delta = mA - mB
    sem = np.hypot(sA, sB)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sem > 0, delta / sem, 0.0)

    cols = {
        "locus": np.arange(L),
        "gene": [GENE_BY_LOCUS.get(j) for j in range(L)],
        "mean_phi_rm_high": mA, "sem_rm_high": sA,
        "mean_phi_by_matched": mB, "sem_by_matched": sB,
        "delta_rm_minus_by": delta, "delta_sem": sem, "delta_z": z,
        "is_focal_qtl": [j in GENE_BY_LOCUS for j in range(L)],
    }
    if has_C:
        mC, sC = ms(results[C_tag]["phi"])
        cols["mean_phi_rm_mean"] = mC
        cols["sem_rm_mean"] = sC
        cols["delta_high_minus_typical"] = mA - mC
    tbl = pd.DataFrame(cols)
    tbl = tbl.reindex(tbl["delta_rm_minus_by"].abs()
                      .sort_values(ascending=False).index).reset_index(drop=True)

    print("  top 15 loci by |delta| (RM-high minus BY-matched):")
    show = ["locus", "gene", "mean_phi_rm_high", "mean_phi_by_matched",
            "delta_rm_minus_by", "delta_z"]
    if has_C:
        show.insert(4, "mean_phi_rm_mean")
    print(tbl.head(15)[show].to_string(index=False))

    sel = tbl.head(top_k)
    loci = sel["locus"].astype(int).tolist()
    labels = [f"L{j}" if pd.isna(g) else str(g)
              for g, j in zip(sel["gene"], sel["locus"])]
    xs = np.arange(len(loci))
    rng = np.random.default_rng(5)

    fig = plt.figure(figsize=(max(13.5, 0.62 * len(loci)), 9.2))
    # bottom=0.16: the rotated gene labels plus the axis label need the room.
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.9], hspace=0.16,
                          left=0.075, right=0.98, top=0.885, bottom=0.16)

    # ---- A: dot plot, one dot per sequence ------------------------------- #
    ax = fig.add_subplot(gs[0])
    order = [A_tag, B_tag] + ([C_tag] if has_C else [])
    width = 0.24
    for k, tag in enumerate(order):
        phi = results[tag]["phi"]
        off = (k - (len(order) - 1) / 2) * width
        c = COHORTS[tag]["color"]
        for i, j in enumerate(loci):
            v = phi[:, j]
            ax.scatter(np.full(len(v), i + off) + rng.normal(0, 0.033, len(v)),
                       v, s=26, color=c, alpha=0.8, lw=0.4, ec="white", zorder=4,
                       label=COHORTS[tag]["short"] if i == 0 else None)
            mu = float(np.nanmean(v))
            ax.plot([i + off - 0.09, i + off + 0.09], [mu, mu], color="k",
                    lw=1.6, zorder=6)
    ax.axhline(0, color="k", lw=0.8)
    for i in range(len(loci) - 1):
        ax.axvline(i + 0.5, color="0.9", lw=0.6, zorder=0)
    ax.set_xticks(xs)
    plt.setp(ax.get_xticklabels(), visible=False)   # panel B carries the labels
    ax.set_xlim(-0.6, len(loci) - 0.4)
    ax.set_ylabel("$\\phi_i$ (one dot per sequence)", fontsize=10)
    ax.set_title("A  per-locus attribution by cohort   "
                 "(black ticks = cohort means; loci ranked by "
                 "|RM-high $-$ BY-matched|)",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, ncol=len(order), loc="upper right")

    # ---- B: the difference, ENA1 clipped so the rest is readable --------- #
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    d = sel["delta_rm_minus_by"].to_numpy()
    e = sel["delta_sem"].to_numpy()
    zz = sel["delta_z"].to_numpy()
    others = np.array([j != ena1 for j in loci])
    if others.any():
        ylim = float(np.nanmax(np.abs(d[others]) + e[others])) * 1.45
    else:
        ylim = float(np.nanmax(np.abs(d) + e)) * 1.2
    ylim = max(ylim, 1e-6)

    pc = np.where(d >= 0, COHORTS[A_tag]["color"], COHORTS[B_tag]["color"])
    for i in range(len(d)):
        clipped = abs(d[i]) > ylim
        yv = np.clip(d[i], -ylim * 0.94, ylim * 0.94)
        if not clipped:
            ax2.errorbar([i], [d[i]], yerr=[e[i]], fmt="none", ecolor="0.35",
                         lw=1.0, capsize=2.5, zorder=3)
        ax2.scatter([i], [yv], s=80 if clipped else 70, color=pc[i], ec="k",
                    lw=0.7, zorder=5,
                    marker="v" if (clipped and d[i] < 0) else
                           "^" if clipped else "o")
        if clipped:
            ax2.annotate(f"{d[i]:+.3f}\n(off scale)", (i, yv),
                         xytext=(0, 14 if d[i] < 0 else -22),
                         textcoords="offset points", ha="center", fontsize=7.5,
                         fontweight="bold")
        elif abs(zz[i]) >= 2:
            ax2.annotate(f"z={zz[i]:.1f}", (i, d[i]),
                         xytext=(0, 11 if d[i] >= 0 else -15),
                         textcoords="offset points", ha="center", fontsize=7)
    ax2.set_ylim(-ylim, ylim)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("$\\Delta$ mean $\\phi_i$\nRM-high $-$ BY-matched",
                   fontsize=9.5)
    ax2.set_title("B  where the cohorts differ   (red above 0 = credited more in "
                  "RM-high, blue below 0 = more in BY-matched; y-axis scaled to "
                  "the non-ENA1 loci)",
                  loc="left", fontsize=11, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=90, fontsize=8.5)
    for tick, j in zip(ax2.get_xticklabels(), loci):
        if j == ena1:
            tick.set_color(COHORTS[A_tag]["color"])
            tick.set_fontweight("bold")
    ax2.set_xlabel(f"locus, ranked by |RM-high $-$ BY-matched| "
                   f"(top {len(loci)} of {L})", fontsize=9.5)

    yA, yB = results[A_tag]["y"], results[B_tag]["y"]
    fig.suptitle(
        "How do ENA1$^{RM}$ and ENA1$^{BY}$ sequences reach the same fitness?\n"
        f"RM-high n={nA}, y {yA.min():+.3f} to {yA.max():+.3f}   |   "
        f"BY-matched n={nB}, y {yB.min():+.3f} to {yB.max():+.3f}   |   "
        f"median |$\\Delta y$| between matched pairs "
        f"{np.median(np.abs(np.sort(yA) - np.sort(yB))):.4f}",
        fontsize=12.5, y=0.982)

    out_dir.mkdir(parents=True, exist_ok=True)
    ppng = out_dir / "shap_cohort_comparison.png"
    fig.savefig(ppng, dpi=190)
    fig.savefig(out_dir / "shap_cohort_comparison.pdf")
    plt.close(fig)
    print(f"  wrote {ppng}")
    return ppng, tbl


def save_shap_tables(res: dict, out_dir: Path, tag: str = "rm_high") -> None:
    """Long-format phi table (one row per landmark x locus) + the summary."""
    n, L = res["phi"].shape
    rows = []
    for r in range(n):
        for j in range(L):
            rows.append({
                "row_in_csv": res["rows"][r],
                "measured_y": res["y"][r],
                "f_x": res["fx"][r],
                "phi_0": res["phi0"],
                "locus": j,
                "gene": GENE_BY_LOCUS.get(j),
                "phi": res["phi"][r, j],
                "phi_post_sd": res["phi_sd"][r, j],
            })
    p = out_dir / f"shap_values_{tag}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"  {p}")


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Self-contained parser. analysis.py's add_common_args is not used."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = ap.add_argument_group("data / model")
    g.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="directory with qtls_li_hq.csv / .seqs.txt / .seqs_key.txt")
    g.add_argument("--params-path", type=Path, default=None,
                   help="pretrained checkpoint (.pth). REQUIRED unless --skip-shap. "
                        "The kernel family is inferred from the checkpoint's theta "
                        "shape, so no --kernel flag is needed or accepted.")
    g.add_argument("--project-root", type=Path, default=None,
                   help="repo root (default: inferred; must contain modules/ and epik/)")
    g.add_argument("--out-dir", type=Path, default=REPO_ROOT / "output/ENA1_analysis")
    g.add_argument("--gpu", action="store_true", help="use CUDA if available")
    g.add_argument("--threads", type=int, default=0,
                   help="torch.set_num_threads(N); 0 leaves it alone")

    m = ap.add_argument_group("model handling")
    m.add_argument("--no-center-y", action="store_true",
                   help="do NOT subtract mean(y) before conditioning. EpiK uses a "
                        "zero prior mean and these checkpoints carry no mean "
                        "parameter, so leaving y uncentered (mean -0.197) makes the "
                        "model predict ~0 everywhere. Centering is the default for "
                        "that reason; this flag is for diagnosis.")
    m.add_argument("--yvar-scale", type=float, default=1.0,
                   help="multiply y_var before conditioning. Mean y_var (0.0276) is "
                        "3x Var(y) (0.0091), which cannot hold under an additive "
                        "noise model, so the stated noise is inflated. Relative "
                        "precision is what matters and a scale preserves it; sweep "
                        "this against held-out fit if the model over-smooths.")
    m.add_argument("--allow-degenerate", action="store_true",
                   help="proceed even if the model fails the sanity check. "
                        "Debugging only -- attributions will be numerical noise.")
    m.add_argument("--require-gpu", action="store_true",
                   help="abort if --gpu was requested but the model landed on CPU")

    k = ap.add_argument_group("landmarks / cohorts")
    k.add_argument("--high-quantile", type=float, default=0.90,
                   help="fitness quantile defining 'globally high'")
    k.add_argument("--min-post-prob", type=float, default=0.75,
                   help="min P(true fitness > threshold), from y_var")
    k.add_argument("--rm-relative", action="store_true",
                   help="threshold within the ENA1(RM) subgroup instead of globally")
    k.add_argument("--precision-quantile", type=float, default=0.25,
                   help="for the RM-typical cohort, restrict to the most precisely "
                        "measured fraction of the RM subgroup")
    k.add_argument("--no-figure", action="store_true",
                   help="skip the landmark-selection figure")

    h = ap.add_argument_group("SHAP")
    h.add_argument("--skip-shap", action="store_true",
                   help="stop after landmark selection; no model, no SHAP")
    h.add_argument("--shap-background", default="empirical",
                   choices=["empirical", "uniform", "rm_wildtype", "by_wildtype"],
                   help="attribution reference. Default 'empirical' uses per-locus "
                        "allele frequencies. AVOID the parental (delta) backgrounds "
                        "for these targets: a delta pi gives phi_i == 0 exactly "
                        "wherever the target matches the reference, and every "
                        "landmark matches the RM parent AT ENA1, so phi_ENA1 would "
                        "be identically zero. phi_0 under both parental backgrounds "
                        "is reported as a validation regardless.")
    h.add_argument("--shap-top-k", type=int, default=25,
                   help="loci shown in each cohort's global importance figure")
    h.add_argument("--compare-top-k", type=int, default=20,
                   help="loci shown in panels A and B of the cohort comparison")
    h.add_argument("--eta", type=float, default=1.0,
                   help="background-channel scale; local accuracy assumes 1.0")
    h.add_argument("--strict-gate", action="store_true",
                   help="abort if the model predicts any landmark below the "
                        "selection threshold")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    if a.project_root:
        globals()["REPO_ROOT"] = Path(a.project_root).resolve()

    # ---------------- Stage 1: landmark discovery (no model) ---------------- #
    banner(f"ENA1 analysis -- {a.data_dir}")
    df = load_segregants(a.data_dir / "qtls_li_hq.csv")
    X = genotype_matrix(df["seq"])
    y = df["y"].to_numpy()

    mapping = recover_landscape_mapping(
        a.data_dir / "qtls_li_hq.seqs.txt",
        a.data_dir / "qtls_li_hq.seqs_key.txt")

    eff = marginal_effects(X, y)
    ena1 = identify_ena1(eff, mapping)
    if ena1 is None:
        print("\nENA1 not identified. Stopping.")
        return 1

    row = eff.loc[eff["locus"] == ena1].iloc[0]
    beneficial_is_b = bool(row["delta_B_minus_A"] > 0)
    print(f"\n  at ENA1 (locus {ena1}): "
          f"'{'B' if beneficial_is_b else 'A'}' = beneficial (BY), "
          f"'{'A' if beneficial_is_b else 'B'}' = deleterious (RM)")
    if ("B" if beneficial_is_b else "A") != BY_LETTER:
        print(f"  !! polarity disagrees with BY_LETTER={BY_LETTER!r}; the "
              "parental-genotype backgrounds below would be inverted")

    inter = test_ena1_interactions(X, y, ena1, beneficial_is_b, mapping)
    ld = ld_check(X, eff)

    out, cand, meta = select_landmarks(
        df, X, ena1, beneficial_is_b, eff,
        a.high_quantile, a.min_post_prob, a.rm_relative)

    if not a.no_figure and len(cand):
        make_figure(out, cand, X, eff, ena1, beneficial_is_b, meta, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    banner("written (stage 1)")
    for name, obj, idx in [("ena1_landmarks.csv", cand, True),
                           ("ena1_marginal_effects.csv", eff, False),
                           ("ena1_conditional_effects.csv", inter, False),
                           ("ena1_ld_pairs.csv", ld, False)]:
        p = out_dir / name
        obj.to_csv(p, index=idx, index_label="row_in_csv" if idx else None)
        print(f"  {p}")

    if a.skip_shap:
        print("\n--skip-shap set; stopping after landmark selection.")
        return 0
    if not len(cand):
        print("\nNo landmarks found; nothing to attribute.")
        return 0

    # ---------------- Stage 2: GP + first-order SHAP ------------------------ #
    sfa = load_model_and_shap(a, df, X)
    if sfa is None:
        print("\nModel unavailable; stage 1 outputs are still written above.")
        return 1

    check_base_values(sfa)

    if not check_model_sanity(sfa, df, strict=not a.allow_degenerate):
        print("\nStopping: fix the model before interpreting anything. "
              "Pass --allow-degenerate to override (for debugging only).")
        return 3

    counts = None
    if a.shap_background == "empirical":
        import torch
        # Per-locus allele counts from the genotype matrix already in hand
        # (X == 1 means the 'B' allele), ordered to match sfa.alphabet.
        counts = torch.zeros(sfa.L, sfa.A, dtype=torch.float64)
        for j, letter in enumerate(sfa.alphabet):
            hit = (X == 1) if letter == "B" else (X == 0)
            counts[:, j] = torch.as_tensor(hit.sum(axis=0).astype(float),
                                           dtype=torch.float64)
    pi = make_pi(sfa, a.shap_background, empirical_counts=counts)
    banner(f"SHAP background: {a.shap_background}")
    freq = _to_np(pi)
    msg = f"  pi: min={freq.min():.3f} max={freq.max():.3f}"
    if "B" in sfa.alphabet:
        bcol = freq[:, sfa.alphabet.index("B")]
        msg += (f"   B-allele frequency: mean={bcol.mean():.3f} "
                f"range=[{bcol.min():.3f}, {bcol.max():.3f}]")
    print(msg)
    check_background_informative(sfa, cand, pi, ena1)

    if not gate_landmark_predictions(sfa, cand, meta, strict=a.strict_gate):
        return 2

    cohorts = select_cohorts(out, cand, meta, precision_q=a.precision_quantile)

    results, summaries = {}, {}
    for tag, df_c in cohorts.items():
        if not len(df_c):
            print(f"\n  cohort {tag} is empty; skipping")
            continue
        banner(f"cohort {COHORTS[tag]['short']}  ({COHORTS[tag]['note']})")
        res = landmark_shap(sfa, df_c, pi, eta=a.eta,
                            background=a.shap_background)
        results[tag] = res
        make_shap_panels(res, eff, ena1, out_dir, tag=tag)
        _, tbl = make_global_shap(res, ena1, out_dir, top_k=a.shap_top_k, tag=tag)
        summaries[tag] = tbl

    _, cmp_tbl = make_cohort_comparison(results, ena1, out_dir,
                                        top_k=a.compare_top_k)

    banner("written (stage 2)")
    for tag, res in results.items():
        save_shap_tables(res, out_dir, tag=tag)
        p = out_dir / f"shap_summary_{tag}.csv"
        summaries[tag].to_csv(p, index=False)
        print(f"  {p}")
    if len(cmp_tbl):
        p = out_dir / "shap_cohort_comparison.csv"
        cmp_tbl.to_csv(p, index=False)
        print(f"  {p}")

    banner("reading the panels")
    print("  - phi_i are first-order Shapley values, NOT ANOVA main effects. On an "
          "epistatic kernel each phi_i already carries locus i's share of the "
          "interactions it participates in -- that is the intended reading here.")
    print("  - under a frequency background phi_ENA1 should be large and negative in "
          "every panel; the explanation is whatever carries positive phi against it.")
    # Loci in tight LD cannot be separated by interventional SHAP: credit splits
    # arbitrarily within a block, so report them together.
    if len(ld):
        strong = ld[ld["r"].abs() > 0.7]
        if len(strong):
            print(f"  - {len(strong)} locus pairs have |r| > 0.7. Interventional SHAP "
                  "splits credit arbitrarily inside a linked block, so read these as "
                  "blocks rather than as individual loci:")
            for _, r in strong.head(10).iterrows():
                gi = r["gene_i"] if not pd.isna(r["gene_i"]) else f"L{int(r['locus_i'])}"
                gj = r["gene_j"] if not pd.isna(r["gene_j"]) else f"L{int(r['locus_j'])}"
                print(f"      {gi} / {gj}   r={r['r']:+.3f}")
    print("  - cross-check the top loci against panel F of the landmark figure: if "
          "SHAP credits a locus the landmarks do not actually share, or ignores one "
          "they all carry, that disagreement is the result worth chasing.")
    print("  - sign agreement in the global figure separates a shared mechanism "
          "(high agreement) from landmark-specific rescue (near 50%).")
    print("  - the cohort comparison is where the answer lives. Panel C is the "
          "paper's Figure 4E computed through the model: loci ON the diagonal "
          "contribute the same way regardless of ENA1 state, loci ABOVE it are "
          "doing extra work in the RM background. HAL9, PHO84 and MKT1 are the "
          "paper's predictions for what should sit above the line.")
    print("  - RM-typical is the control: a locus that differs between RM-high and "
          "BY-matched but ALSO differs the same way in RM-typical is an "
          "ENA1-state effect, not a high-fitness effect.")
    print(f"  - three checkpoint replicates exist (.1/.2/.3). Rerun with "
          "--params-path pointing at each; loci whose sign flips between "
          "replicates are not robust.")
    return 0


if __name__ == "__main__":
    sys.exit(main())