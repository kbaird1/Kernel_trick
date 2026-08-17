#!/usr/bin/env python
"""
plot_shap_results.py
====================
Figures for an ``analysis.py`` output directory at LARGE L, reading the CSVs it already
wrote -- so this is seconds to run and safe to iterate on without refitting.

Why not ``analysis.py --make-plots``
------------------------------------
Its built-in plots size the canvas at ``n_tuples * 0.35`` inches, which is fine at L=4
(10 tuples -> 8 in) and absurd at L=36 (666 tuples -> 233 in, a ~30,000 px strip). More
importantly, 630 bars side by side is the wrong encoding: pairwise interactions over L
positions are a MATRIX, and a triangular heatmap shows all of them at once and legibly.

What it draws
-------------
1. ``global_heatmap_<col>.png`` -- main effects as a bar strip on top, the full pairwise
   interaction matrix as a triangular heatmap below, sharing the position axis. Every pair
   is visible; hot cells are the epistatic ones. Axis ticks are RBD positions, read from
   the ``*_positions.csv`` sidecar, not 0-based indices.
2. ``global_top<N>.png`` -- the N strongest interactions, all three indices side by side,
   for reading off actual numbers. SII vs dividend on the same tuples is the
   smearing-vs-pure contrast.
3. ``local_<seq>_top<N>.png`` -- per target, the N largest |attribution| terms, SIGNED,
   with posterior-sd error bars. Terms whose |z| < 2 are drawn hollow: the GP does not
   resolve them from zero, and at 666 tuples some will look large by chance.

It also prints a text summary: the order-1 vs order-2 attribution split (how epistatic
the landscape is), the top hits per index, and the ``mean_signed`` diagnostic.

Usage
-----
    python scripts/plot_shap_results.py --out-dir output/libraryM_ACE2 \\
        --positions data/library_M/MABAa_ACE2.seqy_positions.csv

    python scripts/plot_shap_results.py --out-dir output/libraryM_ACE2 \\
        --positions data/library_M/MABAa_ACE2.seqy_positions.csv \\
        --value-col dividend_sobol --top-n 40
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_positions(path, n_positions):
    """seq_index -> RBD position label. Falls back to plain indices if absent."""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        lab = {int(r.seq_index): str(r.rbd_position) for r in df.itertuples()}
        return [lab.get(i, str(i)) for i in range(n_positions)]
    print("[plot] no --positions sidecar; labelling axes with 0-based indices")
    return [str(i) for i in range(n_positions)]


def parse_tuple(s):
    """'3x17' -> (3, 17); '3' -> (3,).  Matches analysis.py's 'x'.join encoding."""
    return tuple(int(p) for p in str(s).split("x"))


def summarize(pos_df, inter_df, value_col, labels, top_n):
    """Print the text summary: epistasis split, top hits, mean_signed diagnostic."""
    singles = inter_df[inter_df["order"] == 1]
    pairs = inter_df[inter_df["order"] == 2]

    print("\n" + "=" * 74)
    print("GLOBAL SUMMARY")
    print("=" * 74)
    print(f"  positions: {len(pos_df)}   order-1 terms: {len(singles)}   "
          f"order-2 terms: {len(pairs)}")

    # How much of the total dividend mass sits in interactions rather than main effects.
    # Dividends are the right index for this: they do NOT smear high orders onto subsets,
    # so the split is a real order decomposition rather than double counting.
    m1 = float(singles["dividend_mean_abs"].sum())
    m2 = float(pairs["dividend_mean_abs"].sum())
    if m1 + m2 > 0:
        print(f"  dividend mass: order-1 {m1:.3g} ({m1/(m1+m2):.1%})   "
              f"order-2 {m2:.3g} ({m2/(m1+m2):.1%})")
        print("    (order-2 share is a rough 'how epistatic is this landscape' readout;"
              "\n     it is mass in the tuples ENUMERATED, so raising --interaction-order"
              "\n     would move it -- not an absolute epistasis fraction.)")

    print(f"\n  top {min(top_n, len(pairs))} pairwise by {value_col}:")
    for r in pairs.nlargest(min(top_n, len(pairs)), value_col).itertuples():
        i, j = parse_tuple(r.tuple)
        print(f"    {labels[i]:>4s} x {labels[j]:<4s}  {value_col}={getattr(r, value_col):+.4g}"
              f"   sii={r.sii_mean_abs:.4g}  dividend={r.dividend_mean_abs:.4g}")

    print(f"\n  top 10 positions by mean|phi_i|:")
    for r in pos_df.nlargest(10, "mean_abs_shapley").itertuples():
        print(f"    {labels[int(r.position)]:>4s}  {r.mean_abs_shapley:.4g}")

    # mean_signed is ~0 only when the global targets are drawn from the SAME product
    # distribution used as the background. With an empirical background and targets
    # subsampled from a designed library, positions are correlated and it will NOT be ~0
    # -- which is exactly what this diagnostic is for.
    ms = inter_df["dividend_mean_signed"].abs()
    ma = inter_df["dividend_mean_abs"].abs().max()
    print(f"\n  mean_signed diagnostic: max|mean_signed| = {ms.max():.4g} "
          f"({ms.max()/(ma+1e-30):.1%} of max|a|)")
    if ms.max() > 0.15 * ma:
        print("    -> NOT ~0, so the target sample is far from the background product"
              "\n       distribution. dividend_sobol is therefore a variance-across-your-"
              "\n       sample, NOT a formal Sobol sensitivity index. Expected here.")
    else:
        print("    -> ~0, consistent with targets drawn from the background.")
    print("=" * 74 + "\n")


def plot_global_heatmap(pos_df, inter_df, value_col, labels, out_dir, signed=False):
    """Main-effect strip + full pairwise triangular heatmap, sharing the position axis."""
    L = len(pos_df)
    M = np.full((L, L), np.nan)
    for r in inter_df[inter_df["order"] == 2].itertuples():
        i, j = parse_tuple(r.tuple)
        v = getattr(r, value_col)
        M[max(i, j), min(i, j)] = v          # lower triangle only
    main = pos_df.sort_values("position")["mean_abs_shapley"].to_numpy()

    fig = plt.figure(figsize=(max(9, 0.34 * L + 3), max(9, 0.34 * L + 2.4)), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 6], width_ratios=[40, 1],
                          hspace=0.05, wspace=0.04)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(np.arange(L), main, color="#c0504d", width=0.8)
    ax_bar.set_xlim(-0.5, L - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("mean\n|phi_i|", fontsize=8)
    ax_bar.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax_bar.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    if signed:
        lim = np.nanmax(np.abs(M)) or 1.0
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, origin="upper")
    else:
        im = ax.imshow(M, cmap="magma_r", origin="upper")
    ax.set_xticks(np.arange(L)); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticks(np.arange(L)); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("RBD position"); ax.set_ylabel("RBD position")
    ax.set_title(f"Pairwise interaction: {value_col}  (all {int((L*(L-1))/2)} pairs; "
                 "upper triangle blank by symmetry)", fontsize=10)

    cax = fig.add_subplot(gs[1, 1])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)

    fn = os.path.join(out_dir, f"global_heatmap_{value_col}.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fn}")


def plot_global_top(inter_df, labels, out_dir, value_col, top_n):
    """The N strongest interactions, all three indices, for reading off numbers."""
    pairs = inter_df[inter_df["order"] == 2]
    top = pairs.nlargest(min(top_n, len(pairs)), value_col).iloc[::-1]
    names = [f"{labels[parse_tuple(t)[0]]}x{labels[parse_tuple(t)[1]]}"
             for t in top["tuple"]]
    y = np.arange(len(top)); h = 0.27

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))), dpi=140)
    ax.barh(y - h, top["sii_mean_abs"], h, label="SII (smears down)", color="#8064a2")
    ax.barh(y, top["dividend_mean_abs"], h, label="dividend (pure)", color="#9bbb59")
    ax.barh(y + h, top["dividend_sobol"], h, label="dividend variance", color="#2c7fb8")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("global interaction magnitude")
    ax.set_title(f"Top {len(top)} pairwise interactions by {value_col}", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fn = os.path.join(out_dir, f"global_top{len(top)}.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fn}")


def plot_local(path, labels, out_dir, top_n, z_thresh=2.0):
    """Per target: the N largest |dividend| terms, signed, with posterior-sd error bars."""
    df = pd.read_csv(path)
    seq = os.path.basename(path)[len("local_"):-len(".csv")]
    have_sd = "dividend_sd" in df.columns

    df = df.assign(_mag=df["dividend"].abs())
    top = df.nlargest(min(top_n, len(df)), "_mag").iloc[::-1]
    names = []
    for t, o in zip(top["tuple"], top["order"]):
        pp = parse_tuple(t)
        names.append(labels[pp[0]] if o == 1 else
                     "x".join(labels[p] for p in pp))
    y = np.arange(len(top))
    err = top["dividend_sd"] if have_sd else None

    # Hollow bars where the GP does not resolve the term from zero. At 666 tuples a few
    # will look large by chance, so this is the difference between a hit and noise.
    if have_sd:
        resolved = top["dividend_z"].abs() >= z_thresh
    else:
        resolved = pd.Series(True, index=top.index)
    face = ["#4f81bd" if r else "none" for r in resolved]
    edge = ["#4f81bd"] * len(top)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))), dpi=140)
    ax.barh(y, top["dividend"], 0.7, xerr=err, capsize=2,
            color=face, edgecolor=edge, linewidth=1.1, error_kw=dict(lw=0.8))
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("dividend (signed contribution to this sequence's phenotype)")
    n_res = int(resolved.sum())
    ax.set_title(f"{seq}\ntop {len(top)} terms by |dividend|"
                 + (f"   (filled = |z| >= {z_thresh}: {n_res}/{len(top)} resolved; "
                    "error bars = posterior sd)" if have_sd else ""),
                 fontsize=9)
    fig.tight_layout()
    fn = os.path.join(out_dir, f"local_{seq}_top{len(top)}.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fn}"
          + (f"   ({n_res}/{len(top)} terms with |z| >= {z_thresh})" if have_sd else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="an analysis.py output directory (read AND written)")
    ap.add_argument("--positions", default=None,
                    help="the prepare_libraryM *_positions.csv sidecar, so axes are "
                         "labelled with RBD positions instead of 0-based indices")
    ap.add_argument("--value-col", default="dividend_mean_abs",
                    help="which global column drives the heatmap / ranking "
                         "(dividend_mean_abs, dividend_sobol, sii_mean_abs, "
                         "faith_shap_mean_abs, dividend_mean_signed)")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--z-thresh", type=float, default=2.0,
                    help="|z| below which a local term is drawn hollow (unresolved)")
    ap.add_argument("--no-local", action="store_true", help="skip the per-target figures")
    args = ap.parse_args()

    od = args.out_dir
    pos_df = pd.read_csv(os.path.join(od, "global_position_importance.csv"))
    inter_df = pd.read_csv(os.path.join(od, "global_interactions.csv"))
    if args.value_col not in inter_df.columns:
        raise SystemExit(f"--value-col '{args.value_col}' not in global_interactions.csv "
                         f"(have: {[c for c in inter_df.columns if c != 'tuple']})")

    labels = load_positions(args.positions, len(pos_df))
    summarize(pos_df, inter_df, args.value_col, labels, args.top_n)

    signed = "signed" in args.value_col
    plot_global_heatmap(pos_df, inter_df, args.value_col, labels, od, signed=signed)
    plot_global_top(inter_df, labels, od, args.value_col, args.top_n)

    if not args.no_local:
        for p in sorted(glob.glob(os.path.join(od, "local_*.csv"))):
            plot_local(p, labels, od, args.top_n, z_thresh=args.z_thresh)


if __name__ == "__main__":
    main()