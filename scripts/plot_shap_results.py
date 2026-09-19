#!/usr/bin/env python
"""
plot_shap_results.py
====================
Figures for an ``analysis.py`` output directory at LARGE L, reading the CSVs it already
wrote -- so this runs in seconds and is safe to iterate on without refitting.

Why not ``analysis.py --make-plots``
------------------------------------
Its built-in plots size the canvas at ``n_tuples * 0.35`` inches: fine at L=4 (10 tuples
-> 8 in), absurd at L=36 (666 tuples -> 233 in). And 630 bars side by side is the wrong
encoding -- pairwise interactions over L positions are a MATRIX.

Figures
-------
``global_heatmap_<col>.png``
    Order-1 terms as a bar strip on top, the full pairwise matrix as a triangular heatmap
    below, sharing the position axis. Both panels use the SAME index by default (see
    --strip-index): mixing Shapley main effects with dividend interactions in one figure
    would be misleading, since a({i}) and phi_i are different quantities.

``global_top<N>.png``
    The N strongest interactions with SII and dividend side by side -- the smearing-vs-pure
    contrast on identical tuples.

``global_orders.png``
    (left) total attribution mass per interaction order -- how epistatic the landscape is.
    (right) per-position mass split by order -- which positions are hubs of high-order
    structure rather than just large on their own.

``global_highorder_top<N>.png``
    An UpSet-style membership grid: rows are the strongest tuples of order >= --min-order,
    columns are the positions they involve, with a magnitude bar alongside. This is the one
    encoding that scales past order 2 -- a heatmap cannot show a rank-3 tensor, but a
    membership grid shows tuples of ANY order in one readable panel, and reveals which
    positions recur across the top high-order terms.

``local_<seq>_top<N>.png``
    Per target, the N largest |attribution| terms of ANY order, SIGNED, with posterior-sd
    error bars. Hollow bars are terms with |z| < --z-thresh: the GP does not resolve them
    from zero, and at hundreds of tuples some will look large by chance.

NOTE ON ORDERS: these plots show whatever orders are in the CSV. If you ran
``--interaction-order 2`` there are no order-3 rows to find; re-run the analysis with
``--interaction-order 3`` (7,806 tuples at L=36, ~12x the interpretation cost) to populate
them. The high-order figure says so explicitly rather than drawing an empty panel.

Usage
-----
    python scripts/plot_shap_results.py --out-dir output/libraryM_ACE2 \\
        --positions data/library_M/MABAa_ACE2.seqy_positions.csv

    python scripts/plot_shap_results.py --out-dir output/libraryM_ACE2 \\
        --positions data/library_M/MABAa_ACE2.seqy_positions.csv \\
        --value-col dividend_sobol --top-n 40 --min-order 3
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Human-readable axis/colorbar text per global column, so a figure never shows a bare
# CSV header. Keys are the columns analysis.py writes to global_interactions.csv.
COL_LABEL = {
    "dividend_mean_abs":       "mean$_x$ |a(T)|   (Harsanyi dividend, pure interaction)",
    "dividend_sobol":          "Var$_x$ a(T)   (dispersion of the dividend across targets)",
    "dividend_mean_signed":    "mean$_x$ a(T)   (signed; ~0 iff targets ~ background)",
    "sii_mean_abs":            "mean$_x$ |SII(T)|   (smears high orders onto subsets)",
    "faith_shap_mean_abs":     "mean$_x$ |E$_S$(T)|   (Faith-Shap)",
}


COL_SHORT = {
    "dividend_mean_abs":    "mean$_x$ |a(T)|",
    "dividend_sobol":       "Var$_x$ a(T)",
    "dividend_mean_signed": "mean$_x$ a(T)",
    "sii_mean_abs":         "mean$_x$ |SII(T)|",
    "faith_shap_mean_abs":  "mean$_x$ |E$_S$(T)|",
}


def col_label(col):
    return COL_LABEL.get(col, col)


def col_short(col):
    return COL_SHORT.get(col, col)


def load_positions(path, n_positions):
    """seq_index -> RBD position label. Falls back to 0-based indices if absent."""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        lab = {int(r.seq_index): str(r.rbd_position) for r in df.itertuples()}
        return [lab.get(i, str(i)) for i in range(n_positions)]
    print("[plot] no --positions sidecar; labelling axes with 0-based indices")
    return [str(i) for i in range(n_positions)]


def parse_tuple(s):
    """'3x17' -> (3, 17); '3' -> (3,).  Matches analysis.py's 'x'.join encoding."""
    return tuple(int(p) for p in str(s).split("x"))


def tuple_label(t, labels):
    return "\u00d7".join(labels[p] for p in t)


def orders_present(inter_df):
    return sorted(inter_df["order"].unique())


# ------------------------------------------------------------------ #
# text summary
# ------------------------------------------------------------------ #

def summarize(pos_df, inter_df, value_col, labels, top_n):
    ords = orders_present(inter_df)
    print("\n" + "=" * 78)
    print("GLOBAL SUMMARY")
    print("=" * 78)
    print(f"  positions: {len(pos_df)}   orders present in CSV: {ords}")
    for k in ords:
        print(f"    order {k}: {int((inter_df['order'] == k).sum())} terms")
    if max(ords) < 3:
        print("  NOTE: no order-3+ terms in this run. Re-run analysis.py with"
              "\n        --interaction-order 3 to populate them (at L=36 that is 7,806"
              "\n        tuples, ~12x the interpretation cost of order 2).")

    # Dividends are the right index for an order split: they do NOT smear high orders onto
    # subsets, so mass per order is a real decomposition rather than double counting.
    mass = {k: float(inter_df.loc[inter_df["order"] == k, "dividend_mean_abs"].sum())
            for k in ords}
    tot = sum(mass.values()) or 1.0
    print("\n  dividend mass by order (of the tuples ENUMERATED, not an absolute"
          "\n  epistasis fraction -- raising --interaction-order would move it):")
    for k in ords:
        print(f"    order {k}: {mass[k]:.4g}  ({mass[k]/tot:.1%})")

    for k in [o for o in ords if o >= 2]:
        sub = inter_df[inter_df["order"] == k]
        n = min(top_n, len(sub))
        print(f"\n  top {n} order-{k} by {value_col}:")
        for r in sub.nlargest(n, value_col).itertuples():
            t = parse_tuple(r.tuple)
            print(f"    {tuple_label(t, labels):<22s} {value_col}={getattr(r, value_col):+.4g}"
                  f"   sii={r.sii_mean_abs:.4g}  dividend={r.dividend_mean_abs:.4g}")

    print(f"\n  top 10 positions by mean|phi_i| (first-order Shapley):")
    for r in pos_df.nlargest(10, "mean_abs_shapley").itertuples():
        print(f"    {labels[int(r.position)]:>5s}  {r.mean_abs_shapley:.4g}")

    ms = inter_df["dividend_mean_signed"].abs()
    ma = inter_df["dividend_mean_abs"].abs().max()
    print(f"\n  mean_signed diagnostic: max|mean_signed| = {ms.max():.4g} "
          f"({ms.max()/(ma+1e-30):.1%} of max|a|)")
    if ms.max() > 0.15 * ma:
        print("    -> NOT ~0: the target sample is far from the background product"
              "\n       distribution, so dividend_sobol is a variance across YOUR sample,"
              "\n       not a formal Sobol sensitivity index.")
    else:
        print("    -> ~0, consistent with targets drawn from the background.")
    print("=" * 78 + "\n")


# ------------------------------------------------------------------ #
# 1. pairwise heatmap + order-1 strip
# ------------------------------------------------------------------ #

def plot_global_heatmap(pos_df, inter_df, value_col, labels, out_dir,
                        strip_index="dividend"):
    """Order-1 strip + full pairwise triangular heatmap on a shared position axis."""
    L = len(pos_df)
    M = np.full((L, L), np.nan)
    for r in inter_df[inter_df["order"] == 2].itertuples():
        i, j = parse_tuple(r.tuple)
        M[max(i, j), min(i, j)] = getattr(r, value_col)   # lower triangle only
    n_pairs = int(np.isfinite(M).sum())

    # Strip: default to the ORDER-1 rows of the same column as the heatmap, so both panels
    # are the same index. a({i}) (ANOVA main effect) and phi_i (Shapley) are different
    # objects; showing one above the other unlabelled would misread as a single decomposition.
    if strip_index == "shapley":
        strip = pos_df.sort_values("position")["mean_abs_shapley"].to_numpy()
        strip_lab = "mean$_x$|$\\phi_i$|\n(Shapley)"
        strip_note = "first-order SHAPLEY values (a different index from the heatmap below)"
    else:
        s1 = inter_df[inter_df["order"] == 1]
        vals = {parse_tuple(r.tuple)[0]: getattr(r, value_col) for r in s1.itertuples()}
        strip = np.array([vals.get(i, np.nan) for i in range(L)])
        strip_lab = "order-1\n" + ("|a(i)|" if "abs" in value_col else "a(i)")
        strip_note = f"order-1 terms of the SAME column as the heatmap ({value_col})"

    signed = "signed" in value_col
    fig = plt.figure(figsize=(max(9.5, 0.34 * L + 3.4), max(9.5, 0.34 * L + 3.0)), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 6], width_ratios=[40, 1.1],
                          hspace=0.30, wspace=0.04)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(np.arange(L), strip, color="#c0504d", width=0.8)
    ax_bar.set_xlim(-0.5, L - 0.5)
    ax_bar.set_xticks(np.arange(L))
    ax_bar.set_xticklabels(labels, rotation=90, fontsize=6.5)   # <-- requested x labels
    ax_bar.set_ylabel(strip_lab, fontsize=8)
    ax_bar.set_xlabel("RBD position", fontsize=8, labelpad=2)
    ax_bar.tick_params(axis="y", labelsize=7)
    ax_bar.set_title(f"Order-1 (single-position) contributions -- {strip_note}",
                     fontsize=9, pad=6)
    for s in ("top", "right"):
        ax_bar.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    if signed:
        lim = float(np.nanmax(np.abs(M))) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", origin="upper",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    else:
        im = ax.imshow(M, cmap="magma_r", origin="upper")
    ax.set_xticks(np.arange(L)); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticks(np.arange(L)); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("RBD position"); ax.set_ylabel("RBD position")
    ax.set_title(f"All {n_pairs} pairwise (order-2) interactions "
                 "-- upper triangle blank by symmetry", fontsize=10)

    cax = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(col_label(value_col), fontsize=8.5)          # <-- requested cbar label
    cb.ax.tick_params(labelsize=7)

    fn = os.path.join(out_dir, f"global_heatmap_{value_col}.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fn}")


# ------------------------------------------------------------------ #
# 2. top-N interactions: SII vs dividend
# ------------------------------------------------------------------ #

def plot_global_top(inter_df, labels, out_dir, value_col, top_n, min_order=2):
    """The N strongest interactions of order >= min_order: SII vs dividend."""
    sub = inter_df[inter_df["order"] >= min_order]
    if sub.empty:
        print(f"[plot] no terms of order >= {min_order}; skipping global_top")
        return
    top = sub.nlargest(min(top_n, len(sub)), value_col).iloc[::-1]
    names = [tuple_label(parse_tuple(t), labels) for t in top["tuple"]]
    y = np.arange(len(top)); h = 0.38

    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.34 * len(top))), dpi=140)
    ax.barh(y - h / 2, top["sii_mean_abs"], h, label="SII (smears onto subsets)",
            color="#8064a2")
    ax.barh(y + h / 2, top["dividend_mean_abs"], h, label="dividend (pure interaction)",
            color="#9bbb59")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("global interaction magnitude   (mean$_x$ |value|)")
    ax.set_ylabel("position tuple")
    ax.set_title(f"Top {len(top)} interactions (order >= {min_order}) by {value_col}",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fn = os.path.join(out_dir, f"global_top{len(top)}.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fn}")


# ------------------------------------------------------------------ #
# 3. order decomposition + per-position participation by order
# ------------------------------------------------------------------ #

def plot_global_orders(inter_df, labels, out_dir):
    """Total dividend mass per order, and per-position mass split by order.

    The right panel answers a question the heatmap cannot: which positions are hubs of
    high-order structure? A position can be modest on its own yet appear in many strong
    interactions. Mass is summed over every tuple containing the position, so a position in
    several tuples counts once per tuple -- a PARTICIPATION measure, not a decomposition.
    """
    L = len(labels)
    ords = orders_present(inter_df)
    mass = {k: float(inter_df.loc[inter_df["order"] == k, "dividend_mean_abs"].sum())
            for k in ords}

    part = np.zeros((len(ords), L))
    for r in inter_df.itertuples():
        k = ords.index(r.order)
        for p in parse_tuple(r.tuple):
            part[k, p] += r.dividend_mean_abs

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(max(11, 0.30 * L + 5), 4.6), dpi=140,
        gridspec_kw={"width_ratios": [1, 3.4], "wspace": 0.22})

    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(1, len(ords) - 1)) for i in range(len(ords))]
    ax0.bar([str(k) for k in ords], [mass[k] for k in ords], color=colors)
    tot = sum(mass.values()) or 1.0
    for i, k in enumerate(ords):
        ax0.text(i, mass[k], f"{mass[k]/tot:.0%}", ha="center", va="bottom", fontsize=8)
    ax0.set_xlabel("interaction order"); ax0.set_ylabel("total mean$_x$|a(T)|")
    ax0.set_title("Dividend mass by order", fontsize=10)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)

    bottom = np.zeros(L)
    for i, k in enumerate(ords):
        ax1.bar(np.arange(L), part[i], bottom=bottom, color=colors[i], label=f"order {k}")
        bottom += part[i]
    ax1.set_xlim(-0.6, L - 0.4)
    ax1.set_xticks(np.arange(L)); ax1.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax1.set_xlabel("RBD position")
    ax1.set_ylabel("summed mean$_x$|a(T)| over\ntuples containing the position")
    ax1.set_title("Per-position participation, split by interaction order "
                  "(participation, not a decomposition)", fontsize=10)
    ax1.legend(fontsize=8)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    fn = os.path.join(out_dir, "global_orders.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fn}")


# ------------------------------------------------------------------ #
# 4. UpSet-style membership grid for high-order tuples
# ------------------------------------------------------------------ #

def plot_global_highorder(inter_df, labels, out_dir, value_col, top_n, min_order=3):
    """Strongest tuples of order >= min_order as a membership grid + magnitude bars.

    A heatmap tops out at order 2 (a rank-3 tensor has no 2-D image). A membership grid has
    no such ceiling: one row per tuple, one column per position, a filled dot where the
    position belongs to the tuple. It stays readable at any order, and vertical alignment
    of dots immediately shows which positions recur across the top high-order terms --
    which is the actual question ("isolate the important ones") rather than "draw them all".

    Only positions appearing in at least one selected tuple get a column, so the panel is
    as narrow as the result allows.
    """
    sub = inter_df[inter_df["order"] >= min_order]
    if sub.empty:
        avail = orders_present(inter_df)
        print(f"[plot] SKIPPED global_highorder: no terms of order >= {min_order} "
              f"(CSV has orders {avail}).")
        print(f"       Re-run analysis.py with --interaction-order {max(min_order, 3)} "
              "to populate them.")
        return

    top = sub.nlargest(min(top_n, len(sub)), value_col)
    tuples = [parse_tuple(t) for t in top["tuple"]]
    vals = top[value_col].to_numpy()
    used = sorted({p for t in tuples for p in t})
    cx = {p: i for i, p in enumerate(used)}
    n, m = len(tuples), len(used)

    fig = plt.figure(figsize=(max(8.5, 0.42 * m + 4.5), max(4.2, 0.30 * n + 1.8)), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[max(1.0, 0.42 * m), 2.4], wspace=0.04)

    axg = fig.add_subplot(gs[0, 0])
    for r in range(n):
        if r % 2 == 0:
            axg.axhspan(r - 0.5, r + 0.5, color="0.94", zorder=0)
    for r, t in enumerate(tuples):
        xs = [cx[p] for p in t]
        axg.plot([min(xs), max(xs)], [r, r], color="#2b5c8f", lw=1.4, zorder=1)
        axg.scatter(xs, [r] * len(xs), s=46, color="#2b5c8f", zorder=2)
    axg.set_xlim(-0.6, m - 0.4); axg.set_ylim(n - 0.5, -0.5)
    axg.set_xticks(range(m))
    axg.set_xticklabels([labels[p] for p in used], rotation=90, fontsize=7)
    axg.set_yticks(range(n))
    axg.set_yticklabels([f"{tuple_label(t, labels)}  ({len(t)})" for t in tuples],
                        fontsize=7)
    axg.set_xlabel("RBD position")
    axg.set_title(f"Top {n} interactions of order >= {min_order}: which positions "
                  "each involves", fontsize=10)
    for s in ("top", "right", "left"):
        axg.spines[s].set_visible(False)

    axb = fig.add_subplot(gs[0, 1], sharey=axg)
    axb.barh(range(n), vals, 0.62, color="#9bbb59")
    # NOT set_yticks([]): the y axis is SHARED with axg, so that would erase axg's tuple
    # labels too. Hide the duplicates on this axes only.
    axb.tick_params(axis="y", labelleft=False, length=0)
    axb.set_xlabel(col_short(value_col), fontsize=9)
    axb.grid(axis="x", alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    fn = os.path.join(out_dir, f"global_highorder_top{n}.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fn}   (orders shown: "
          f"{sorted({len(t) for t in tuples})})")


# ------------------------------------------------------------------ #
# 5. local, any order, with a real legend for resolved/unresolved
# ------------------------------------------------------------------ #

def plot_local(path, labels, out_dir, top_n, z_thresh=2.0, min_order=1):
    """Per target: N largest |dividend| terms of ANY order, signed, with sd error bars."""
    df = pd.read_csv(path)
    seq = os.path.basename(path)[len("local_"):-len(".csv")]
    have_sd = "dividend_sd" in df.columns

    df = df[df["order"] >= min_order]
    if df.empty:
        print(f"[plot] {seq}: no terms of order >= {min_order}; skipped")
        return
    top = df.assign(_mag=df["dividend"].abs()).nlargest(
        min(top_n, len(df)), "_mag").iloc[::-1]
    names = [tuple_label(parse_tuple(t), labels) for t in top["tuple"]]
    y = np.arange(len(top))

    resolved = (top["dividend_z"].abs() >= z_thresh if have_sd
                else pd.Series(True, index=top.index))
    face = ["#4f81bd" if r else "white" for r in resolved]

    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.34 * len(top))), dpi=140)
    ax.barh(y, top["dividend"], 0.7,
            xerr=(top["dividend_sd"] if have_sd else None), capsize=2,
            color=face, edgecolor="#4f81bd", linewidth=1.2,
            error_kw=dict(lw=0.8, ecolor="0.35"))
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("dividend a(T): signed contribution to THIS sequence's phenotype")
    ax.set_ylabel("position tuple")
    orders_shown = sorted({len(parse_tuple(t)) for t in top["tuple"]})
    ax.set_title(f"{seq}\ntop {len(top)} terms by |dividend|   (orders shown: "
                 f"{orders_shown})", fontsize=9)
    if have_sd:
        n_res = int(resolved.sum())
        ax.legend(handles=[
            Patch(facecolor="#4f81bd", edgecolor="#4f81bd",
                  label=f"|z| >= {z_thresh}: resolved from 0 ({n_res}/{len(top)})"),
            Patch(facecolor="white", edgecolor="#4f81bd",
                  label=f"|z| < {z_thresh}: NOT resolved -- could be noise"),
            Line2D([0], [0], color="0.35", lw=0.8, label="error bar = posterior sd"),
        ], fontsize=7.5, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fn = os.path.join(out_dir, f"local_{seq}_top{len(top)}.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    msg = f"[plot] wrote {fn}"
    if have_sd:
        msg += f"   ({int(resolved.sum())}/{len(top)} with |z| >= {z_thresh})"
    print(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="an analysis.py output directory (read AND written)")
    ap.add_argument("--positions", default=None,
                    help="prepare_libraryM *_positions.csv sidecar, so axes carry RBD "
                         "positions instead of 0-based indices")
    ap.add_argument("--value-col", default="dividend_mean_abs",
                    help="global column driving the heatmap / rankings")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--min-order", type=int, default=3,
                    help="lowest order for the high-order membership figure (default 3)")
    ap.add_argument("--z-thresh", type=float, default=2.0,
                    help="|z| below which a local term is drawn hollow (unresolved)")
    ap.add_argument("--strip-index", default="dividend", choices=["dividend", "shapley"],
                    help="what the heatmap's top strip shows: 'dividend' (default, same "
                         "index as the heatmap) or 'shapley' (mean|phi_i|, a DIFFERENT "
                         "index -- labelled as such in the figure)")
    ap.add_argument("--no-local", action="store_true")
    args = ap.parse_args()

    od = args.out_dir
    pos_df = pd.read_csv(os.path.join(od, "global_position_importance.csv"))
    inter_df = pd.read_csv(os.path.join(od, "global_interactions.csv"))
    if args.value_col not in inter_df.columns:
        raise SystemExit(f"--value-col '{args.value_col}' not in global_interactions.csv "
                         f"(have: {[c for c in inter_df.columns if c != 'tuple']})")

    labels = load_positions(args.positions, len(pos_df))
    summarize(pos_df, inter_df, args.value_col, labels, args.top_n)

    plot_global_heatmap(pos_df, inter_df, args.value_col, labels, od,
                        strip_index=args.strip_index)
    plot_global_top(inter_df, labels, od, args.value_col, args.top_n)
    plot_global_orders(inter_df, labels, od)
    plot_global_highorder(inter_df, labels, od, args.value_col, args.top_n,
                          min_order=args.min_order)

    if not args.no_local:
        for p in sorted(glob.glob(os.path.join(od, "local_*.csv"))):
            plot_local(p, labels, od, args.top_n, z_thresh=args.z_thresh)


if __name__ == "__main__":
    main()