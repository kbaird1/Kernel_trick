"""
redraw_figures.py
=================

Rebuild the E2 and composite paper figures from CSVs already on disk. No model,
no GPU, no re-attribution -- reads the output directory of a finished
ENA1_paper_analysis.py run and redraws in seconds.

WHY THIS EXISTS
---------------
Two problems with the figures as originally drawn, both found by inspection
after the full-landscape run.

1. E2 panel (c) plotted h_ENA1 against "background fitness"
   f(x) - phi_ENA1(x), intending to remove ENA1's own contribution so that any
   remaining trend came from the other 82 loci. It does remove the mean offset
   (the two groups' mean background fitness agree to four decimals) but NOT the
   correlation structure. Diagnostic on the real data:

       ENA1(BY):  r = +0.128, slope +0.126
       ENA1(RM):  r = -0.686, slope -0.531

   Opposite signs, so there is no single relationship being shown and the
   pooled r = -0.326 is an average of two different ones. The cause:
   bg = phi_0 + sum_{i != ENA1} phi_i, and each phi_i = (z_i - p_i) h_i with
   h_i depending on ENA1's allele. The between-group difference is therefore
   -sum_{i != ENA1} (z_i - p_i) delta_i, i.e. exactly E3's interaction terms --
   and those are by definition the loci that modulate h_ENA1, so the
   contaminant is maximally aligned with the y-axis. The panel is dropped.

2. E2 panel (b) plotted h_ENA1 against a count of beneficial alleles. The
   relationship is non-monotone with r = -0.06: dosage does not predict
   buffering, because the loci being counted interact with ENA1 in OPPOSITE
   directions (MKT1, HAL9, PHO84 positive; BUL2, SUP45, TAO3 negative) and
   summing them averages the signal away.

Both are replaced by one panel that states the mechanism directly: h_ENA1 split
by the allele states at the three loci E3 identifies.

COST PANEL (composite panel a)
------------------------------
By default the cost panel is now drawn against the NUMBER OF SEQUENCES in the
space, |A|^L, on a log axis, with sequence length L kept as a secondary axis on
top. Plotted against L the enumeration curves are merely straight lines; against
the size of the space they show what the method is actually being asked to sum
over (~1e25 sequences at L = 83). --cost-x length restores the old axis.

TERMINOLOGY
-----------
"Background" is reserved for the reference distribution pi. Variation over the
genotypes is "across genotypes".

USAGE
-----
    # reads output/ENA1_paper_rep1, writes output/ENA1_paper_rep1_redraw
    python scripts/redraw_figures.py --out-dir output/ENA1_paper_rep1

    # old cost-panel axis (sequence length)
    python scripts/redraw_figures.py --out-dir output/ENA1_paper_rep1 \\
        --cost-x length

    # cost panel of the composite figure, if stage 1 ran elsewhere
    python scripts/redraw_figures.py --out-dir output/ENA1_paper_rep1 \\
        --timing-csv output/gkt_E1/E1_timing.csv

The source directory is READ-ONLY. Figures go to <out-dir>_redraw unless
--fig-dir says otherwise, so the originals are never overwritten;
--overwrite-in-place opts into replacing them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _d in (_HERE, _ROOT / "modules", _ROOT):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from ENA1_paper_analysis import (                              # noqa: E402
    GENE_BY_LOCUS, PAPER_EPSILON_SIGN, C_RM, C_BY, C_NEU, C_ACC,
    banner, sub, save_fig, _style, _panel, gene_label,
    brute_force_unit_cost,
    # Shared with the analysis script ON PURPOSE, so a redraw cannot disagree
    # with a fresh run.
    _draw_cost_panel, _panel_compensators, compensator_table,
    report_compensators, _eq, EQ_H, EQ_EPS, COMPENSATORS, locus_of,
)

ENA1_GENE = "ENA1"

_locus_of = locus_of        # the analysis script's version


def load(out_dir: Path, name: str, **kw) -> pd.DataFrame | None:
    p = out_dir / name
    if not p.exists():
        print(f"  [missing] {p.name}")
        return None
    df = pd.read_csv(p, **kw)
    print(f"  loaded {p.name}  ({len(df):,} rows)")
    return df


# =========================================================================== #
# cost panel against the size of the sequence space
# =========================================================================== #

def _draw_cost_panel_seqspace(ax, timing: pd.DataFrame,
                              fontsize: float = 6.6) -> None:
    """Per-target wall-clock cost against the NUMBER OF SEQUENCES |A|^L.

    Same four curves and the same projections as _draw_cost_panel, with only
    the x-coordinate changed from L to |A|^L (log axis). The projections are
    the same functions of L -- L 2^(L-1) subset terms for the enumerated mean,
    4^L kernel entries for the enumerated variance -- evaluated at each L and
    placed at |A|^L. Sequence length is kept on a secondary top axis.
    """
    from matplotlib.lines import Line2D
    t = timing.copy()
    if "space_size" not in t or t["space_size"].isna().all():
        t["space_size"] = 2.0 ** t["L"].astype(float)
    # Alphabet size from the table itself, so the axis is right for any A.
    r0 = t.dropna(subset=["space_size"]).iloc[0]
    A = float(round(float(r0["space_size"]) ** (1.0 / float(r0["L"]))))
    handles = []

    g = t.dropna(subset=["seconds_gkt"]).sort_values("L")
    ax.plot(g["space_size"], g["seconds_gkt"], "o-", ms=3.5, color=C_ACC,
            lw=1.6)
    handles.append(Line2D([], [], color=C_ACC, marker="o", ms=3.5, lw=1.6,
                          label="GKT, mean"))
    if "seconds_gkt_variance" in t:
        gv = t.dropna(subset=["seconds_gkt_variance"]).sort_values("L")
        if len(gv):
            ax.plot(gv["space_size"], gv["seconds_gkt_variance"], "^-", ms=3.5,
                    color=C_BY, lw=1.6)
            handles.append(Line2D([], [], color=C_BY, marker="^", ms=3.5,
                                  lw=1.6, label="GKT, mean + posterior var"))

    L_max = int(g["L"].max())
    unit, _ = brute_force_unit_cost(t)
    b = t.dropna(subset=["seconds_brute_force"]).sort_values("L")
    if len(b):
        ax.plot(b["space_size"], b["seconds_brute_force"], "s", ms=3.5,
                color=C_RM)
        if np.isfinite(unit):
            Ls = np.arange(int(b["L"].min()), L_max + 1, dtype=float)
            ax.plot(A ** Ls, unit * Ls * 2.0 ** (Ls - 1), ":", color=C_RM,
                    lw=1.6)
        handles.append(Line2D([], [], color=C_RM, marker="s", ms=3.5, lw=1.6,
                              ls=":", label="enumeration, mean (projected)"))
    if "seconds_brute_force_variance" in t:
        bv = t.dropna(subset=["seconds_brute_force_variance"]).sort_values("L")
        if len(bv):
            C_VAR = "#7B3294"
            ax.plot(bv["space_size"], bv["seconds_brute_force_variance"], "D",
                    ms=3.5, color=C_VAR)
            unit_v = (float(bv["seconds_brute_force_variance"].iloc[-1])
                      / 4.0 ** float(bv["L"].iloc[-1]))
            Lv = np.arange(int(bv["L"].min()), L_max + 1, dtype=float)
            ax.plot(A ** Lv, unit_v * 4.0 ** Lv, ":", color=C_VAR, lw=1.6)
            handles.append(Line2D([], [], color=C_VAR, marker="D", ms=3.5,
                                  lw=1.6, ls=":",
                                  label="enumeration, variance (projected)"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    n_max = A ** L_max
    ax.axvline(n_max, color="0.6", lw=0.8, ls="--", zorder=0)
    ax.text(n_max, 0.24, f"$L={L_max}$: {n_max:.1e}\nsequences  ",
            transform=ax.get_xaxis_transform(), ha="right", va="bottom",
            fontsize=fontsize, color="0.35")
    ax.set_xlabel(f"sequences in the space, ${int(A)}^L$")
    ax.set_ylabel("seconds per genotype")
    # Sequence length on a top axis, drawn by hand: ticks at A^L labelled L.
    # (secondary_xaxis with function transforms picks its own log scale here
    # and mislabels the ticks.)
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    Lt = [v for v in (10, 20, 40, 60, L_max) if v <= L_max]
    top.set_xticks([A ** v for v in Lt])
    top.set_xticklabels([str(v) for v in Lt], fontsize=fontsize + 0.5)
    top.minorticks_off()
    top.set_xlabel("sequence length $L$", fontsize=fontsize + 1)
    top.spines["top"].set_visible(True)
    ax.legend(handles=handles, fontsize=fontsize, loc="upper left",
              borderpad=0.3, labelspacing=0.3, handlelength=2.2)


def draw_cost(ax, timing: pd.DataFrame, cost_x: str = "sequences",
              fontsize: float = 6.4) -> None:
    """Dispatch to the chosen cost-panel axis."""
    if cost_x == "length":
        _draw_cost_panel(ax, timing, fontsize=fontsize)
    else:
        _draw_cost_panel_seqspace(ax, timing, fontsize=fontsize)


# =========================================================================== #
# E2 figure
# =========================================================================== #

def fig_e2(out_dir: Path, per_geno: pd.DataFrame, per_locus: pd.DataFrame,
           long: pd.DataFrame | None, t_comp: pd.DataFrame, n_show: int = 30):
    """E2 figure: distribution, per-locus survey, then the compensating loci."""
    _style()
    fig = plt.figure(figsize=(15.0, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.26, left=0.065,
                          right=0.985, top=0.87, bottom=0.14)

    # (a) distribution of h_ENA1
    ax = fig.add_subplot(gs[0, 0])
    h = per_geno["h_ENA1"].to_numpy()
    tol = per_geno["rm_tolerant"].to_numpy(dtype=bool)
    rm = (per_geno["ena1_state"] == "RM").to_numpy()
    bins = np.linspace(h.min(), h.max(), 60)
    ax.hist(h[rm & ~tol], bins=bins, color=C_NEU, alpha=0.85,
            label="all other")
    if tol.any():
        ax.hist(h[tol], bins=bins, color=C_RM, alpha=0.9,
                label="predicted RM-tolerant")
    ax.axvline(float(h.mean()), color="k", ls="--", lw=1.0)
    ax.set_xlabel("$h_{ENA1}$   (effect of ENA1 BY$\\to$RM)")
    ax.set_ylabel("genotypes")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _eq(ax, EQ_H, "lower left")
    _panel(ax, "a", "Distribution of $h_{ENA1}$ across genotypes")

    # (b) per-locus survey
    ax = fig.add_subplot(gs[0, 1:3])
    pl = per_locus.copy()
    if "is_focal_qtl" in pl and "varies_in_targets" in pl:
        pl = pl[pl["is_focal_qtl"] | pl["varies_in_targets"]]
    drawn = False
    if long is not None:
        have = set(int(j) for j in long["locus"].unique())
        pl_v = pl[pl["locus"].astype(int).isin(have)].head(n_show)
        order = [int(j) for j in pl_v["locus"]]
        if order:
            groups = [long.loc[long["locus"] == j, "h"].to_numpy()
                      for j in order]
            parts = ax.violinplot(groups, positions=range(len(order)),
                                  widths=0.85, showextrema=False,
                                  showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(C_ACC)
                pc.set_alpha(0.7)
                pc.set_linewidth(0)
            if "cmedians" in parts:
                parts["cmedians"].set_color("k")
                parts["cmedians"].set_linewidth(1.1)
            drawn = True
            print(f"  panel (b): violins for {len(order)} loci")
    if not drawn:
        pl_v = pl.head(n_show)
        order = [int(j) for j in pl_v["locus"]]
        print("  [note] panel (b) drawn as mean +- sd (no long-format CSV)")
        ax.errorbar(range(len(order)), pl_v["mean_h"], yerr=pl_v["sd_h"],
                    fmt="o", ms=4, color=C_ACC, ecolor="0.4", capsize=2,
                    lw=0, elinewidth=1.0)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([gene_label(j) for j in order], rotation=90,
                       fontsize=7.5)
    ena1 = _locus_of(ENA1_GENE)
    for tick, j in zip(ax.get_xticklabels(), order):
        if j == ena1:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    ax.set_ylabel("$h_i$")
    ax.set_xlabel("locus, ordered by $|$mean $h_i|$")
    _panel(ax, "b", "Per-locus effect across genotypes "
                    "(width = epistasis; $\\pi$ fixed)")

    # (c) the three compensating loci
    ax = fig.add_subplot(gs[1, :])
    _panel_compensators(ax, [str(x) for x in per_geno["seq"]],
                        per_geno["h_ENA1"].to_numpy(), t_comp)
    _panel(ax, "c", "$h_{ENA1}$ by allele state at the three compensating loci")

    n = len(per_geno)
    fig.suptitle(f"E2   {n:,} genotypes $\\times$ 83 loci = {n * 83:,} exact "
                 "first-order Shapley values", fontsize=12, y=0.955)
    return save_fig(fig, out_dir, "fig_E2_substitution_landscape")


# =========================================================================== #
# composite paper figure
# =========================================================================== #

def fig_paper(out_dir: Path, per_geno: pd.DataFrame, t_comp: pd.DataFrame,
              e3: pd.DataFrame | None, e4: pd.DataFrame | None,
              timing: pd.DataFrame | None, throughput: pd.DataFrame | None,
              cost_x: str = "sequences"):
    _style()
    fig = plt.figure(figsize=(15.0, 8.6))
    gs = fig.add_gridspec(2, 6, hspace=0.52, wspace=0.95, left=0.055,
                          right=0.985, top=0.86, bottom=0.15)

    # (a) cost -- against the number of sequences by default
    ax = fig.add_subplot(gs[0, 0:2])
    if timing is not None and len(timing):
        draw_cost(ax, timing, cost_x=cost_x, fontsize=6.4)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "pass --timing-csv for the cost panel",
                ha="center", va="center", fontsize=9)
    _panel(ax, "a", "Per-target wall-clock cost")

    # (b) distribution
    ax = fig.add_subplot(gs[0, 2:4])
    h = per_geno["h_ENA1"].to_numpy()
    tol = per_geno["rm_tolerant"].to_numpy(dtype=bool)
    rm = (per_geno["ena1_state"] == "RM").to_numpy()
    bins = np.linspace(h.min(), h.max(), 55)
    ax.hist(h[rm & ~tol], bins=bins, color=C_NEU, alpha=0.85, label="all other")
    if tol.any():
        ax.hist(h[tol], bins=bins, color=C_RM, alpha=0.9,
                label="predicted RM-tolerant")
    ax.set_xlabel("$h_{ENA1}$   (effect of ENA1 BY$\\to$RM)")
    ax.set_ylabel("genotypes")
    ax.legend(fontsize=7.5, loc="upper left")
    _eq(ax, EQ_H, "lower left", 7.5)
    _panel(ax, "b", "Distribution of $h_{ENA1}$")

    # (c) compensating loci
    ax = fig.add_subplot(gs[0, 4:6])
    _panel_compensators(ax, [str(x) for x in per_geno["seq"]],
                        per_geno["h_ENA1"].to_numpy(), t_comp)
    _panel(ax, "c", "$h_{ENA1}$ by compensating-locus state")

    # (d) epsilon_j
    ax = fig.add_subplot(gs[1, 0:3])
    if e3 is not None and len(e3):
        t3 = e3[e3["design_identified"]] if "design_identified" in e3 else e3
        t3 = t3.head(15)
        xs = np.arange(len(t3))
        if ("epsilon_posterior_sd" in t3
                and t3.get("epsilon_posterior_converged",
                           pd.Series(False, index=t3.index)).any()):
            fallback = t3.get("gp_posterior_sd_h", t3["epsilon_sem"])
            eb = t3["epsilon_posterior_sd"].where(
                t3["epsilon_posterior_converged"].astype(bool), fallback)
            print("  panel (d): converged posterior s.d. of epsilon where "
                  "available")
        else:
            eb = (t3["gp_posterior_sd_h"] if "gp_posterior_sd_h" in t3
                  else t3["epsilon_sem"])
        ax.bar(xs, t3["epsilon_j"], yerr=eb,
               color=np.where(t3["epsilon_j"] >= 0, C_RM, C_BY),
               width=0.74, error_kw=dict(lw=0.8, ecolor="0.25", capsize=2))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([g if isinstance(g, str) else f"L{int(l)}"
                            for g, l in zip(t3["gene"], t3["locus"])],
                           rotation=90, fontsize=8)
        for tick, g in zip(ax.get_xticklabels(), t3["gene"]):
            if g in PAPER_EPSILON_SIGN:
                tick.set_fontweight("bold")
        ax.set_ylabel("$\\varepsilon_j$")
        _eq(ax, EQ_EPS, "lower right", 7.5)
    else:
        ax.axis("off")
    _panel(ax, "d", "$\\varepsilon_j$ per locus")

    # (e) pi sweep
    ax = fig.add_subplot(gs[1, 3:6])
    if e4 is not None and len(e4):
        ena1 = _locus_of(ENA1_GENE)
        cmap = plt.get_cmap("tab20")
        for k, (loc, grp) in enumerate(e4.groupby("locus")):
            grp = grp.sort_values("t")
            is_e = int(loc) == ena1
            ax.plot(grp["t"], grp["mean_h"], "-",
                    color=C_RM if is_e else cmap(k % 20),
                    lw=2.2 if is_e else 1.0, zorder=5 if is_e else 2,
                    label="ENA1" if is_e else None)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("$t$   (0 = RM parent, 1 = BY parent)")
        ax.set_ylabel("mean $h_j(t)$")
        ax.legend(fontsize=7.5)
    else:
        ax.axis("off")
    _panel(ax, "e", "$h_j$ vs reference background $\\pi(t)$")

    n = len(per_geno)
    fig.suptitle("Exact first-order Shapley attribution with GP posterior "
                 f"uncertainty\nacross the {n:,} measured segregants",
                 fontsize=12, y=0.955)
    return save_fig(fig, out_dir, "fig_paper_experiments")


# =========================================================================== #
# main
# =========================================================================== #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="finished ENA1_paper_analysis.py output directory. "
                         "READ-ONLY: nothing is written here.")
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help="where the redrawn figures go. Defaults to "
                         "<out-dir>_redraw, deliberately NOT --out-dir, so the "
                         "original PNGs are never overwritten.")
    ap.add_argument("--overwrite-in-place", action="store_true",
                    help="write into --out-dir after all, replacing the "
                         "existing figures.")
    ap.add_argument("--timing-csv", type=Path, default=None,
                    help="E1_timing.csv for the composite cost panel; defaults "
                         "to <out-dir>/E1_timing.csv")
    ap.add_argument("--cost-x", choices=["sequences", "length"],
                    default="sequences",
                    help="x-axis of the composite cost panel: 'sequences' "
                         "(default) = number of sequences |A|^L on a log axis, "
                         "with L on a secondary top axis; 'length' = the "
                         "original sequence-length axis")
    ap.add_argument("--n-show", type=int, default=30,
                    help="loci in E2 panel (b)")
    ap.add_argument("--skip-long", action="store_true",
                    help="do not read E2_phi_and_h_long.csv; panel (b) falls "
                         "back to mean +- sd from the per-locus summary")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if not out.exists():
        print(f"!! {out} does not exist")
        return 1

    if a.overwrite_in_place:
        figs = out
    elif a.fig_dir is not None:
        figs = Path(a.fig_dir)
    else:
        figs = out.parent / f"{out.name}_redraw"
    figs.mkdir(parents=True, exist_ok=True)

    banner(f"redraw: read {out}  ->  write {figs}")
    if figs == out:
        print("  !! --overwrite-in-place: the existing figures in this "
              "directory WILL be replaced")
    else:
        print("  source directory is read-only; originals are untouched")
        existing = sorted(p.name for p in figs.glob("fig_*"))
        if existing:
            print(f"  [note] {figs.name} already holds "
                  f"{len(existing)} fig_* file(s); those will be replaced")
    print(f"  cost panel x-axis: {a.cost_x}")
    per_geno = load(out, "E2_per_genotype.csv")
    per_locus = load(out, "E2_substitution_effects_by_locus.csv")
    if per_geno is None or per_locus is None:
        print("\n!! E2_per_genotype.csv and E2_substitution_effects_by_locus.csv "
              "are both required")
        return 1
    long = None if a.skip_long else load(
        out, "E2_phi_and_h_long.csv", usecols=["locus", "h"])
    e3 = load(out, "E3_compensating_loci.csv")
    e4 = load(out, "E4_pi_sweep.csv")
    tput = load(out, "E2_throughput.csv")
    tcsv = a.timing_csv or (out / "E1_timing.csv")
    timing = pd.read_csv(tcsv) if Path(tcsv).exists() else None
    if timing is None:
        print(f"  [missing] {tcsv} -- the composite cost panel will be blank")

    seqs = [str(x) for x in per_geno["seq"]]
    h_ena1 = per_geno["h_ENA1"].to_numpy()
    t_comp = report_compensators(seqs, h_ena1, e3_table=e3)

    banner("redrawing")
    fig_e2(figs, per_geno, per_locus, long, t_comp, n_show=a.n_show)
    fig_paper(figs, per_geno, t_comp, e3, e4, timing, tput, cost_x=a.cost_x)

    notes = [
        "Figures redrawn by redraw_figures.py from the CSVs in",
        f"    {out}",
        "No model was run; the numbers are identical to that run's.",
        "",
        "Changes from the originals:",
        f"  * Composite panel (a) x-axis: {a.cost_x}"
        + ("  (number of sequences |A|^L, log; L on the top axis)"
           if a.cost_x == "sequences" else ""),
        "  * E2: dosage and background-fitness panels removed; replaced by",
        "    h_ENA1 split by allele state at MKT1/HAL9/PHO84.",
        "  * 'background' refers only to the reference distribution pi.",
    ]
    (figs / "REDRAW_NOTES.txt").write_text("\n".join(notes) + "\n")
    print(f"  wrote {figs / 'REDRAW_NOTES.txt'}")

    banner("done")
    print(f"  figures in {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())