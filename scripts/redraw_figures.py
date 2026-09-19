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
by the allele states at the three loci E3 identifies. On the real data those
eight combinations are perfectly monotone in the number of BY alleles,

    BBB 0.1076  <  BAB 0.1243  <  BBA 0.1267  <  ABB 0.1321
        <  BAA 0.1421  <  ABA 0.1543  <  AAB 0.1582  <  AAA 0.1791

spanning 0.0715 of the 0.171 total range in h_ENA1 -- three loci out of 83
accounting for 42% of all variation in ENA1's effect. The per-locus differences
(-0.0307, -0.0208, -0.0200) reproduce E3's delta_j to four decimals, and they
sum to 0.0715, the full range, so the three combine additively with no
detectable three-way term. That is consistent with E5 putting order 3 at 0.8%
of variance, reached independently.

TERMINOLOGY
-----------
"Background" is reserved for the reference distribution pi. Variation over the
131,072 enumerated genotypes is "across genotypes". The two were conflated in
the original axis labels, which invited the reading that panel (d)'s violins
showed variation in pi -- they do not; pi is fixed at the empirical allele
frequencies throughout E2.

USAGE
-----
    # reads output/ENA1_paper_rep1, writes output/ENA1_paper_rep1_redraw
    python scripts/redraw_figures.py --out-dir output/ENA1_paper_rep1

    # somewhere else, e.g. straight into the paper's figure directory
    python scripts/redraw_figures.py --out-dir output/ENA1_paper_rep1 \\
        --fig-dir paper/figures

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
    GENE_BY_LOCUS, PAPER_DELTA_SIGN, C_RM, C_BY, C_NEU, C_ACC,
    banner, sub, save_fig, _style, _panel, gene_label,
    brute_force_unit_cost,
)

# The three loci E3 identifies as compensating for the broken ENA1 allele.
# Hard-coded rather than re-derived so the redraw cannot silently pick a
# different set from a different run; check them against E3's ranking.
COMPENSATORS = ["MKT1", "HAL9", "PHO84"]
ENA1_GENE = "ENA1"


def _locus_of(gene: str) -> int:
    for loc, g in GENE_BY_LOCUS.items():
        if g == gene:
            return int(loc)
    raise KeyError(f"{gene} not in GENE_BY_LOCUS")


def load(out_dir: Path, name: str, **kw) -> pd.DataFrame | None:
    p = out_dir / name
    if not p.exists():
        print(f"  [missing] {p.name}")
        return None
    df = pd.read_csv(p, **kw)
    print(f"  loaded {p.name}  ({len(df):,} rows)")
    return df


# =========================================================================== #
# the replacement panel: h_ENA1 by allele state at the compensating loci
# =========================================================================== #

def compensator_table(per_geno: pd.DataFrame) -> pd.DataFrame:
    """h_ENA1 grouped by the allele combination at the compensating loci.

    Uses only the genotype string and h_ENA1, so it has no arithmetic
    entanglement with f(x) -- which is exactly what went wrong with the
    background-fitness panel this replaces.
    """
    d = per_geno.copy()
    for g in COMPENSATORS:
        d[g] = d["seq"].str[_locus_of(g)]
    t = (d.groupby(COMPENSATORS)["h_ENA1"]
          .agg(n="size", mean="mean", sd="std")
          .reset_index())
    t["n_BY"] = sum((t[g] == "B").astype(int) for g in COMPENSATORS)
    t["label"] = t[COMPENSATORS].agg("".join, axis=1)
    return t.sort_values("mean").reset_index(drop=True)


def report_compensators(per_geno: pd.DataFrame, e3: pd.DataFrame | None):
    t = compensator_table(per_geno)
    sub("h_ENA1 by allele state at the compensating loci")
    print(f"  {'combination':>12}{'n_BY':>6}{'n':>9}{'mean h':>10}{'sd':>9}")
    for _, r in t.iterrows():
        print(f"  {r['label']:>12}{int(r['n_BY']):>6}{int(r['n']):>9,}"
              f"{r['mean']:>10.4f}{r['sd']:>9.4f}")
    rng = float(t["mean"].max() - t["mean"].min())
    tot = float(per_geno["h_ENA1"].max() - per_geno["h_ENA1"].min())
    print(f"\n  range across the {len(t)} combinations: {rng:.4f} "
          f"of a total h_ENA1 range of {tot:.4f} ({rng / tot:.0%})")
    mono = t["n_BY"].is_monotonic_decreasing or t["n_BY"].is_monotonic_increasing
    print(f"  monotone in the number of BY alleles: {mono}")

    per = {}
    for g in COMPENSATORS:
        d = per_geno.copy()
        d[g] = d["seq"].str[_locus_of(g)]
        m = d.groupby(g)["h_ENA1"].mean()
        per[g] = float(m.get("B", np.nan) - m.get("A", np.nan))
        print(f"  {g:<6} A {m.get('A', np.nan):+.4f}  B {m.get('B', np.nan):+.4f}"
              f"   B-A {per[g]:+.4f}")
    s = sum(abs(v) for v in per.values())
    print(f"\n  |per-locus differences| sum to {s:.4f} against a full range of "
          f"{rng:.4f}")
    if abs(s - rng) < 0.05 * rng:
        print("  -> the three combine ADDITIVELY; no detectable three-way term, "
              "consistent with E5 placing order 3 at under 1% of variance")

    if e3 is not None:
        sub("cross-check against E3's delta_j")
        d3 = e3.set_index("gene")["delta_rm_minus_by"]
        for g in COMPENSATORS:
            if g in d3.index:
                print(f"  {g:<6} this panel {-per[g]:+.4f}   "
                      f"E3 delta_j {float(d3[g]):+.4f}   "
                      f"diff {abs(-per[g] - float(d3[g])):.2e}")
        print("  (opposite sign by construction: delta is RM minus BY, this is "
              "B minus A)")
    return t


def _panel_compensators(ax, per_geno: pd.DataFrame, t: pd.DataFrame):
    """Violins of h_ENA1, one per allele combination, ordered by mean."""
    d = per_geno.copy()
    for g in COMPENSATORS:
        d[g] = d["seq"].str[_locus_of(g)]
    d["label"] = d[COMPENSATORS].agg("".join, axis=1)
    order = list(t["label"])
    groups = [d.loc[d["label"] == lab, "h_ENA1"].to_numpy() for lab in order]

    parts = ax.violinplot(groups, positions=range(len(order)), widths=0.82,
                          showextrema=False, showmedians=True)
    nby = t["n_BY"].to_numpy()
    # Colour by how many buffering alleles the combination carries.
    cmap = plt.get_cmap("Blues")
    for pc, k in zip(parts["bodies"], nby):
        pc.set_facecolor(cmap(0.30 + 0.20 * k))
        pc.set_alpha(0.95)
        pc.set_linewidth(0)
    if "cmedians" in parts:
        parts["cmedians"].set_color("k")
        parts["cmedians"].set_linewidth(1.1)
    ax.plot(range(len(order)), t["mean"], "o-", color=C_RM, ms=3.5, lw=1.3,
            zorder=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlabel("allele state at " + " / ".join(COMPENSATORS)
                  + "   (A = RM, B = BY)")
    ax.set_ylabel("$h_{ENA1}$")


# =========================================================================== #
# E2 figure
# =========================================================================== #

def fig_e2(out_dir: Path, per_geno: pd.DataFrame, per_locus: pd.DataFrame,
           long: pd.DataFrame | None, t_comp: pd.DataFrame, n_show: int = 30):
    _style()
    fig = plt.figure(figsize=(14.5, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.22, left=0.065,
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
    ax.set_xlabel("$h_{ENA1}$")
    ax.set_ylabel("genotypes")
    ax.legend(fontsize=8, frameon=False)
    _panel(ax, "a", "Distribution of $h_{ENA1}$ across genotypes")

    # (b) the replacement panel
    ax = fig.add_subplot(gs[0, 1])
    _panel_compensators(ax, per_geno, t_comp)
    _panel(ax, "b", "$h_{ENA1}$ by allele state at the compensating loci")

    # (c) per-locus, across genotypes
    ax = fig.add_subplot(gs[1, :])
    pl = per_locus.copy()
    if "is_focal_qtl" in pl and "varies_in_targets" in pl:
        pl = pl[pl["is_focal_qtl"] | pl["varies_in_targets"]]
    drawn = False
    if long is not None:
        # INTERSECT rather than requiring full coverage. The long-format CSV
        # carries the focal QTLs plus the first 40 varying loci, which need not
        # contain every locus in the top n_show by |mean h|; demanding full
        # coverage threw away the violins entirely whenever one locus was
        # absent. Take the top n_show loci that ARE present.
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
            print(f"  panel (c): violins for {len(order)} loci present in the "
                  "long-format CSV")
    if not drawn:
        # Fall back to mean +- sd from the per-locus summary, which is always
        # present even when the long-format file is skipped or too large.
        pl_v = pl.head(n_show)
        order = [int(j) for j in pl_v["locus"]]
        print("  [note] panel (c) drawn as mean +- sd (no long-format CSV)")
        ax.errorbar(range(len(order)), pl_v["mean_h"], yerr=pl_v["sd_h"],
                    fmt="o", ms=4, color=C_ACC, ecolor="0.4", capsize=2,
                    lw=0, elinewidth=1.0)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([gene_label(j) for j in order], rotation=90, fontsize=8)
    ena1 = _locus_of(ENA1_GENE)
    for tick, j in zip(ax.get_xticklabels(), order):
        if j == ena1:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    ax.set_ylabel("$h_i$")
    ax.set_xlabel("locus, ordered by $|$mean $h_i|$")
    _panel(ax, "c", "Per-locus substitution effect across genotypes "
                    "(width = epistasis; $\\pi$ fixed)")

    n = len(per_geno)
    fig.suptitle(f"E2   {n:,} genotypes $\\times$ 83 loci = {n * 83:,} exact "
                 "first-order Shapley values", fontsize=12, y=0.955)
    return save_fig(fig, out_dir, "fig_E2_substitution_landscape")


# =========================================================================== #
# composite paper figure
# =========================================================================== #

def fig_paper(out_dir: Path, per_geno: pd.DataFrame, t_comp: pd.DataFrame,
              e3: pd.DataFrame | None, e4: pd.DataFrame | None,
              timing: pd.DataFrame | None, throughput: pd.DataFrame | None):
    _style()
    fig = plt.figure(figsize=(15.0, 8.6))
    gs = fig.add_gridspec(2, 6, hspace=0.52, wspace=0.95, left=0.055,
                          right=0.985, top=0.86, bottom=0.15)

    # (a) cost
    ax = fig.add_subplot(gs[0, 0:2])
    if timing is not None and len(timing):
        g = timing.dropna(subset=["seconds_gkt"]).sort_values("L")
        ax.plot(g["L"], g["seconds_gkt"], "o-", ms=3.5, color=C_ACC, lw=1.6,
                label="GKT (measured)")
        unit, L_meas = brute_force_unit_cost(timing)
        b = timing.dropna(subset=["seconds_brute_force"]).sort_values("L")
        if len(b):
            ax.plot(b["L"], b["seconds_brute_force"], "s-", ms=3.5, color=C_RM,
                    lw=1.6, label=f"enumeration (meas., $L\\leq{L_meas}$)")
        if np.isfinite(unit) and len(b):
            Ls = np.arange(int(b["L"].max()), int(g["L"].max()) + 1)
            ax.plot(Ls, unit * Ls * 2.0 ** (Ls - 1), ":", color=C_RM, lw=1.6,
                    label="enumeration (projected)")
        if throughput is not None and len(throughput):
            r = throughput.iloc[0]
            spg = float(r["seconds"]) / float(r["n_targets"])
            ax.plot([83], [spg], "*", ms=14, color="k", zorder=6,
                    label=f"this work, $n$={int(r['n_condition']):,}")
        ax.set_yscale("log")
        ax.set_xlabel("sequence length $L$")
        ax.set_ylabel("seconds per genotype")
        ax.legend(fontsize=6.6, loc="upper left", borderpad=0.3,
                  labelspacing=0.3, handlelength=1.5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "pass --timing-csv for the cost panel",
                ha="center", va="center", fontsize=9)
    _panel(ax, "a", "Wall-clock cost")

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
    ax.set_xlabel("$h_{ENA1}$")
    ax.set_ylabel("genotypes")
    ax.legend(fontsize=7.5)
    _panel(ax, "b", "Distribution of $h_{ENA1}$")

    # (c) the replacement panel
    ax = fig.add_subplot(gs[0, 4:6])
    _panel_compensators(ax, per_geno, t_comp)
    _panel(ax, "c", "$h_{ENA1}$ by compensating-locus state")

    # (d) delta_j
    ax = fig.add_subplot(gs[1, 0:3])
    if e3 is not None and len(e3):
        t3 = e3[e3["design_identified"]] if "design_identified" in e3 else e3
        t3 = t3.head(15)
        xs = np.arange(len(t3))
        eb = (t3["gp_posterior_sd_h"] if "gp_posterior_sd_h" in t3
              else t3["delta_sem"])
        ax.bar(xs, t3["delta_rm_minus_by"], yerr=eb,
               color=np.where(t3["delta_rm_minus_by"] >= 0, C_RM, C_BY),
               width=0.74, error_kw=dict(lw=0.8, ecolor="0.25", capsize=2))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([g if isinstance(g, str) else f"L{int(l)}"
                            for g, l in zip(t3["gene"], t3["locus"])],
                           rotation=90, fontsize=8)
        for tick, g in zip(ax.get_xticklabels(), t3["gene"]):
            if g in PAPER_DELTA_SIGN:
                tick.set_fontweight("bold")
        ax.set_ylabel("$\\delta_j$")
    else:
        ax.axis("off")
    _panel(ax, "d", "$\\delta_j$ per locus   (bold: reported in the source "
                    "paper)")

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
                 "uncertainty on a $2^{83}$ sequence space\n"
                 f"{n * 83:,} Shapley values, each a $2^{{82}}$-term sum",
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
                         "original PNGs are never overwritten. Created if "
                         "absent.")
    ap.add_argument("--overwrite-in-place", action="store_true",
                    help="write into --out-dir after all, replacing the "
                         "existing figures. Off by default: the originals are "
                         "the record of what a given run produced, and a "
                         "redraw is a different object even when it uses the "
                         "same numbers.")
    ap.add_argument("--timing-csv", type=Path, default=None,
                    help="E1_timing.csv for the composite cost panel; defaults "
                         "to <out-dir>/E1_timing.csv")
    ap.add_argument("--n-show", type=int, default=30,
                    help="loci in E2 panel (c)")
    ap.add_argument("--skip-long", action="store_true",
                    help="do not read E2_phi_and_h_long.csv (hundreds of MB at "
                         "the full landscape); panel (c) falls back to "
                         "mean +- sd from the per-locus summary")
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

    t_comp = report_compensators(per_geno, e3)

    banner("redrawing")
    fig_e2(figs, per_geno, per_locus, long, t_comp, n_show=a.n_show)
    fig_paper(figs, per_geno, t_comp, e3, e4, timing, tput)

    # A note beside the figures recording what they are, since a bare pair of
    # PNGs in a new directory is otherwise indistinguishable from the run's own
    # output.
    notes = [
        "Figures redrawn by redraw_figures.py from the CSVs in",
        f"    {out}",
        "No model was run; the numbers are identical to that run's.",
        "",
        "Changes from the originals:",
        "  * E2 panel (b), h_ENA1 vs beneficial-allele dosage, REMOVED.",
        "    Non-monotone with r = -0.06: the loci counted interact with",
        "    ENA1 in opposite directions, so summing them cancels the",
        "    signal.",
        "  * E2 panel (c), h_ENA1 vs background fitness, REMOVED.",
        "    Opposite slopes in the two ENA1 groups (+0.126 BY, -0.531 RM):",
        "    subtracting phi_ENA1 removes the mean offset but not the",
        "    correlation, and the residual is built from exactly the loci",
        "    that modulate h_ENA1.",
        "  * Both replaced by h_ENA1 split by allele state at",
        "    MKT1/HAL9/PHO84, which states the mechanism directly and has",
        "    no arithmetic entanglement with f(x).",
        "  * 'background' now refers only to the reference distribution",
        "    pi; variation over genotypes is labelled 'across genotypes'.",
    ]
    (figs / "REDRAW_NOTES.txt").write_text("\n".join(notes) + "\n")
    print(f"  wrote {figs / 'REDRAW_NOTES.txt'}")

    banner("done")
    print(f"  figures in {figs}")
    print("  E2 panel (b) now shows h_ENA1 by compensating-locus state; the "
          "dosage and background-fitness panels are gone.")
    print("  'background' now refers only to the reference distribution pi; "
          "variation over genotypes is labelled as such.")
    return 0


if __name__ == "__main__":
    sys.exit(main())