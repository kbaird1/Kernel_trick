"""
redraw_paper_figures.py
=======================

Redraw the paper's figures from two finished runs, without recomputing anything:

    --ena1-dir   output of ENA1_paper_analysis.py   (e.g. output/ENA1_paper_rep1)
    --e5-dir     output of E5_variance_components.py (e.g. output/E5_full)

Both source folders are READ-ONLY. Everything is written to --out-dir, which
must be a different folder (it is created if absent, and the script refuses to
write into either source folder).

MAIN FIGURE  Yeast_ENA1_Sequence-Function_Map   (6.75 x 4.9 in, 2 x 2)
    left column = the two timing comparisons, so panels follow the text
    (a) Shapley cost per genotype, GKT vs enumeration
    (b) Shapley values phi_i of the all-RM parent at every locus: bars =
        posterior mean, whiskers = +-1 posterior s.d. (exact posterior;
        needs SHAP_ENA1_experiment.py --stages parental)
    (c) variance-component cost (posterior means of all V_k), GKT vs
        enumeration -- (a) and (c): 2^L below, sequence length L on top
    (d) order spectrum: prior, MAP and posterior mean of V_k, orders 1..8
        plus the aggregate V_{k>8}; whiskers = moment-matched 95% range
    Fonts: 7 pt ticks, 8 pt axis labels, 9 pt bold panel letters.

STANDALONE FIGURES (one image each, same fonts, formal titles)
    Epistatic_Coefficient_per_Locus   epsilon_j for the top loci  (was main a)
    h_ENA1_by_Modifier_Allele_State   h_ENA1 by MKT1/HAL9/PHO84    (was main b)
    Variance_Determined_by_the_Data   share of prior V_k determined
                                      by the data                  (was main d)
    Shapley_Values_RM_Parent          main panel (c) at full width
    Shapley_Cost                      cost per genotype, GKT vs enumeration
    Variance_Cost                     cost of the order spectrum, GKT vs
                                      enumeration
        (both: number of genotypes 2^L on the bottom axis, sequence length L
        on the top axis)
    Shapley_Validation                GKT vs brute-force Shapley values, R^2
    Variance_Component_Validation     GKT vs brute force for E[V_k | y] per
                                      order (--vc-validation picks another
                                      quantity), R^2
    Background_Distribution_vs_Mean_h mean h_j(t) as the reference background
                                      slides from the RM to the BY parent
    Fraction_Retained_by_MAP          V_k(f^MAP) / E[V_k | y]
    Variance_Explained                variance explained by orders <= k, with
                                      delta-method Beta intervals recomputed
                                      from E5_component_covariance.csv
    Distribution_of_h_ENA1            h_ENA1 for all genotypes and the
                                      predicted RM-tolerant set; inset in the
                                      paper's notation
    Per-Locus_Substitution_Effects    h_i across genotypes for the top loci

USAGE
    python scripts/redraw_paper_figures.py \\
        --ena1-dir output/ENA1_paper_rep1 --e5-dir output/E5_full \\
        --out-dir output/paper_figures
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REVISION = "paperfig-rev16-2026-10-06"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Same constants as ENA1_paper_analysis.py (kept local so this script reads
# CSVs only and imports nothing from the analysis code).
GENE_BY_LOCUS = {
    7: "SUS1", 8: "SUP45", 12: "ENA1", 33: "TAO3", 39: "HAL5", 47: "PTR2",
    51: "DPH5", 54: "HAP1", 57: "VIP1", 59: "PHO84", 60: "BUL2", 69: "MKT1",
    70: "MLF3", 74: "HAL9", 75: "IRA2", 77: "NRT1",
}
PAPER_GENES = {"HAL9", "PHO84", "MKT1", "HAP1"}
MODIFIERS = {"MKT1": 69, "HAL9": 74, "PHO84": 59}
ENA1_LOCUS = 12
BY_LETTER, RM_LETTER = "B", "A"
C_BY, C_RM, C_NEU, C_ACC = "#3B7EA1", "#C1553B", "#7A7A7A", "#5B6C8F"
C_INF, C_VAR = "#7B3294", "#7B3294"

TICK, LABEL, LETTER, LEGEND, TITLE = 7, 8, 9, 6.3, 9
SINGLE = (3.25, 2.6)        # standalone figure size (two fit side by side)


def gene_label(locus) -> str:
    g = GENE_BY_LOCUS.get(int(locus))
    return g if g else f"L{int(locus)}"


def style() -> None:
    plt.rcParams.update({
        "font.size": TICK, "axes.labelsize": LABEL, "axes.titlesize": TITLE,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "lines.linewidth": 1.1,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def letter(ax, s: str) -> None:
    ax.set_title(s, loc="left", fontsize=LETTER, fontweight="bold", pad=3)


def title(ax, s: str) -> None:
    ax.set_title(s, fontsize=TITLE, fontweight="bold", pad=4)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {out / name}.png")


def as_bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


def every_third_decade(ax, values) -> None:
    """Log y ticks on multiples of three decades, at most ~5 labels."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if not len(v):
        return
    lo, hi = int(np.floor(np.log10(v.min()))), int(np.ceil(np.log10(v.max())))
    step = 3 * max(1, int(np.ceil((hi - lo) / 15.0)))
    start = hi - step * ((hi - lo) // step)
    ax.set_yticks([10.0 ** e for e in range(start, hi + 1, step)])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(10.0 ** (lo - 0.4), 10.0 ** (hi + 0.4))


def r_squared(x, y, log: bool = False):
    """R^2 of y against the identity line y = x. -> (R^2, 1 - R^2).

    1 - R^2 is returned separately because at machine-precision agreement it
    is ~1e-30, and 1 - 1e-30 rounds to exactly 1.0 in floating point.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if log:
        ok = (x > 0) & (y > 0)
        x, y = np.log10(x[ok]), np.log10(y[ok])
    ss_res = float(((y - x) ** 2).sum())
    ss_tot = float(((x - x.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan"), float("nan")
    gap = ss_res / ss_tot
    return 1.0 - gap, gap


def r2_text(r2: float, gap: float) -> str:
    """R^2 to five decimal places; '~' when the rounding hides a nonzero gap."""
    if not np.isfinite(gap):
        return "$R^2$ undefined"
    sym = "=" if gap == 0 else "\\approx"
    return "$R^2 " + sym + " " + f"{1.0 - gap:.5f}" + "$"


# =========================================================================== #
# loading
# =========================================================================== #

def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing input {path}")
    return path


def optional(path: Path, **kw):
    if path.exists():
        return pd.read_csv(path, **kw)
    print(f"  [missing] {path} -- the figure that needs it is skipped")
    return None


def load_inputs(a) -> dict:
    ena1, e5 = Path(a.ena1_dir), Path(a.e5_dir)
    d = {
        "per_geno": pd.read_csv(need(ena1 / "E2_per_genotype.csv"),
                                usecols=["seq", "h_ENA1", "rm_tolerant",
                                         "ena1_state"]),
        "e3": pd.read_csv(need(ena1 / "E3_compensating_loci.csv")),
        "e4": pd.read_csv(need(ena1 / "E4_pi_sweep.csv")),
        "t_shap": pd.read_csv(need(Path(a.timing_csv) if a.timing_csv
                                   else ena1 / "E1_timing.csv")),
        "vc": pd.read_csv(need(e5 / "E5_figure_table.csv")),
        "per_locus": optional(ena1 / "E2_substitution_effects_by_locus.csv"),
        "long": optional(ena1 / "E2_phi_and_h_long.csv",
                         usecols=["locus", "h"]),
        "val_shap": optional(ena1 / "E1_validation_scatter.csv"),
        "val_vc": optional(e5 / "E5_validation_spectrum.csv"),
        "t_vc": optional(Path(a.e5_timing_csv) if a.e5_timing_csv
                         else e5 / "E5_timing.csv"),
    }
    pdir = Path(a.parental_dir)
    d["par"] = optional(pdir / "SHAP_parental_targets.csv")
    d["par_summ"] = optional(pdir / "SHAP_parental_summary.csv")
    cpath = e5 / "E5_component_covariance.csv"
    d["cov"] = (pd.read_csv(cpath).drop(columns="order").to_numpy()
                if cpath.exists() else None)
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k:<10} {len(v):>9,} rows")
    if d["cov"] is None:
        print("  [missing] E5_component_covariance.csv -- variance-explained "
              "intervals omitted")
    return d


# =========================================================================== #
# variance explained, recomputed from the covariance (read-only)
# =========================================================================== #

def beta_interval(m, s, level=0.95):
    m, s = np.asarray(m, float), np.asarray(s, float)
    lo, hi = np.full(m.shape, np.nan), np.full(m.shape, np.nan)
    tiny = np.isfinite(m) & np.isfinite(s) & (
        (s <= 1e-12 * np.maximum(m, 1e-300)) | (m >= 1) | (m <= 0))
    lo[tiny], hi[tiny] = m[tiny], m[tiny]
    ok = (np.isfinite(m) & np.isfinite(s) & ~tiny & (s ** 2 < m * (1 - m)))
    if ok.any():
        q = (0.5 * (1 - level), 0.5 * (1 + level))
        nu = m[ok] * (1 - m[ok]) / s[ok] ** 2 - 1.0
        try:
            from scipy.stats import beta
            lo[ok] = beta.ppf(q[0], m[ok] * nu, (1 - m[ok]) * nu)
            hi[ok] = beta.ppf(q[1], m[ok] * nu, (1 - m[ok]) * nu)
        except ImportError:
            lo[ok] = np.clip(m[ok] - 1.96 * s[ok], 0, 1)
            hi[ok] = np.clip(m[ok] + 1.96 * s[ok], 0, 1)
    f = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(m)
    lo[f], hi[f] = np.minimum(lo[f], m[f]), np.maximum(hi[f], m[f])
    return lo, hi


def variance_explained(post_mean, cov, level=0.95):
    """Delta-method posterior of F_k = S_k / T from the cross-order covariance."""
    m_all = np.asarray(post_mean, float)[1:]
    C = np.asarray(cov, float)[1:, 1:]
    K = C.shape[0]
    ET = m_all.sum()
    csum = np.cumsum(C, axis=0)
    var_S = np.array([csum[k, :k + 1].sum() for k in range(K)])
    cov_ST = csum.sum(axis=1)
    F = np.cumsum(m_all[:K]) / ET
    vF = (var_S - 2 * F * cov_ST + F ** 2 * C.sum()) / ET ** 2
    lo, hi = beta_interval(F, np.sqrt(np.clip(vF, 0, None)), level)
    return F, lo, hi


# =========================================================================== #
# shared panel builders
# =========================================================================== #

def modifier_labels(seqs) -> pd.DataFrame:
    s = pd.Series([str(x) for x in seqs])
    d = pd.DataFrame({g: s.str[j] for g, j in MODIFIERS.items()})
    d["label"] = d[list(MODIFIERS)].agg("".join, axis=1)
    d["n_by"] = sum((d[g] == BY_LETTER).astype(int) for g in MODIFIERS)
    return d


def panel_compensators(ax, per_geno, compact: bool = True) -> None:
    """h_ENA1 by allele state at MKT1/HAL9/PHO84: eight violins by mean."""
    lab = modifier_labels(per_geno["seq"])
    h = per_geno["h_ENA1"].to_numpy()
    df = pd.DataFrame({"label": lab["label"], "n_by": lab["n_by"], "h": h})
    t = (df.groupby("label").agg(mean=("h", "mean"), n_by=("n_by", "first"))
           .sort_values("mean").reset_index())
    order = list(t["label"])
    groups = [df.loc[df["label"] == l, "h"].to_numpy() for l in order]
    parts = ax.violinplot(groups, positions=range(len(order)), widths=0.85,
                          showextrema=False, showmedians=False)
    cmap = plt.get_cmap("Blues")
    for pc, k in zip(parts["bodies"], t["n_by"]):
        pc.set_facecolor(cmap(0.30 + 0.20 * int(k)))
        pc.set_alpha(0.95)
        pc.set_linewidth(0)
    ax.plot(range(len(order)), t["mean"], "o-", color=C_RM, ms=2.2, lw=0.8,
            zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90 if compact else 0,
                       fontsize=TICK - (1 if compact else 0))
    ax.set_xlabel("MKT1 / HAL9 / PHO84 allele\n(A = RM, B = BY)")
    ax.set_ylabel("$h_{\\mathrm{ENA1}}$")


def panel_distribution(ax, per_geno, show_tolerant: bool = True) -> None:
    """Distribution of h_ENA1 across all genotypes.

    The inset uses the paper's notation, phi_i = (1[x_i = RM] - pi^i_RM) h_i
    (identical in value to the analysis code's phi_i = (p_i - z_i) h_i, since
    1[x_i = RM] - pi^i_RM = (1 - z_i) - (1 - p_i)). show_tolerant overlays the
    model-predicted RM-tolerant set (always drawn): ENA1(RM) genotypes whose
    PREDICTED fitness f(x) -- the GP posterior mean, not the measurement -- is
    at or above the 90th percentile of f(x) among ENA1(RM) genotypes. The set
    is read from E2_per_genotype.csv (column rm_tolerant).
    """
    h = per_geno["h_ENA1"].to_numpy()
    bins = np.linspace(h.min(), h.max(), 55)
    ax.hist(h, bins=bins, color="#8C8C8C", alpha=0.85, label="all genotypes")
    if show_tolerant:
        tol = as_bool(per_geno["rm_tolerant"])
        if tol.any():
            ax.hist(h[tol], bins=bins, color=C_RM, alpha=0.9,
                    label="predicted RM-tolerant")
        ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.1)
    ax.set_xlabel("$h_{\\mathrm{ENA1}}$   (effect of ENA1 BY$\\to$RM)")
    ax.set_ylabel("number of genotypes")
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    ax.text(0.97, 0.96,
            "$\\phi_i = (\\mathbf{1}[x_i = \\mathrm{RM}]"
            " - \\pi^{i}_{\\mathrm{RM}})\\,h_i$",
            transform=ax.transAxes, ha="right", va="top", fontsize=LEGEND,
            bbox=dict(fc="white", ec="0.75", lw=0.5, alpha=0.92, pad=1.5))


def epsilon_table(e3, top: int) -> pd.DataFrame:
    t = e3.copy()
    if "design_identified" in t:
        t = t[as_bool(t["design_identified"])]
    t = t.reindex(t["epsilon_j"].abs().sort_values(ascending=False).index)
    t = t.head(top).copy()
    sd = t.get("gp_posterior_sd_h", t["epsilon_sem"]).to_numpy(dtype=float)
    if "epsilon_posterior_sd" in t:
        e = t["epsilon_posterior_sd"].to_numpy(dtype=float)
        conv = (as_bool(t["epsilon_posterior_converged"])
                if "epsilon_posterior_converged" in t else np.isfinite(e))
        sd = np.where(conv & np.isfinite(e), e, sd)
    t["whisker"] = sd
    return t


def panel_epsilon(ax, e3, top: int) -> None:
    t = epsilon_table(e3, top)
    xs = np.arange(len(t))
    eps = t["epsilon_j"].to_numpy()
    ax.bar(xs, eps, yerr=t["whisker"], width=0.72,
           color=np.where(eps >= 0, C_RM, C_BY),
           error_kw=dict(lw=0.6, ecolor="0.25", capsize=1.2))
    ax.axhline(0, color="k", lw=0.5)
    labs = [gene_label(l) for l in t["locus"]]
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=90, fontsize=TICK - 1)
    for tick, g in zip(ax.get_xticklabels(), labs):
        if g in PAPER_GENES:
            tick.set_fontweight("bold")
    ax.set_xlim(-0.6, len(t) - 0.4)
    ax.set_ylabel("$\\varepsilon_j$")


def gamma_interval(m, sd, level=0.95):
    """Moment-matched Gamma interval for a nonnegative quantity (scalar)."""
    if not (np.isfinite(m) and np.isfinite(sd) and m > 0 and sd > 0):
        return np.nan, np.nan
    q = (0.5 * (1 - level), 0.5 * (1 + level))
    try:
        from scipy.stats import gamma
        sh, sc = (m / sd) ** 2, sd ** 2 / m
        lo, hi = gamma.ppf(q[0], sh, scale=sc), gamma.ppf(q[1], sh, scale=sc)
    except ImportError:
        lo, hi = max(m - 1.96 * sd, 0.0), m + 1.96 * sd
    return min(lo, m), max(hi, m)


def beyond_aggregate(vc, cov, max_order: int, level: float = 0.95) -> dict:
    """V_{k > K}: every order above max_order summed into one component.

    prior / MAP / posterior mean are plain sums over orders K+1..L. The
    posterior range uses the cross-order covariance of the orders it covers
    (the run computed it up to order 10; the share of variance above that is
    ~1e-24, so the omission is immaterial). The data-resolved share is
    sum tr(Gi M_k) / sum tr(A_k K) over the same orders.
    """
    t = vc.sort_values("order")
    beyond = t[t["order"] > max_order]
    row = t[t["order"] == max_order]
    out = {"prior": float(beyond["prior"].sum()),
           "post": float(beyond["post"].sum()),
           "plugin": float(np.clip(beyond["plugin"], 0, None).sum())}
    out["plugin_ok"] = (bool(as_bool(row["tail_plugin_resolved"])[0])
                        if "tail_plugin_resolved" in t and len(row)
                        else bool(as_bool(beyond["resolved_plugin"]).any()))
    out["post_ok"] = (bool(as_bool(row["tail_post_resolved"])[0])
                      if "tail_post_resolved" in t and len(row) else True)
    out["post_lo"] = out["post_hi"] = np.nan
    if cov is not None and cov.shape[0] > max_order + 1:
        idx = np.arange(max_order + 1, cov.shape[0])
        sd = float(np.sqrt(max(cov[np.ix_(idx, idx)].sum(), 0.0)))
        out["post_lo"], out["post_hi"] = gamma_interval(out["post"], sd, level)
    ri = as_bool(beyond["resolved_informed"]) if "resolved_informed" in beyond \
        else np.ones(len(beyond), bool)
    num = float(beyond["tr_Gi_M"].to_numpy(float)[ri].sum()) \
        if "tr_Gi_M" in beyond else np.nan
    den = float(beyond["prior"].sum())
    out["informed"] = num / den if den > 0 and ri.any() else np.nan
    return out


def _beyond_axis(ax, k, max_order: int, compact: bool):
    """x ticks 1..K plus a separated '>K' position; returns that position."""
    xa = float(max_order) + 1.3
    ax.axvline(max_order + 0.65, color="0.6", lw=0.6, ls=":", zorder=0)
    ticks = list(k[::2] if compact else k) + [xa]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(v)) for v in ticks[:-1]] + [f"$>${max_order}"])
    ax.set_xlim(k.min() - 0.6, xa + 0.6)
    return xa


def panel_spectrum(ax, vc, max_order: int, compact: bool = False,
                   cov=None, level: float = 0.95) -> None:
    t = vc[(vc["order"] >= 1) & (vc["order"] <= max_order)]
    k = t["order"].to_numpy()
    rp, rq = as_bool(t["resolved_plugin"]), as_bool(t["resolved_post"])
    pr, pl, po = (t[c].to_numpy(dtype=float) for c in ("prior", "plugin",
                                                       "post"))
    ax.plot(k, pr, "o-", color=C_NEU, ms=4, mfc="white", mew=0.9, lw=0.9,
            label="prior")
    ax.plot(k[rp], pl[rp], "s-", color=C_ACC, ms=2.8, lw=0.9, label="MAP")
    ax.plot(k[rq], po[rq], "^-", color=C_RM, ms=3.2, lw=0.9,
            label="posterior mean")
    if "post_lo" in t:
        lo, hi = t["post_lo"].to_numpy(float), t["post_hi"].to_numpy(float)
        g = rq & np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
        ax.errorbar(k[g], po[g], yerr=np.clip(np.vstack([po[g] - lo[g],
                                                        hi[g] - po[g]]), 0,
                                              None),
                    fmt="none", ecolor=C_RM, elinewidth=0.7, capsize=1.5)
    # V_{k > K}: one aggregate point per estimator, set off by a dotted line
    agg = beyond_aggregate(vc, cov, max_order, level)
    xa = _beyond_axis(ax, k, max_order, compact)
    vals = [pr, pl[rp], po[rq]]
    ax.plot([xa], [agg["prior"]], "o", color=C_NEU, ms=4, mfc="white",
            mew=0.9)
    vals.append([agg["prior"]])
    if agg["plugin_ok"] and agg["plugin"] > 0:
        ax.plot([xa], [agg["plugin"]], "s", color=C_ACC, ms=2.8)
        vals.append([agg["plugin"]])
    if agg["post_ok"] and agg["post"] > 0:
        ax.plot([xa], [agg["post"]], "^", color=C_RM, ms=3.2)
        vals.append([agg["post"]])
        if np.isfinite(agg["post_lo"]) and agg["post_lo"] > 0:
            ax.errorbar([xa], [agg["post"]],
                        yerr=[[agg["post"] - agg["post_lo"]],
                              [agg["post_hi"] - agg["post"]]],
                        fmt="none", ecolor=C_RM, elinewidth=0.7, capsize=1.5)
    ax.set_yscale("log")
    every_third_decade(ax, np.concatenate([np.ravel(v) for v in vals]))
    ax.set_xlabel("order $k$" if compact else "interaction order $k$")
    ax.set_ylabel("$V_k$")
    ax.legend(loc="lower left", handlelength=1.1, borderaxespad=0.1,
              fontsize=LEGEND - (0.8 if compact else 0), labelspacing=0.2)


def panel_resolved(ax, vc, max_order: int, compact: bool = False,
                   cov=None) -> list:
    """Share of the prior order-k variance determined by the data.

    The axis carries no tick above 10^2 (a share cannot exceed 100%); the space
    above 100% exists only to hold the value labels and is sized to them in
    fit_labels_inside() once the layout is final. Returns the label artists.
    """
    t = vc[(vc["order"] >= 1) & (vc["order"] <= max_order)]
    k = t["order"].to_numpy()
    inf = 100.0 * t["informed"].to_numpy(dtype=float)
    ri = as_bool(t["resolved_informed"]) & (inf > 0)
    ax.bar(k[ri], inf[ri], 0.65, color=C_INF)
    labels = []
    for kk, v in zip(k[ri], inf[ri]):
        if kk <= 4:
            labels.append(ax.text(
                kk, v * 1.3, f"{v:.0f}%" if v >= 10 else f"{v:.1f}%",
                ha="center", va="bottom", fontsize=TICK - 1.5, color="0.2",
                rotation=90))
    agg = beyond_aggregate(vc, cov, max_order)
    xa = _beyond_axis(ax, k, max_order, compact)
    vals = list(inf[ri])
    a_inf = 100.0 * agg["informed"]
    if np.isfinite(a_inf) and a_inf > 0:
        ax.bar([xa], [a_inf], 0.65, color=C_INF)
        vals.append(a_inf)
    ax.set_yscale("log")
    lo = int(np.floor(np.log10(min(vals)))) if vals else -2
    span = 2 - lo
    step = 3 * max(1, int(np.ceil(span / 15.0)))
    ax.set_yticks([10.0 ** e for e in range(2, lo - 1, -step)][::-1])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(10.0 ** (lo - 0.4), 10.0 ** 3)
    ax.set_xlabel("order $k$" if compact else "interaction order $k$")
    ax.set_ylabel("% of prior $V_k$\ndetermined" if compact
                  else "% of prior $V_k$ determined")
    return labels


def fit_labels_inside(fig, ax, labels, margin: float = 1.25) -> None:
    """Raise the log y-limit just enough that every label sits inside the axes.

    Text has a fixed size on the page, so the number of decades it spans
    depends on the final axes height; measure after layout and iterate (the
    limit change itself rescales the text's extent in data units).
    """
    if not labels:
        return
    for _ in range(4):
        fig.canvas.draw()
        inv = ax.transData.inverted()
        top = max(inv.transform((0, lab.get_window_extent().y1))[1]
                  for lab in labels)
        lo, hi = ax.get_ylim()
        if top * margin <= hi:
            break
        ax.set_ylim(lo, top * margin)
    print(f"    panel (d) y-limit {ax.get_ylim()[1]:.3g}% (highest tick 100%)")


# =========================================================================== #
# main figure
# =========================================================================== #

def fig_main(d, a, out: Path) -> None:
    """Main figure, 2 x 2, timing comparisons in the left column so the
    panel order follows the text:
      (a) Shapley cost                 (b) Shapley values of the RM parent
      (c) variance-component cost      (d) order spectrum (k > 8 aggregate)
    """
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.9),
                             constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    draw_cost_shapley(ax_a, d["t_shap"])
    if d["par"] is not None:
        panel_parental(ax_b, d["par"], d["par_summ"], a.parental_target)
    else:
        ax_b.axis("off")
        ax_b.text(0.5, 0.5, "no SHAP_parental_targets.csv\n(run SHAP_ENA1 "
                            "--stages parental)", ha="center", va="center",
                  transform=ax_b.transAxes, fontsize=LABEL, color="0.4")
    if d["t_vc"] is not None:
        draw_cost_variance(ax_c, d["t_vc"])
    else:
        ax_c.axis("off")
    panel_spectrum(ax_d, d["vc"], a.max_order, cov=d["cov"], level=a.level)
    for ax, s_ in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        letter(ax, s_)
    align_letters(fig, [(ax_a, ax_b), (ax_c, ax_d)])
    save(fig, out, "Yeast_ENA1_Sequence-Function_Map")


def align_letters(fig, rows) -> None:
    """Put each row's panel letters at the same height.

    The left panels carry a sequence-length axis on top, so their letters sit
    above it while the right panels' letters sit at the plain axes top. After
    the layout is final, freeze it and lift each right-hand letter to the
    height of its row's left-hand letter.
    """
    fig.canvas.draw()
    try:
        fig.set_layout_engine("none")                 # matplotlib >= 3.6
    except AttributeError:
        fig.set_constrained_layout(False)
    r = fig.canvas.get_renderer()
    for left, right in rows:
        # letter() uses loc="left", i.e. the axes' LEFT title artist
        y_px = left._left_title.get_window_extent(r).y0
        y_ax = right.transAxes.inverted().transform((0, y_px))[1]
        right.set_title(right._left_title.get_text(), loc="left", y=y_ax,
                        pad=0, fontsize=LETTER, fontweight="bold",
                        va="bottom")


# ---------- panels moved out of the main figure, as standalones ----------

def fig_epsilon(e3, out: Path, top: int) -> None:
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    panel_epsilon(ax, e3, top)
    title(ax, "Epistatic Coefficient $\\varepsilon_j$ per Locus")
    save(fig, out, "Epistatic_Coefficient_per_Locus")


def fig_compensators(per_geno, out: Path) -> None:
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    panel_compensators(ax, per_geno, compact=False)
    title(ax, "$h_{\\mathrm{ENA1}}$ by Modifier Allele State")
    save(fig, out, "h_ENA1_by_Modifier_Allele_State")


def fig_determined(vc, cov, out: Path, max_order: int) -> None:
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    labels = panel_resolved(ax, vc, max_order, cov=cov)
    title(ax, "Variance Determined by the Data")
    fit_labels_inside(fig, ax, labels)
    save(fig, out, "Variance_Determined_by_the_Data")


# =========================================================================== #
# standalone figures
# =========================================================================== #

def _cost_axes(ax, L_values, ylabel: str) -> None:
    """Bottom: number of genotypes 2^L (log). Top: sequence length L."""
    L_values = np.asarray(L_values, dtype=float)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of genotypes, $2^L$")
    ax.set_ylabel(ylabel)
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    L_max = int(L_values.max())
    Lt = [v for v in (10, 20, 40, 60) if v < L_max - 5] + [L_max]
    Lt = [v for v in Lt if v >= L_values.min()]
    top.set_xticks([2.0 ** v for v in Lt])
    top.set_xticklabels([str(v) for v in Lt], fontsize=TICK)
    top.minorticks_off()
    top.set_xlabel("sequence length $L$", fontsize=LABEL)
    top.spines["top"].set_visible(True)
    return top


def draw_cost_shapley(ax, t):
    """Shapley cost per genotype: GKT mean and mean + posterior variance
    against enumeration (measured points with the dotted projection merged
    into one legend entry). Returns the top (sequence-length) axis."""
    g = t.dropna(subset=["seconds_gkt"]).sort_values("L")
    L_max = int(g["L"].max())
    X = lambda L: 2.0 ** np.asarray(L, dtype=float)              # noqa: E731
    hs = []
    ax.plot(X(g["L"]), g["seconds_gkt"], "o-", ms=2.6, color=C_ACC)
    hs.append(Line2D([], [], color=C_ACC, marker="o", ms=2.6,
                     label="GKT, mean"))
    if "seconds_gkt_variance" in t:
        gv = t.dropna(subset=["seconds_gkt_variance"]).sort_values("L")
        ax.plot(X(gv["L"]), gv["seconds_gkt_variance"], "^-", ms=2.6,
                color=C_BY)
        hs.append(Line2D([], [], color=C_BY, marker="^", ms=2.6,
                         label="GKT, mean + posterior variance"))
    b = t.dropna(subset=["seconds_brute_force"]).sort_values("L")
    if len(b):
        row = b.loc[b["L"].idxmax()]
        unit = float(row["seconds_brute_force"]) / float(row["total_terms"])
        Ls = np.arange(int(b["L"].min()), L_max + 1, dtype=float)
        ax.plot(X(b["L"]), b["seconds_brute_force"], "s", ms=2.6, color=C_RM)
        ax.plot(X(Ls), unit * Ls * 2.0 ** (Ls - 1), ":", color=C_RM)
        hs.append(Line2D([], [], color=C_RM, marker="s", ms=2.6, ls=":",
                         label="enumeration, mean (projected)"))
    if "seconds_brute_force_variance" in t:
        bv = t.dropna(subset=["seconds_brute_force_variance"]).sort_values("L")
        if len(bv):
            uv = (float(bv["seconds_brute_force_variance"].iloc[-1])
                  / 4.0 ** float(bv["L"].iloc[-1]))
            Lv = np.arange(int(bv["L"].min()), L_max + 1, dtype=float)
            ax.plot(X(bv["L"]), bv["seconds_brute_force_variance"], "D",
                    ms=2.4, color=C_VAR)
            ax.plot(X(Lv), uv * 4.0 ** Lv, ":", color=C_VAR)
            hs.append(Line2D([], [], color=C_VAR, marker="D", ms=2.4, ls=":",
                             label="enumeration, variance (projected)"))
    top = _cost_axes(ax, t["L"], "seconds per genotype")
    ax.legend(handles=hs, loc="upper left", handlelength=1.8)
    return top


def draw_cost_variance(ax, t):
    """Cost of the order spectrum, GKT against enumeration.

    What is timed (E5 timing stage): one call that returns the posterior mean
    E[V_k | y] -- together with V_k(f^MAP) and the prior tr(A_k K) -- for
    EVERY order k at once; enumeration computes the same quantities. The
    posterior variances of V_k are not part of this timing. Legend wording
    matches the Shapley cost panel ("GKT, mean ..."). Returns the top
    (sequence-length) axis."""
    X = lambda L: 2.0 ** np.asarray(L, dtype=float)              # noqa: E731
    hs = []
    g = t.dropna(subset=["seconds_gkt"]).sort_values("L")
    ax.plot(X(g["L"]), g["seconds_gkt"], "o-", ms=2.6, color=C_ACC)
    hs.append(Line2D([], [], color=C_ACC, marker="o", ms=2.6,
                     label="GKT, mean of all $V_k$"))
    b = t.dropna(subset=["seconds_brute_force"]).sort_values("L")
    if len(b):
        ax.plot(X(b["L"]), b["seconds_brute_force"], "s", ms=2.6, color=C_RM)
        if "projected_brute_force" in t:
            pr = t[t["L"] >= int(b["L"].max())].sort_values("L")
            ax.plot(X(pr["L"]), pr["projected_brute_force"], ":", color=C_RM)
        hs.append(Line2D([], [], color=C_RM, marker="s", ms=2.6, ls=":",
                         label="enumeration, mean of all $V_k$ (projected)"))
    top = _cost_axes(ax, t["L"], "seconds for the order spectrum")
    ax.legend(handles=hs, loc="upper left", handlelength=1.8)
    return top


def fig_cost_shapley(t, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE[0], 2.9), constrained_layout=True)
    draw_cost_shapley(ax, t)
    title(ax, "Shapley Cost")
    save(fig, out, "Shapley_Cost")


def fig_cost_variance(t, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE[0], 2.9), constrained_layout=True)
    draw_cost_variance(ax, t)
    title(ax, "Variance Cost")
    save(fig, out, "Variance_Cost")


# ---------- Shapley values of one parental genotype ----------

def panel_parental(ax, par, summ, target: str = "RM",
                   wide: bool = False) -> None:
    """phi_i of one parental genotype at every locus, genome order.

    Raw Shapley values (not h_i) under the empirical background, with
    +-1 posterior s.d. whiskers; mapped QTLs labelled. The inset gives the
    genotype's predicted fitness f(x).
    """
    t = par[par["target"] == target].sort_values("locus")
    loci = t["locus"].to_numpy()
    phi = t["phi"].to_numpy(float)
    sd = t["phi_sd"].to_numpy(float)
    ax.bar(loci, phi, width=0.8, color=np.where(phi >= 0, C_RM, C_BY),
           yerr=np.nan_to_num(sd), error_kw=dict(lw=0.5, ecolor="0.3",
                                                 capsize=0))
    ax.axhline(0, color="k", lw=0.5)
    named = [int(l) for l in loci if GENE_BY_LOCUS.get(int(l))]
    # Ticks at every mapped QTL; names only where they cannot collide: walk
    # the loci by |phi| and skip any within `gap` positions of one already
    # named (adjacent pairs like MKT1/MLF3 keep the larger effect).
    mag = dict(zip(loci.astype(int), np.abs(phi)))
    gap = 3 if wide is False else 1
    chosen = []
    key_loci = [ENA1_LOCUS] + list(MODIFIERS.values()) + [54]   # + HAP1
    order = ([l for l in key_loci if l in named]
             + sorted((l for l in named if l not in key_loci),
                      key=lambda q: -mag.get(q, 0.0)))
    for l in order:
        if all(abs(l - c) > gap for c in chosen):
            chosen.append(l)
    ax.set_xticks(named)
    ax.set_xticklabels([gene_label(l) if l in chosen else "" for l in named],
                       rotation=90, fontsize=TICK - 1)
    for tick, l in zip(ax.get_xticklabels(), named):
        if l == ENA1_LOCUS:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    ax.set_xlim(loci.min() - 1, loci.max() + 1)
    ax.set_xlabel("locus (genome order)")
    ax.set_ylabel(f"$\\phi_i$  ({target} parent)")
    if summ is not None and (summ["target"] == target).any():
        r = summ[summ["target"] == target].iloc[0]
        txt = "predicted fitness $f(x) = " + f"{r['f_x']:+.3f}" + "$"
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right",
                va="bottom", multialignment="left", fontsize=LEGEND,
                bbox=dict(fc="white", ec="0.75", lw=0.5, alpha=0.92, pad=1.5))


def fig_parental(par, summ, out: Path, target: str = "RM") -> None:
    fig, ax = plt.subplots(figsize=(6.75, 2.4), constrained_layout=True)
    panel_parental(ax, par, summ, target, wide=True)
    title(ax, f"Shapley Values of the {target} Parent")
    save(fig, out, f"Shapley_Values_{target}_Parent")


def fig_validation_shapley(v, out: Path) -> None:
    x, y = v["brute_force"].to_numpy(float), v["gkt"].to_numpy(float)
    r2, gap = r_squared(x, y)
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    lim = float(np.abs(np.r_[x, y]).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.scatter(x, y, s=14, color=C_ACC, ec="k", lw=0.3, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("brute force $\\phi_i$  ($2^{L-1}$ subsets)")
    ax.set_ylabel("GKT $\\phi_i$")
    ax.text(0.04, 0.96, f"{r2_text(r2, gap)}\nmax $|\\Delta|$ = "
                        f"{np.abs(x - y).max():.1e}\n$L = {len(x)}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=TICK)
    title(ax, "Shapley Validation")
    save(fig, out, "Shapley_Validation")
    print(f"    Shapley validation: 1 - R^2 = {gap:.2e}")


VC_QUANTITIES = {
    "posterior": ("V_k_posterior_mean", "$E[V_k\\,|\\,y]$"),
    "map": ("V_k_plugin", "$V_k(f^{MAP})$"),
    "plugin": ("V_k_plugin", "$V_k(f^{MAP})$"),       # old name, same thing
    "prior": ("prior_tr_AK", "$\\mathrm{tr}(A_kK)$"),
    "trace": ("tr_A_Sigma", "$\\mathrm{tr}(A_k\\Sigma)$"),
}


def fig_validation_vc(v, out: Path, quantity: str = "posterior") -> None:
    """GKT against brute-force enumeration for ONE variance-component quantity.

    Default: the posterior mean E[V_k | y] per order k -- the quantity the
    paper reports, which brute force computes by building every A_k and
    Sigma over all 2^L genotypes.
    """
    col, lab = VC_QUANTITIES[quantity]
    x = v[f"{col}_bf"].to_numpy(float)
    y = v[f"{col}_gkt"].to_numpy(float)
    k = v["order"].to_numpy()
    ok = (x > 0) & (y > 0)
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    lo, hi = x[ok].min() / 5, x[ok].max() * 5
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.scatter(x[ok], y[ok], s=14, color=C_ACC, ec="k", lw=0.3, zorder=3)
    for kk, xx, yy in zip(k[ok], x[ok], y[ok]):
        ax.annotate(f"$k={int(kk)}$", (xx, yy), xytext=(4, -6),
                    textcoords="offset points", fontsize=TICK - 1.5,
                    color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    r2, gap = r_squared(x[ok], y[ok], log=True)
    rel = float(np.max(np.abs(x[ok] - y[ok]) / np.abs(x[ok])))
    ax.text(0.04, 0.96, f"{r2_text(r2, gap)}  (log scale)\nmax relative "
                        f"$|\\Delta|$ = {rel:.1e}\n$L = {len(k) - 1}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=TICK)
    ax.set_xlabel(f"brute force {lab}")
    ax.set_ylabel(f"GKT {lab}")
    title(ax, "Variance Component Validation")
    save(fig, out, "Variance_Component_Validation")
    print(f"    variance-component validation ({col}): 1 - R^2 (log) = "
          f"{gap:.2e}")


def fig_background(e4, out: Path) -> None:
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    end = {}
    for loc, grp in e4.groupby("locus"):
        grp = grp.sort_values("t")
        end[int(loc)] = (grp["t"].to_numpy(), grp["mean_h"].to_numpy())
    # Highlight ENA1 and the three loci with the largest endpoint asymmetry;
    # everything else in grey.
    asym = sorted((l for l in end if l != ENA1_LOCUS),
                  key=lambda l: -abs(end[l][1][0] - end[l][1][-1]))[:3]
    key = [ENA1_LOCUS] + asym
    colours = {ENA1_LOCUS: C_RM}
    for l, c in zip(asym, ["#2E7D32", "#6A3D9A", "#E08E0B"]):
        colours[l] = c
    for l, (tt, hh) in end.items():
        if l not in key:
            ax.plot(tt, hh, "-", color="0.75", lw=0.6, zorder=1)
    for l in key:
        tt, hh = end[l]
        ax.plot(tt, hh, "-", color=colours[l],
                lw=1.8 if l == ENA1_LOCUS else 1.2, zorder=3,
                label=gene_label(l))
    ax.plot([], [], "-", color="0.75", lw=0.6, label="other mapped loci")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("$t$   (0 = RM parent, 1 = BY parent)")
    ax.set_ylabel("mean $h_j(t)$")
    ax.legend(loc="lower right", handlelength=1.5)
    title(ax, "Background Distribution vs. Mean $h_j$")
    save(fig, out, "Background_Distribution_vs_Mean_h")


def fig_fraction_retained(vc, out: Path, max_order: int) -> None:
    t = vc[(vc["order"] >= 1) & (vc["order"] <= max_order)]
    ok = as_bool(t["resolved_plugin"]) & as_bool(t["resolved_post"])
    k = t["order"].to_numpy()[ok]
    r = (t["plugin"] / t["post"]).to_numpy(dtype=float)[ok]
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.plot(k, r, "o-", ms=3, color=C_ACC)
    for kk, rr in zip(k, r):
        ax.annotate(f"{rr:.2f}" if rr >= 0.01 else f"{rr:.0e}", (kk, rr),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=TICK - 1, color=C_ACC)
    ax.set_yscale("log")
    every_third_decade(ax, np.r_[r, 2.0])
    ax.set_xticks(t["order"].to_numpy())
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("$V_k(f^{MAP})\\,/\\,E[V_k\\,|\\,y]$")
    title(ax, "Fraction Retained by the MAP")
    save(fig, out, "Fraction_Retained_by_MAP")


def fig_variance_explained(vc, cov, out: Path, max_order: int,
                           level: float) -> None:
    t = vc[vc["order"] >= 1].sort_values("order")
    pl = np.clip(t["plugin"].to_numpy(dtype=float), 0, None)
    cp = np.cumsum(pl) / pl.sum()
    po_all = vc.sort_values("order")["post"].to_numpy(dtype=float)
    cq = np.cumsum(po_all[1:]) / po_all[1:].sum()
    kk = t["order"].to_numpy()
    sel = kk <= max_order
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    ax.plot(kk[sel], cp[sel], "o-", ms=3, color=C_ACC, label="MAP")
    ax.plot(kk[sel], cq[sel], "s-", ms=3, color=C_RM,
            label="posterior" + (f" ({int(100 * level)}% range)"
                                 if cov is not None else ""))
    ylo = float(np.nanmin(np.r_[cp[sel], cq[sel]]))
    if cov is not None:
        F, lo, hi = variance_explained(po_all, cov, level)
        n = min(len(F), int(sel.sum()))
        x = kk[:n]
        g = np.isfinite(lo[:n]) & np.isfinite(hi[:n])
        ax.errorbar(x[g], cq[:n][g],
                    yerr=np.clip(np.vstack([cq[:n][g] - lo[:n][g],
                                            hi[:n][g] - cq[:n][g]]), 0, None),
                    fmt="none", ecolor=C_RM, elinewidth=0.8, capsize=2)
        if g.any():
            ylo = min(ylo, float(np.nanmin(lo[:n][g])))
    pad = 0.08 * (1 - ylo)
    ax.set_ylim(max(0.0, ylo - pad), 1.0 + 0.5 * pad)
    ax.set_xticks(kk[sel])
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("variance explained by orders $\\leq k$")
    ax.legend(loc="lower right")
    title(ax, "Variance Explained by Order")
    save(fig, out, "Variance_Explained")


def fig_distribution(per_geno, out: Path, show_tolerant: bool = True) -> None:
    """The original distribution of h_ENA1, standalone."""
    fig, ax = plt.subplots(figsize=SINGLE, constrained_layout=True)
    panel_distribution(ax, per_geno, show_tolerant=show_tolerant)
    title(ax, "Distribution of $h_{\\mathrm{ENA1}}$")
    save(fig, out, "Distribution_of_h_ENA1")


def fig_per_locus(per_locus, long, out: Path, n_show: int) -> None:
    """h_i across genotypes for the top loci (the old E2 panel b)."""
    pl = per_locus.copy()
    if "is_focal_qtl" in pl and "varies_in_targets" in pl:
        pl = pl[as_bool(pl["is_focal_qtl"]) | as_bool(pl["varies_in_targets"])]
    have = set(int(j) for j in long["locus"].unique())
    pl = pl[pl["locus"].astype(int).isin(have)]
    pl = pl.reindex(pl["mean_h"].abs().sort_values(ascending=False).index)
    order = [int(j) for j in pl["locus"].head(n_show)]
    groups = [long.loc[long["locus"] == j, "h"].to_numpy() for j in order]
    fig, ax = plt.subplots(figsize=(6.75, 2.6), constrained_layout=True)
    parts = ax.violinplot(groups, positions=range(len(order)), widths=0.85,
                          showextrema=False, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_ACC)
        pc.set_alpha(0.7)
        pc.set_linewidth(0)
    if "cmedians" in parts:
        parts["cmedians"].set_color("k")
        parts["cmedians"].set_linewidth(0.9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([gene_label(j) for j in order], rotation=90,
                       fontsize=TICK)
    for tick, j in zip(ax.get_xticklabels(), order):
        if j == ENA1_LOCUS:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylabel("$h_i$")
    ax.set_xlabel("locus, ordered by $|$mean $h_i|$")
    title(ax, "Per-Locus Effect Across Genotypes")
    save(fig, out, "Per-Locus_Substitution_Effects")


# =========================================================================== #
# main
# =========================================================================== #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ena1-dir", type=Path,
                    default=REPO_ROOT / "output" / "ENA1_paper_rep1")
    ap.add_argument("--e5-dir", type=Path,
                    default=REPO_ROOT / "output" / "E5_full")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="new folder for the figures (default "
                         "output/paper_figures_<timestamp>)")
    ap.add_argument("--parental-dir", type=Path,
                    default=REPO_ROOT / "output" / "SHAP_parental_rep1",
                    help="output of SHAP_ENA1_experiment.py --stages parental "
                         "(SHAP_parental_targets.csv / _summary.csv)")
    ap.add_argument("--parental-target", choices=["RM", "BY"], default="RM",
                    help="parent whose Shapley values go in main panel (c)")
    ap.add_argument("--timing-csv", type=Path, default=None,
                    help="Shapley timings (default <ena1-dir>/E1_timing.csv)")
    ap.add_argument("--e5-timing-csv", type=Path, default=None,
                    help="variance-component timings (default "
                         "<e5-dir>/E5_timing.csv)")
    ap.add_argument("--top-eps", type=int, default=12,
                    help="loci shown in main panel (b)")
    ap.add_argument("--n-loci", type=int, default=30,
                    help="loci in the per-locus effect figure")
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--vc-validation", choices=sorted(VC_QUANTITIES),
                    default="posterior",
                    help="which variance-component quantity the validation "
                         "figure compares (default: posterior mean E[V_k|y])")
    a = ap.parse_args()

    def absolute(p: Path) -> Path:
        p = Path(p)
        return (p if p.is_absolute() else REPO_ROOT / p).resolve()

    a.ena1_dir, a.e5_dir = absolute(a.ena1_dir), absolute(a.e5_dir)
    a.parental_dir = absolute(a.parental_dir)
    out = absolute(a.out_dir) if a.out_dir else (
        REPO_ROOT / "output"
        / f"paper_figures_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    for src in (a.ena1_dir, a.e5_dir, a.parental_dir):
        if out == src or src in out.parents:
            print(f"!! --out-dir {out} is inside the source folder {src}; "
                  "choose a separate folder so the run outputs stay untouched")
            return 1
    out.mkdir(parents=True, exist_ok=True)

    print(f"redraw_paper_figures {REVISION}")
    print(f"  ENA1 run (read only): {a.ena1_dir}")
    print(f"  E5 run   (read only): {a.e5_dir}")
    print(f"  parental (read only): {a.parental_dir}")
    print(f"  writing to          : {out}")
    d = load_inputs(a)
    style()

    fig_main(d, a, out)
    # panels that left the main figure, now standalone (appendix)
    fig_epsilon(d["e3"], out, a.top_eps)
    fig_compensators(d["per_geno"], out)
    fig_determined(d["vc"], d["cov"], out, a.max_order)
    if d["par"] is not None:
        fig_parental(d["par"], d["par_summ"], out, a.parental_target)
    fig_cost_shapley(d["t_shap"], out)
    if d["t_vc"] is not None:
        fig_cost_variance(d["t_vc"], out)
    if d["val_shap"] is not None:
        fig_validation_shapley(d["val_shap"], out)
    if d["val_vc"] is not None:
        fig_validation_vc(d["val_vc"], out, a.vc_validation)
    fig_background(d["e4"], out)
    fig_fraction_retained(d["vc"], out, a.max_order)
    fig_variance_explained(d["vc"], d["cov"], out, a.max_order, a.level)
    fig_distribution(d["per_geno"], out)
    if d["per_locus"] is not None and d["long"] is not None:
        fig_per_locus(d["per_locus"], d["long"], out, a.n_loci)

    (out / "REDRAW_NOTES.txt").write_text("\n".join([
        f"redraw_paper_figures.py {REVISION}",
        f"ENA1 source: {a.ena1_dir}",
        f"E5 source:   {a.e5_dir}",
        "No model was run and no source file was modified.",
        f"top {a.top_eps} loci in main panel (b); orders 1-{a.max_order}; "
        f"{a.n_loci} loci in the per-locus figure",
    ]) + "\n")
    print(f"  wrote {out / 'REDRAW_NOTES.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())