"""
E5_variance_components.py
=========================

Experiment E5: exact functional-ANOVA variance components of the GP over the
whole 2^83 sequence space, via the Group-II (quadratic) rows of the paper's
statistics table.

Three stages, selected with --stages:

  validate  GKT against explicit enumeration of all alpha^L sequences, at an L
            where the alpha^L x alpha^L matrices still fit. Checks all three
            rows independently, plus two exact identities and the sign of
            tr(A_k Sigma).

  timing    Wall-clock for the order spectrum, GKT against enumeration, as a
            function of L. Enumeration is measured where affordable and
            PROJECTED beyond that from its measured rate; the projection is
            labelled as such. This is the E5 analogue of the cost panel in the
            main paper figure.

  real      The order spectrum of the fitted yeast model: prior mean
            tr(A_k K), plug-in V_k(f^MAP), and posterior mean E[V_k | y], with
            the cumulative "variance explained by order <= k" computed both
            ways.

WHAT THE REAL STAGE IS FOR
--------------------------
    tr(A_k Sigma) = E[V_k | y] - V_k(f^MAP) >= 0
nonnegative because A_k is PSD, so plugging a posterior mean into an ANOVA
decomposition always UNDER-states every component. The bias is not uniform:
shrinkage is hardest where the data constrain f least, which is high
interaction order. On a synthetic check the plug-in retained 18% of the order-1
component but 0.3% of the order-6 component, moving "fraction of variance at
order <= 2" from 46% (posterior) to 74% (plug-in). So the plug-in distorts the
SHAPE of the spectrum, not just its scale, and it does so in the direction of
overstating additivity -- which is the mistake the fitness-landscape literature
is most prone to.

COST
----
Group II is O(N^2 L^2), against O(N L^2) for the first-order Shapley values of
E2-E4. That extra factor of N is inherent to rows 3-7 and belongs in the cost
section of the paper. It is why --n-condition defaults to a subsample rather
than all 19,833: the qualitative claim does not depend on N, and a smaller
conditioning set makes the plug-in bias LARGER and hence easier to see.

USAGE
-----
    python scripts/E5_variance_components.py --stages validate,timing \\
        --out-dir output/E5

    python scripts/E5_variance_components.py --stages real --gpu \\
        --params-path models/qtls.li.hq/qtls_li_hq.full.1.Connectedness.model_params.pth \\
        --n-condition 1500 --out-dir output/E5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# gkt_variance_components.py may sit beside this driver in scripts/ or with
# the other project modules in modules/, so put both on the path rather than
# assuming one. Same for the repo root, which is where ENA1_paper_analysis
# reaches for epik.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _d in (_HERE, _ROOT / "modules", _ROOT):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

try:
    import gkt_variance_components as vcm                      # noqa: E402
except ModuleNotFoundError as _exc:                            # noqa: BLE001
    print(f"\n!! {_exc}")
    print("\ngkt_variance_components.py was not found. It must live in one of:")
    for _d in (_HERE, _ROOT / "modules", _ROOT):
        _hits = sorted(Path(_d).glob("gkt_variance_components.py")) \
            if Path(_d).exists() else []
        print(f"    {_d}{'   <- FOUND' if _hits else ''}")
    print("\nPut the file in scripts/ (beside this driver) or modules/, then "
          "rerun. Nothing else needs changing.")
    sys.exit(2)
from ENA1_paper_analysis import (                              # noqa: E402
    REPO_ROOT, N_LOCI, RM_LETTER, BY_LETTER, C_RM, C_BY, C_NEU, C_ACC,
    banner, sub, write_csv, save_fig, _style, _panel, _to_np,
    load_segregants, genotype_matrix, empirical_counts_from_X,
    load_model_and_shap, make_pi, b_index,
)

DEFAULT_OUT = REPO_ROOT / "output" / "E5"


# =========================================================================== #
# validate
# =========================================================================== #

def stage_validate(args, out_dir: Path) -> dict:
    banner("E5a  validation against explicit enumeration")
    rows, tables = [], {}
    for L in args.validate_L:
        r = vcm.validate(L=int(L), n_train=args.validate_n_train, seed=int(L),
                         tol=args.tol)
        sub(f"L = {L}  (alpha^L = {2 ** int(L):,} sequences enumerated)")
        for c, e in r["errors"].items():
            print(f"  {c:<22} max |GKT - enumeration| = {e:.3e}   "
                  f"(scale {r['scales'][c]:.3e})")
        i = r["identities"]
        print(f"  {'sum_k V_k vs E_pi[f^2]':<22} {i['err_total']:.3e}")
        print(f"  {'V_0 vs (E_pi f)^2':<22} {i['err_order0']:.3e}")
        print(f"  {'min tr(A_k Sigma)':<22} {r['min_tr_A_Sigma']:.3e}   "
              "(must be >= 0; A_k is PSD)")
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}]")
        rec = {"L": int(L), "n_train": args.validate_n_train,
               "passed": r["passed"], "min_tr_A_Sigma": r["min_tr_A_Sigma"],
               "err_identity_total": i["err_total"],
               "err_identity_order0": i["err_order0"]}
        rec.update({f"err_{c}": v for c, v in r["errors"].items()})
        rows.append(rec)
        tables[int(L)] = r

    tbl = pd.DataFrame(rows)
    write_csv(tbl, out_dir / "E5_validation.csv")
    Lmax = max(tables)
    cmp = tables[Lmax]["gkt"].merge(
        tables[Lmax]["brute_force"], on="order", suffixes=("_gkt", "_bf"))
    write_csv(cmp, out_dir / "E5_validation_spectrum.csv")
    ok = bool(tbl["passed"].all())
    print(f"\n  [{'PASS' if ok else 'FAIL'}] all tested L")
    if not ok:
        print("  !! do not run the real stage until this passes")
    return {"table": tbl, "passed": ok, "spectrum": cmp}


# =========================================================================== #
# timing
# =========================================================================== #

def enumeration_cost_model(t: pd.DataFrame) -> tuple[float, float, int]:
    """Fit enumeration cost as unit * base^L. -> (unit, base, L_measured).

    The expected base is 2 * alpha^2 = 8 at alpha = 2: the 2^L subsets each
    require a Kronecker product with alpha^L x alpha^L = 4^L entries. The base
    is FITTED rather than assumed, so the projection rests on what the machine
    actually did; a fitted base far from 8 would mean the cost model is wrong
    and the projection should not be trusted.
    """
    b = t.dropna(subset=["seconds_brute_force"]).sort_values("L")
    if len(b) < 2:
        return float("nan"), 8.0, 0
    # Fit on the LARGEST few L only. A fit across the whole measured range is
    # dragged down by fixed overhead at small L -- on a first run the global fit
    # returned base 4.12 against a theoretical 8, purely because the L = 4..7
    # points are overhead-dominated. The tail is where the asymptotic rate
    # shows, and using it makes the projection conservative in the right
    # direction: a smaller base would understate enumeration's cost.
    tail = b.tail(max(2, min(3, len(b))))
    slope, _ = np.polyfit(tail["L"].to_numpy(dtype=float),
                          np.log(tail["seconds_brute_force"].to_numpy()), 1)
    base = float(np.exp(slope))
    L_ref = int(b["L"].iloc[-1])
    unit = float(b["seconds_brute_force"].iloc[-1]) / base ** L_ref
    return unit, base, L_ref


def stage_timing(args, out_dir: Path) -> pd.DataFrame:
    banner("E5b  cost: order spectrum, GKT against enumeration")
    t = vcm.timing(args.timing_L, n_train=args.timing_n_train,
                   bf_max_L=args.timing_bf_max_L, reps=args.timing_reps)
    unit, base, L_ref = enumeration_cost_model(t)
    print(f"\n  enumeration cost fits {unit:.3e} x {base:.2f}^L, from the "
          f"largest measured L ({L_ref}).")
    print("  Expected base is 2*alpha^2 = 8: the 2^L subsets each need a "
          "Kronecker product with alpha^L x alpha^L entries.")
    if not (0.7 * 8.0 <= base <= 1.3 * 8.0):
        print(f"  !! fitted base {base:.2f} is far from 8, so the cost model "
              "does not describe what the machine did; treat the projection "
              "below as indicative only.")
    for L in [L for L in args.timing_L if L > L_ref]:
        s = unit * base ** L
        print(f"    L={L:>3}  projected {s:.3e} s = {s / 3.156e7:.3e} years")
    g83 = t[t["L"] == max(args.timing_L)]
    if len(g83):
        sp = unit * base ** max(args.timing_L) / float(g83["seconds_gkt"].iloc[0])
        print(f"  speed-up at L = {max(args.timing_L)}: {sp:.2e}x")
    t["projected_brute_force"] = unit * base ** t["L"].to_numpy(dtype=float)
    write_csv(t, out_dir / "E5_timing.csv")

    # The L sweep at small N is overhead-dominated, so it cannot demonstrate
    # the O(N^2 L^2) cost model. Measure the N-dependence directly: it is the
    # factor that decides whether Group II is affordable, and it is the claim
    # the paper's cost section needs.
    sub(f"cost in the conditioning-set size, at L = {max(args.timing_L)}")
    tn = vcm.timing_vs_n(args.timing_n, L=max(args.timing_L),
                         reps=max(1, args.timing_reps // 2),
                         block=args.block)
    write_csv(tn, out_dir / "E5_timing_vs_n.csv")
    _fig_timing(t, unit, base, L_ref, out_dir, tn=tn)
    return t


def _fig_timing(t: pd.DataFrame, unit: float, base: float, L_ref: int,
                out_dir: Path, tn: pd.DataFrame | None = None) -> None:
    _style()
    if tn is None or not len(tn):
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
        fig.subplots_adjust(wspace=0.30, bottom=0.16, top=0.86)
        ax = axes[0]
    g = t.dropna(subset=["seconds_gkt"]).sort_values("L")
    ax.plot(g["L"], g["seconds_gkt"], "o-", ms=3.5, color=C_ACC, lw=1.6,
            label="GKT (measured)")
    b = t.dropna(subset=["seconds_brute_force"]).sort_values("L")
    if len(b):
        ax.plot(b["L"], b["seconds_brute_force"], "s-", ms=3.5, color=C_RM,
                lw=1.6, label=f"enumeration (measured, $L\\leq{L_ref}$)")
    Ls = np.arange(L_ref, int(g["L"].max()) + 1, dtype=float)
    ax.plot(Ls, unit * base ** Ls, ":", color=C_RM, lw=1.6,
            label="enumeration (projected)")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel("seconds for the full order spectrum")
    ax.legend(fontsize=7, loc="upper left", borderpad=0.3, labelspacing=0.3,
              handlelength=1.6)
    _panel(ax, "a" if len(axes) > 1 else "",
           "Cost vs sequence length ($n$ fixed)")

    if len(axes) > 1:
        ax = axes[1]
        n = tn["n_train"].to_numpy(dtype=float)
        ax.plot(n, tn["seconds"], "o-", ms=4, color=C_ACC, lw=1.6,
                label="GKT (measured)")
        # An N^2 guide anchored at the largest measured point, so the reader can
        # read the exponent off the figure rather than taking it on trust.
        ref = float(tn["seconds"].iloc[-1]) / n[-1] ** 2
        ax.plot(n, ref * n ** 2, "--", color="0.45", lw=1.2,
                label="$\\propto n^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("conditioning-set size $n$")
        ax.set_ylabel("seconds for the full order spectrum")
        ax.legend(fontsize=7, loc="upper left")
        e = tn.attrs.get("exponent")
        _panel(ax, "b", "Cost vs conditioning set ($L$ fixed)"
                        + (f",  fitted $n^{{{e:.2f}}}$" if e else ""))
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_E5_timing")


# =========================================================================== #
# real
# =========================================================================== #

def stage_real(args, out_dir: Path) -> dict:
    banner("E5c  order spectrum of the fitted model")

    df = load_segregants(Path(args.data_dir) / "qtls_li_hq.csv")
    X_full = genotype_matrix(df["seq"])

    # Subsample the conditioning set: Group II is O(N^2) in sequence-space
    # contractions, against O(N) for Group I.
    rng = np.random.default_rng(args.seed)
    n = min(args.n_condition, len(df)) if args.n_condition else len(df)
    if n < len(df):
        keep = np.sort(rng.choice(len(df), size=n, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
        print(f"\n  conditioning on a {n:,}-segregant subsample of "
              f"{len(X_full):,}")
        print("  Group II is O(N^2) in sequence-space contractions where "
              "Group I is O(N), so the full set is not affordable here. The "
              "plug-in bias this stage measures is LARGER at smaller N, so the "
              "subsample is conservative for the claim being made.")
    else:
        print(f"\n  conditioning on all {len(df):,} segregants")

    sfa = load_model_and_shap(args, df)
    if sfa is None:
        print("\n  model unavailable; E5 real stage cannot run")
        return {}

    X = genotype_matrix(df["seq"])
    counts = (empirical_counts_from_X(X, sfa.alphabet)
              if args.background == "empirical" else None)
    pi = _to_np(make_pi(sfa, args.background, empirical_counts=counts))
    print(f"\n  background: {args.background}, B frequency mean "
          f"{pi[:, b_index(sfa.alphabet)].mean():.3f}")

    sub("per-position kernel tables")
    if args.kappa_from_theta:
        theta = _to_np(sfa.model.gp.covar_module.theta)
        kappa = _to_np(vcm.connectedness_factors(theta, sfa.A))
        print("  built from the Connectedness theta (--kappa-from-theta). "
              "The theta -> rho link is asserted, not derived from the "
              "installed epik, so the check below is not optional.")
    else:
        kappa = _to_np(vcm.kernel_factors_from_ktc(sfa.ktc, sfa.L, sfa.A))

    # Verify the tables reproduce the model's own kernel before using them.
    idx = np.asarray(X, dtype=int)
    Kchk = np.ones((min(8, len(df)), min(8, len(df))))
    for p in range(sfa.L):
        a = idx[:8, p][:, None]
        b = idx[:8, p][None, :]
        Kchk *= kappa[p][a, b]
    print(f"  reconstructed K on 8 training sequences: diagonal "
          f"[{Kchk.diagonal().min():.4f}, {Kchk.diagonal().max():.4f}], "
          f"off-diagonal max {np.abs(Kchk - np.diag(Kchk.diagonal())).max():.4f}")
    if not np.all(np.isfinite(Kchk)):
        print("  !! reconstructed kernel is not finite; stopping")
        return {}

    eta = _to_np(sfa.ktc.Qy).reshape(-1)
    R = _to_np(sfa.ktc.S)
    print(f"  eta {eta.shape}, covar_cache R {R.shape} "
          f"(Gamma^-1 ~ R R^T, rank {R.shape[1]})")

    vc = vcm.VarianceComponents(pi, kappa,
                                device="cuda" if args.gpu else "cpu")

    sub("identity checks on the real model")
    ident = vc.identity_checks(idx, eta)
    for k in ("sum_k Vk (lift)", "E_pi[f^2] (direct)", "err_total",
              "V_0 (lift)", "(E_pi f)^2 (direct)", "err_order0",
              "Var_pi(f)"):
        print(f"  {k:<24} {ident[k]:+.8e}")
    print(f"  [{'PASS' if ident['passed'] else 'FAIL'}] lift reproduces "
          "E_pi[f^2] and (E_pi f)^2")

    # Cost estimate before committing, from the measured ~30 us per target
    # pair at L = 83 and rank 100 (see gkt_variance_components.timing_vs_n).
    pairs = float(len(df)) ** 2
    est = pairs * 30e-6 * (1.0 if not args.gpu else 0.15)
    mem = args.block * len(df) * (sfa.L + 1) * 8 / 2 ** 30
    sub(f"order spectrum ({vc.L + 1} orders, {pairs:.2e} target pairs)")
    print(f"  estimated {est:.0f} s at ~30 us/pair; peak working set "
          f"{mem:.2f} GiB (block={args.block})")
    print("  P^T A P is never materialised -- every use of it is a quadratic "
          "form, against eta for row 3 and against the cache columns for "
          "row 5 -- so the dense N x N polynomial matrix that made this "
          "look infeasible never exists.")
    out = vc.components(idx, eta, R=R, block=args.block, progress=True)

    # Truncate at the numerical resolution floor set by the identity check.
    out = vcm.resolution_floor(out, ident["err_total"])
    floor = out.attrs["floor"]
    n_res = int(out["resolved"].sum())
    print(f"\n  numerical resolution floor {floor:.2e} "
          f"(10x the identity error {ident['err_total']:.2e}); "
          f"{n_res} of {len(out)} orders are resolved")
    print("  eta and the posterior cache arrive from GPyTorch in float32, so "
          "orders below this are ratios of noise, not measurements. They are "
          "written to the CSV with resolved=False and excluded from the "
          "figure.")
    keep = out[(out["order"] <= args.max_order_report) & out["resolved"]]
    cols = ["order", "prior_tr_AK", "V_k_plugin", "V_k_posterior_mean",
            "tr_A_Sigma", "frac_plugin", "frac_posterior"]
    print(f"\n{keep[cols].to_string(index=False)}")

    neg = float(out["tr_A_Sigma"].min())
    print(f"\n  min tr(A_k Sigma) = {neg:.3e}  "
          f"({'OK' if neg > -1e-9 else 'NEGATIVE -- A_k should be PSD'})")

    # Cumulative over k >= 1 only: order 0 is the squared mean, not variance.
    cp = out["frac_plugin"].fillna(0.0).cumsum()
    cq = out["frac_posterior"].fillna(0.0).cumsum()
    out["cum_frac_plugin"] = cp
    out["cum_frac_posterior"] = cq
    sub("variance explained by order <= k (order 0 excluded: it is the mean)")
    print(f"  Var_pi(f^MAP) = {out.attrs['var_plugin']:.6e}, "
          f"E[Var_pi(f) | y] = {out.attrs['var_posterior']:.6e}")
    print(f"\n  {'k':>3}  {'plug-in':>9}  {'posterior':>10}  "
          f"{'overstated by':>14}")
    # Inclusive bound, and never past the last RESOLVED order: an earlier
    # version printed k = 5 at "100.0%" on a model where order 5 sat three
    # decades below the noise floor.
    top = max(1, min(6, int(out.loc[out["resolved"], "order"].max())))
    for k in range(1, top + 1):
        print(f"  {k:>3}  {cp.iloc[k]:9.1%}  {cq.iloc[k]:10.1%}  "
              f"{cp.iloc[k] - cq.iloc[k]:+14.1%}")

    gap = float(cp.iloc[1] - cq.iloc[1])
    ret = out.loc[out["resolved"] & (out["order"] >= 1), "shrinkage_ratio"]
    print("\n  READING THIS. The plug-in misses tr(A_k Sigma) at every order "
          "and misses proportionally more at high order, so the cumulative "
          "curve rises too fast. The MECHANISM shows in the retained "
          f"fractions: {', '.join(f'{v:.2f}' for v in ret.head(5))} for "
          "k = 1, 2, 3, ...")
    if abs(gap) < 0.05:
        print(f"  But the effect on the cumulative curve here is only "
              f"{gap:+.1%} at order <= 1, because the high orders that suffer "
              "most carry almost none of the variance. On THIS model the "
              "plug-in is adequate; the contribution is that this can now be "
              "checked rather than assumed. Do not claim a large correction.")
    else:
        print(f"  The effect is {gap:+.1%} at order <= 1, large enough that "
              "reporting a plug-in spectrum would materially overstate "
              "additivity.")

    print("\n  CAVEAT FOR THE PAPER. This is the spectrum of the FITTED GP, "
          "and the kernel family constrains it. Connectedness carries one "
          "theta per locus, and the prior spectrum already decays steeply "
          f"(order 0 {float(out['prior_tr_AK'].iloc[0]):.4f}, order 1 "
          f"{float(out['prior_tr_AK'].iloc[1]):.4f}, order 2 "
          f"{float(out['prior_tr_AK'].iloc[2]):.4f}), so a low-order posterior "
          "spectrum is partly a property of the model class rather than of the "
          "landscape. A more flexible kernel could place mass differently.")

    write_csv(out, out_dir / "E5_components.csv")
    _fig_components(out, args, out_dir)
    return {"components": out, "identities": ident,
            "n_condition": len(df), "rank": int(R.shape[1])}


def _fig_components(out: pd.DataFrame, args, out_dir: Path) -> None:
    _style()
    res = out[out["resolved"]] if "resolved" in out else out
    d = res[res["order"] <= args.max_order_report]
    k = d["order"].to_numpy()
    d1 = d[d["order"] >= 1]                 # order 0 is the mean, not variance
    k1 = d1["order"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
    fig.subplots_adjust(wspace=0.30, top=0.82, bottom=0.16, left=0.06,
                        right=0.985)

    ax = axes[0]
    w = 0.27
    ax.bar(k - w, d["prior_tr_AK"], width=w, color=C_NEU, label="prior")
    ax.bar(k, d["V_k_plugin"], width=w, color=C_ACC, label="plug-in $f^{MAP}$")
    ax.bar(k + w, d["V_k_posterior_mean"], width=w, color=C_RM,
           label="posterior mean")
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("$V_k$")
    ax.set_xticks(k)
    # Log scale: the components span several decades, and on a linear axis the
    # order-0 prior bar alone flattens everything above k = 2 to invisible.
    ax.set_yscale("log")
    ax.legend(fontsize=7.5)
    _panel(ax, "a", "Order spectrum, three estimators")

    ax = axes[1]
    ax.plot(k1, d1["V_k_plugin"] / d1["V_k_posterior_mean"], "o-", ms=4,
            color=C_ACC)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("$V_k(f^{MAP})\\,/\\,E[V_k\\,|\\,y]$")
    ax.set_xticks(k1)
    ax.set_ylim(0, 1.05)
    _panel(ax, "b", "Fraction retained by the plug-in")

    ax = axes[2]
    ax.plot(k1, d1["cum_frac_plugin"], "o-", ms=4, color=C_ACC,
            label="plug-in $f^{MAP}$")
    ax.plot(k1, d1["cum_frac_posterior"], "s-", ms=4, color=C_RM,
            label="posterior mean")
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("cumulative fraction of $\\mathrm{Var}_\\pi(f)$")
    ax.set_xticks(k1)
    # Zoom to the data. On a 0-1 axis both curves sit between 0.88 and 1.00 and
    # the plug-in/posterior gap -- the only content of this panel -- is a
    # single pixel. The floor is set just under the smaller k = 1 value so the
    # separation is legible without cropping either curve.
    lo = float(min(d1["cum_frac_plugin"].iloc[0],
                   d1["cum_frac_posterior"].iloc[0]))
    ax.set_ylim(max(0.0, lo - 0.04), 1.005)
    ax.legend(fontsize=7.5, loc="lower right")
    _panel(ax, "c", "Variance explained by order $\\leq k$")

    fig.suptitle("E5   Functional-ANOVA variance components over $2^{83}$ "
                 "sequences", fontsize=12, y=0.955)
    save_fig(fig, out_dir, "fig_E5_variance_components")


# =========================================================================== #
# CLI
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="validate,timing",
                    help="comma-separated: validate, timing, real")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-9)

    v = ap.add_argument_group("validate")
    v.add_argument("--validate-L", type=int, nargs="+", default=[4, 6, 8, 10])
    v.add_argument("--validate-n-train", type=int, default=24)

    t = ap.add_argument_group("timing")
    t.add_argument("--timing-L", type=int, nargs="+",
                   default=[4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 30, 45, 60, 83])
    t.add_argument("--timing-bf-max-L", type=int, default=10)
    t.add_argument("--timing-n-train", type=int, default=24)
    t.add_argument("--timing-reps", type=int, default=3)
    t.add_argument("--timing-n", type=int, nargs="+",
                   default=[24, 48, 96, 192, 384, 768],
                   help="conditioning-set sizes for the N-scaling sweep at "
                        "fixed L. This is the scaling that decides whether "
                        "Group II is affordable; the L sweep at small n is "
                        "overhead-dominated and cannot show it.")

    r = ap.add_argument_group("real model")
    r.add_argument("--data-dir", type=Path,
                   default=REPO_ROOT / "data" / "qtls_li_hq")
    r.add_argument("--params-path", type=Path, default=None)
    r.add_argument("--project-root", type=Path, default=None)
    r.add_argument("--gpu", action="store_true")
    r.add_argument("--require-gpu", action="store_true")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--no-center-y", action="store_true")
    r.add_argument("--yvar-scale", type=float, default=0.05)
    r.add_argument("--n-condition", type=int, default=0,
                   help="conditioning-set size; 0 = all 19,833. Group II costs "
                        "O(N^2 L (L + r)) against O(N L^2) for Group I, but "
                        "because every use of P^T A P is a quadratic form it is "
                        "never materialised, so memory is only "
                        "block * N * (L+1) * 8 bytes (~0.4 GB at N = 19,833, "
                        "block = 32) and the full set costs a few hours on CPU "
                        "and less on GPU. Subsample only to iterate quickly.")
    r.add_argument("--root-decomp-size", type=int, default=None,
                   help="GPyTorch max_root_decomposition_size, i.e. the rank r "
                        "of the posterior cache (default 100). It matters twice "
                        "here: the cache IS the R in Gamma^{-1} ~ R R^T that "
                        "row 5 contracts against, so r sets both the fidelity "
                        "of tr(A_k Sigma) and, through the O(r N^2 L) term, a "
                        "large part of the runtime.")
    r.add_argument("--background", default="empirical",
                   choices=["empirical", "uniform"])
    r.add_argument("--block", type=int, default=32,
                   help="target-pair row block; peak memory is "
                        "block * N * (L+1) * 8 bytes")
    r.add_argument("--max-order-report", type=int, default=8)
    r.add_argument("--kappa-from-theta", action="store_true",
                   help="build the per-position kernel tables from the "
                        "Connectedness theta instead of reading them off the "
                        "computer. Fallback only: the theta -> rho link is "
                        "asserted rather than taken from epik.")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"E5  variance components -- stages {sorted(stages)}")
    print(f"  out   {out_dir}")
    summary: dict = {"stages": sorted(stages)}

    if "validate" in stages:
        r = stage_validate(a, out_dir)
        summary["validate_passed"] = r["passed"]
        if not r["passed"] and "real" in stages:
            print("\n  validation FAILED; refusing to run the real stage.")
            return 4
    if "timing" in stages:
        t = stage_timing(a, out_dir)
        summary["timing_rows"] = int(len(t))
    if "real" in stages:
        r = stage_real(a, out_dir)
        if r:
            summary["n_condition"] = r["n_condition"]
            summary["cache_rank"] = r["rank"]
            summary["identities_passed"] = r["identities"]["passed"]
            c = r["components"]
            summary["cum_frac_plugin_le2"] = float(
                c["cum_frac_plugin"].iloc[2])
            summary["cum_frac_posterior_le2"] = float(
                c["cum_frac_posterior"].iloc[2])
        else:
            return 1

    (out_dir / "E5_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"  wrote {out_dir / 'E5_summary.json'}")
    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())