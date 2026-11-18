"""
E5_variance_components.py
=========================

Experiment E5: exact functional-ANOVA variance components of the GP over the
whole 2^83 sequence space, via the Group-II (quadratic) rows of the paper's
statistics table.

("Functional ANOVA" = the Hoeffding/Sobol decomposition of f into additive,
pairwise, three-way, ... parts that are orthogonal under the background pi, so
Var_pi(f) = sum_k V_k exactly, one component per interaction order.)

Stages, selected with --stages:

  validate  GKT against explicit enumeration of all alpha^L sequences, both
            trace routes, both variance routes, numpy AND (if importable) the
            torch backend that the real stage uses on the GPU.
  timing    Wall-clock for the order spectrum, GKT against enumeration.
  real      The fitted yeast model. Rows 3-5 at --n-condition through the
            GPyTorch cache, and -- with --component-variance-n -- an EXACT
            float64 sub-problem on which means AND variances are computed
            together. The figure uses the exact sub-problem when present, so
            every panel describes one posterior at one N.
  replot    Redraw the figure from E5_figure_table.csv / E5_figure_meta.json
            in --out-dir. No model, no GPU, seconds. Use it to iterate on the
            figure.

WHAT THE FIGURE SAYS
--------------------
  (a) V_k per order as bars: prior mean, plug-in V_k(f^MAP), posterior mean,
      with exact 95% intervals from rows 6-7 (prior interval from
      tr(A_k K A_k K), posterior from Var[V_k | y]). Where the data are
      uninformative the posterior bar equals the prior bar; the plug-in bar
      collapses. That contrast is the shrinkage-toward-additivity result.
  (b) What the data constrain: 1 - tr(A_k Sigma)/tr(A_k K), the share of the
      prior's order-k variance pinned down by the data. Exact, in [0, 1].
  (c) Where the plug-in sits in the posterior: V_k(f^MAP)/E[V_k|y] as points
      against the posterior 95% range of V_k/E[V_k|y] as grey boxes.
  (d) Fraction of Var_pi(f) above order k (reverse cumulative sum; rev2's
      1 - cumsum plateaued at float64 epsilon).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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

try:
    import gkt_variance_components as vcm                      # noqa: E402
except ModuleNotFoundError as _exc:                            # noqa: BLE001
    print(f"\n!! {_exc}")
    print("\ngkt_variance_components.py was not found. It must live in one of:")
    for _d in (_HERE, _ROOT / "modules", _ROOT):
        _hits = sorted(Path(_d).glob("gkt_variance_components.py")) \
            if Path(_d).exists() else []
        print(f"    {_d}{'   <- FOUND' if _hits else ''}")
    sys.exit(2)
from ENA1_paper_analysis import (                              # noqa: E402
    REPO_ROOT, C_RM, C_NEU, C_ACC,
    banner, sub, write_csv, save_fig, _style, _panel, _to_np,
    load_segregants, genotype_matrix, empirical_counts_from_X,
    load_model_and_shap, make_pi, b_index,
)

SCRIPT_REVISION = "e5-rev6c-2026-10-02"
REQUIRED_MODULE_REVISION = "gkt-vc-rev6-2026-10-02"
REVISION_NOTES = (
    "rev6b: panels a, b, d log-y; spectrum as points+lines with V_{k>K}; "
    "panel c variance explained (linear); title states what N is. rev5: full-data capable -- truncated variance lift (orders <= "
    "--variance-max-order), chunked Frobenius term, Cholesky factorisation; "
    "2x2 figure with posterior whiskers and variance-explained intervals"
)
C_INF = "#7B3294"
DEFAULT_OUT = REPO_ROOT / "output" / "E5"
_EPS = float(np.finfo(np.float64).eps)


class _Tee:
    """Duplicate a stream to a file, flushed on every write."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = open(path, "w", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def close(self):
        try:
            self.file.close()
        except Exception:                                      # noqa: BLE001
            pass


# =========================================================================== #
# validate
# =========================================================================== #

def _print_validation(r):
    for c, e in r["errors"].items():
        print(f"  {c:<26} max |GKT - enumeration| = {e:.3e}   "
              f"(scale {r['scales'][c]:.3e})")
    i = r["identities"]
    print(f"  {'sum_k V_k vs E_pi[f^2]':<26} {i['err_total']:.3e}")
    print(f"  {'V_0 vs (E_pi f)^2':<26} {i['err_order0']:.3e}")
    print(f"  {'min tr(A_k Sigma)':<26} {r['min_tr_A_Sigma']:.3e}   "
          "(must be >= 0; A_k is PSD)")
    print(f"  {'error-estimate coverage':<26} "
          f"{r['error_estimate_coverage']:.2f}   (share of orders where the "
          "per-order error estimate covers the observed disagreement)")
    print(f"  [{'PASS' if r['passed'] else 'FAIL'}]")


def stage_validate(args, out_dir: Path) -> dict:
    banner("E5a  validation against explicit enumeration")
    rows, tables = [], {}
    for L in args.validate_L:
        r = vcm.validate(L=int(L), n_train=args.validate_n_train, seed=int(L),
                         tol=args.tol)
        sub(f"L = {L}  (alpha^L = {2 ** int(L):,} sequences)  [{r['backend']}]")
        _print_validation(r)
        rec = {"L": int(L), "backend": r["backend"], "passed": r["passed"],
               "min_tr_A_Sigma": r["min_tr_A_Sigma"],
               "err_identity_total": r["identities"]["err_total"],
               "err_identity_order0": r["identities"]["err_order0"],
               "error_estimate_coverage": r["error_estimate_coverage"]}
        rec.update({f"err_{c}": v for c, v in r["errors"].items()})
        rows.append(rec)
        tables[int(L)] = r

    # The GPU run uses the torch backend. Check it here against brute force,
    # on the device the real stage will use, before any real result exists.
    if vcm._have_torch():
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Lt = int(min(8, max(args.validate_L)))
        r = vcm.validate(L=Lt, n_train=args.validate_n_train, seed=Lt,
                         tol=args.tol, backend="torch", device=dev)
        sub(f"L = {Lt}  torch backend on {dev} (the code path rows 3-7 use "
            "with --gpu)")
        _print_validation(r)
        rec = {"L": Lt, "backend": r["backend"], "passed": r["passed"],
               "min_tr_A_Sigma": r["min_tr_A_Sigma"],
               "err_identity_total": r["identities"]["err_total"],
               "err_identity_order0": r["identities"]["err_order0"],
               "error_estimate_coverage": r["error_estimate_coverage"]}
        rec.update({f"err_{c}": v for c, v in r["errors"].items()})
        rows.append(rec)
    else:
        print("\n  torch not importable: torch backend NOT validated")

    tbl = pd.DataFrame(rows)
    write_csv(tbl, out_dir / "E5_validation.csv")
    Lmax = max(tables)
    cmp = tables[Lmax]["gkt"].merge(
        tables[Lmax]["brute_force"], on="order", suffixes=("_gkt", "_bf"))
    write_csv(cmp, out_dir / "E5_validation_spectrum.csv")
    ok = bool(tbl["passed"].all())
    print(f"\n  [{'PASS' if ok else 'FAIL'}] all tested L and backends")
    return {"table": tbl, "passed": ok, "spectrum": cmp}


# =========================================================================== #
# timing
# =========================================================================== #

def enumeration_cost_model(t: pd.DataFrame) -> tuple[float, float, int]:
    """Fit enumeration cost as unit * base^L from the largest measured L."""
    b = t.dropna(subset=["seconds_brute_force"]).sort_values("L")
    if len(b) < 2:
        return float("nan"), 8.0, 0
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
          f"largest measured L ({L_ref}). Expected base 2*alpha^2 = 8.")
    if not (0.7 * 8.0 <= base <= 1.3 * 8.0):
        print(f"  !! fitted base {base:.2f} is far from 8; treat the "
              "projection as indicative only.")
    for L in [L for L in args.timing_L if L > L_ref]:
        s = unit * base ** L
        print(f"    L={L:>3}  projected {s:.3e} s = {s / 3.156e7:.3e} years")
    g83 = t[t["L"] == max(args.timing_L)]
    if len(g83):
        sp = unit * base ** max(args.timing_L) / float(g83["seconds_gkt"].iloc[0])
        print(f"  speed-up at L = {max(args.timing_L)}: {sp:.2e}x")
    t["projected_brute_force"] = unit * base ** t["L"].to_numpy(dtype=float)
    write_csv(t, out_dir / "E5_timing.csv")

    sub(f"cost in the conditioning-set size, at L = {max(args.timing_L)}")
    tn = vcm.timing_vs_n(args.timing_n, L=max(args.timing_L),
                         reps=max(1, args.timing_reps // 2),
                         block=args.block)
    write_csv(tn, out_dir / "E5_timing_vs_n.csv")
    _fig_timing(t, unit, base, L_ref, out_dir, tn=tn)
    return t


def _fig_timing(t, unit, base, L_ref, out_dir, tn=None) -> None:
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
    ax.legend(fontsize=7, loc="upper left")
    _panel(ax, "a" if len(axes) > 1 else "",
           "Cost vs sequence length ($n$ fixed)")
    if len(axes) > 1:
        ax = axes[1]
        n = tn["n_train"].to_numpy(dtype=float)
        ax.plot(n, tn["seconds"], "o-", ms=4, color=C_ACC, lw=1.6,
                label="GKT (measured)")
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

def _model_extra_noise(sfa):
    """The learned second noise term, so the exact rebuild matches the model."""
    chains = ("model.likelihood.second_noise_covar.noise",
              "likelihood.second_noise_covar.noise",
              "model.gp.likelihood.second_noise_covar.noise",
              "gp.likelihood.second_noise_covar.noise",
              "model.likelihood.second_noise",
              "likelihood.second_noise")
    for ch in chains:
        obj = sfa
        try:
            for a in ch.split("."):
                obj = getattr(obj, a)
            val = float(np.asarray(_to_np(obj), dtype=float).reshape(-1)[0])
            return val, ch
        except Exception:                                      # noqa: BLE001
            continue
    return 0.0, None


def _print_spectrum(out: pd.DataFrame, title: str, max_rows: int = 8):
    """Resolved orders only, per series; then the cumulative comparison."""
    sub(title)
    rp = out["resolved_plugin"].to_numpy(bool)
    rq = out["resolved_posterior"].to_numpy(bool)
    print(f"  resolved orders (value > {out.attrs.get('safety', 10):.0f} x its "
          f"own error estimate): plug-in 0-{int(out['order'][rp].max())}, "
          f"posterior 0-{int(out['order'][rq].max())}")
    shown = out[out["order"] <= max_rows]
    print(f"\n  {'k':>3} {'prior':>11} {'plug-in':>11} {'posterior':>11} "
          f"{'tr(A Sig)':>11} {'informed':>9}")
    for _, r in shown.iterrows():
        pl = (f"{r['V_k_plugin']:11.3e}" if r["resolved_plugin"]
              else f"{'< prec.':>11}")
        po = (f"{r['V_k_posterior_mean']:11.3e}" if r["resolved_posterior"]
              else f"{'< prec.':>11}")
        inf = (f"{100 * r['data_informed_frac']:8.3g}%" if r["resolved_informed"]
               else f"{'~0':>9}")
        print(f"  {int(r['order']):>3} {r['prior_tr_AK']:11.3e} {pl} {po} "
              f"{r['tr_A_Sigma']:11.3e} {inf}")

    cp = out["frac_plugin"].fillna(0.0).cumsum()
    cq = out["frac_posterior"].fillna(0.0).cumsum()
    out["cum_frac_plugin"] = cp
    out["cum_frac_posterior"] = cq
    print(f"\n  Var_pi(f^MAP) = {out.attrs['var_plugin']:.6e}, "
          f"E[Var_pi(f) | y] = {out.attrs['var_posterior']:.6e}")
    print(f"  {'k':>3}  {'plug-in <=k':>12}  {'posterior <=k':>13}  "
          f"{'difference':>11}")
    for k in range(1, 5):
        print(f"  {k:>3}  {cp.iloc[k]:12.2%}  {cq.iloc[k]:13.2%}  "
              f"{cp.iloc[k] - cq.iloc[k]:+11.2%}")
    return out


def _build_exact(nv, args, df, idx, kappa, L, extra_noise):
    """An exact float64 GP sub-problem on nv of the conditioning sequences.

    One eigendecomposition of Gamma gives Gamma^-1, its symmetric root and the
    condition number at once. Means and variances are then both EXACT for a GP
    conditioned on these nv points, so every panel refers to one posterior.
    """
    n = len(df)
    if nv >= n:
        sel = np.arange(n)
    else:
        sel = np.sort(np.random.default_rng(args.seed).choice(
            n, size=nv, replace=False))
    Xi = idx[sel]
    K = np.ones((len(sel), len(sel)))
    for p in range(L):
        K *= kappa[p][Xi[:, p][:, None], Xi[:, p][None, :]]
    y = (df["y"].to_numpy(dtype=float)[sel]
         - (0.0 if args.no_center_y else float(df["y"].mean())))
    noise = np.maximum(df["y_var"].to_numpy(dtype=float)[sel] * args.yvar_scale,
                       args.variance_noise_floor) + extra_noise
    t0 = time.perf_counter()
    fac = vcm.exact_gp_factors(K + np.diag(noise), y)
    del K
    print(f"  exact factorisation of the {len(sel):,} x {len(sel):,} system: "
          f"{time.perf_counter() - t0:.1f} s")
    Gi, Ri, eta = fac["Gi"], fac["R"], fac["eta"]
    return {"sel": sel, "X": Xi, "Gi": Gi, "R": Ri, "eta": eta,
            "cond": fac["cond"], "N": len(sel)}


def _exact_rows(args, df, idx, kappa, vc, sfa, eta_cache, out_cache):
    """Rows 3-7 on one exact sub-problem, inside a wall-clock budget."""
    nv = int(min(args.component_variance_n, len(df)))
    extra, where = _model_extra_noise(sfa)
    sub(f"exact sub-problem (requested N = {nv:,})")
    print(f"  learned extra noise {extra:.3e} "
          f"({'from sfa.' + where if where else 'NOT FOUND -- using 0'})")

    budget = float(args.variance_budget_hours) * 3600.0
    for attempt in range(4):
        sp = _build_exact(nv, args, df, idx, kappa, vc.L, extra)
        est = vc.estimate_variance_seconds(
            sp["X"], r=sp["N"], block_biv=args.block_bivariate,
            block_uni=args.block, max_order=args.variance_max_order,
            frob_chunk_bytes=args.frob_chunk_gb * 2 ** 30)
        tot = est["bivariate_s"] + est["univariate_s"]
        print(f"  N = {nv:,}: projected variance-pass time {tot / 60:.1f} min "
              f"(bivariate {est['bivariate_s'] / 60:.1f}, frobenius "
              f"{est['univariate_s'] / 60:.1f}) on {est['backend']}; "
              f"budget {budget / 60:.0f} min")
        for n2 in (1000, 1500, 2000, 3000, 4000):
            pj = (est["bivariate_s"] * (n2 / nv) ** 2
                  + est["univariate_s"] * (n2 / nv) ** 3)
            print(f"      projection at N = {n2:>5,}: {pj / 3600:6.2f} h")
        if tot <= budget or nv <= 200:
            break
        lo_n, hi_n = 100, nv
        while hi_n - lo_n > 50:
            mid = (lo_n + hi_n) // 2
            pj = (est["bivariate_s"] * (mid / nv) ** 2
                  + est["univariate_s"] * (mid / nv) ** 3)
            lo_n, hi_n = (mid, hi_n) if pj <= 0.9 * budget else (lo_n, mid)
        new = max(200, (lo_n // 100) * 100)
        print(f"  !! over budget: reducing N {nv:,} -> {new:,}")
        nv = new

    print(f"\n  exact float64 solve: cond(Gamma) = {sp['cond']:.2e}, so eta "
          f"and Gamma^-1 carry relative error ~ {sp['cond'] * _EPS:.1e}; that "
          "is the input error used in the per-order resolution.")
    if nv == len(df):
        rel = float(np.linalg.norm(sp["eta"] - eta_cache)
                    / np.linalg.norm(sp["eta"]))
        print(f"  cross-check against the GPyTorch dual weights on the SAME "
              f"{nv:,} points: ||eta_exact - eta_gpytorch|| / ||eta_exact|| "
              f"= {rel:.2e}")
        if rel > 1e-2:
            print("  !! large disagreement: the rebuild's noise model does not "
                  "match the fitted likelihood, or GPyTorch solved with a loose "
                  "CG tolerance. The exact sub-problem is the one to trust for "
                  "the figure; check --variance-noise-floor / --yvar-scale.")

    ex = vc.components(sp["X"], sp["eta"], Gamma_inv=sp["Gi"],
                       block=args.block, error_estimate=True,
                       input_rel_err=max(sp["cond"] * _EPS, _EPS))
    ex = vcm.resolution_per_order(ex, safety=args.resolution_safety)
    ident = vc.identity_checks(sp["X"], sp["eta"])
    print(f"  identity check on the exact sub-problem: err_total "
          f"{ident['err_total']:.2e}  [{'PASS' if ident['passed'] else 'FAIL'}]")
    ex = _print_spectrum(ex, f"posterior means, exact sub-problem, N = {nv:,}")
    if nv == len(df):
        print("\n  GPyTorch-cache route vs exact, tr(A_k Sigma):")
        for k in range(1, 5):
            a, b = out_cache["tr_A_Sigma"].iloc[k], ex["tr_A_Sigma"].iloc[k]
            print(f"    k={k}: cache {a:.4e}   exact {b:.4e}   "
                  f"ratio {a / b:.3f}")

    sub(f"posterior variance and covariance of the components, N = {nv:,}")
    vt = vc.component_variances(sp["X"], sp["eta"], R=sp["R"],
                                Gamma_inv=sp["Gi"],
                                block_biv=args.block_bivariate,
                                block_uni=args.block, progress=True,
                                max_order=args.variance_max_order,
                                frob_chunk_bytes=args.frob_chunk_gb * 2 ** 30)
    print(f"  variances and covariances computed for orders 0-"
          f"{vt.attrs['max_order']} (truncated lift, exact for those orders); "
          f"Frobenius term in {vt.attrs['frob_chunks']} chunk(s)")
    cov = np.array(vt.attrs["cov_theta"])
    tb, tu = vt.attrs["seconds_bivariate"], vt.attrs["seconds_univariate"]
    print(f"  done: bivariate {tb:.1f} s, frobenius {tu:.1f} s "
          f"on {vt.attrs['backend']}")
    for n2 in (1000, 1500, 2000, 3000):
        pj = tb * (n2 / nv) ** 2 + tu * (n2 / nv) ** 3
        print(f"      measured-rate projection at N = {n2:>5,}: "
              f"{pj / 3600:6.2f} h")
    neg = vt["var_theta_k"] < 0
    if neg.any():
        print(f"  !! {int(neg.sum())} orders with negative variance "
              f"(min {float(vt['var_theta_k'].min()):+.2e}); set to NaN")
        vt.loc[neg, ["var_theta_k", "sd_theta_k"]] = np.nan

    m = ex.merge(vt, on="order")
    print(f"\n  {'k':>3}{'E[V_k|y]':>12}{'sd[V_k|y]':>12}{'prior sd':>12}"
          f"{'plug-in z':>11}")
    for _, r in m[(m["order"] >= 1) & m["resolved_posterior"]].head(8).iterrows():
        z = ((r["V_k_posterior_mean"] - r["V_k_plugin"]) / r["sd_theta_k"]
             if r["sd_theta_k"] > 0 else np.nan)
        print(f"  {int(r['order']):>3}{r['V_k_posterior_mean']:12.3e}"
              f"{r['sd_theta_k']:12.3e}{r['prior_sd_theta_k']:12.3e}"
              f"{z:11.2f}")
    print("  plug-in z = (E[V_k|y] - V_k(f^MAP)) / sd[V_k|y]: how many "
          "posterior s.d. the plug-in sits below the posterior mean.")

    ve = vcm.variance_explained(ex["V_k_posterior_mean"].to_numpy(), cov,
                                args.interval_level)
    pl = np.clip(ex["V_k_plugin"].to_numpy()[1:], 0, None)
    cpl = np.cumsum(pl) / pl.sum()
    lvl = int(round(100 * args.interval_level))
    print(f"\n  variance explained by order <= k, with the {lvl}% posterior "
          "range from the full cross-order covariance:")
    print(f"  (posterior share of Var_pi(f) above order "
          f"{vt.attrs['max_order']}, whose covariance is omitted: "
          f"{ve['tail_share_above_K']:.1e})")
    print(f"  {'k':>3}{'plug-in':>10}{'posterior':>11}{'range':>22}"
          f"{'plug-in inside?':>17}")
    for i in range(min(5, len(ve["frac"]))):
        inside = ve["lo"][i] <= cpl[i] <= ve["hi"][i]
        print(f"  {i + 1:>3}{cpl[i]:10.2%}{ve['frac'][i]:11.2%}"
              f"   [{ve['lo'][i]:7.2%}, {ve['hi'][i]:7.2%}]"
              f"{'yes' if inside else 'NO':>17}")
    return ex, vt, nv, cov


def _add_variance_parts(t, var, level):
    """Posterior s.d. and its two parts, keyed to the paper's Table 1.

        row 5  delta-method variance     4 mu^T A_k Sigma A_k mu
        row 6  curvature term            2 tr(A_k Sigma A_k Sigma)
        exact  Var[V_k | y]            = row 5 + row 6

    Also the plug-in's share of the TRUE V_k with its posterior interval.
    V_k is random under the posterior and the plug-in is fixed, so
    r = V_k(f^MAP) / V_k is random too, and because r is monotone decreasing
    in V_k its quantiles are exact images of V_k's:
        q_lo(r) = plugin / q_hi(V_k),   q_hi(r) = plugin / q_lo(V_k).
    That is the interval that belongs on the plug-in point in panel (c).
    """
    t = t.copy()
    if var is not None:
        v = var.set_index("order").reindex(t["order"])
        t["post_sd"] = v["sd_theta_k"].to_numpy()
        dm = v["mu_A_Sigma_A_mu"].to_numpy(dtype=float)
        cv = v["tr_A_Sigma_A_Sigma"].to_numpy(dtype=float)
        t["delta_var"] = 4.0 * dm
        t["curv_var"] = 2.0 * cv
        t["delta_sd"] = np.sqrt(np.clip(4.0 * dm, 0.0, None))
        t["curv_sd"] = np.sqrt(np.clip(2.0 * cv, 0.0, None))
        t["delta_valid"] = dm > 0
    else:
        for c in ("post_sd", "delta_var", "curv_var", "delta_sd", "curv_sd"):
            t[c] = np.nan
        t["delta_valid"] = False
    t["post_lo"], t["post_hi"] = vcm.gamma_interval(t["post"], t["post_sd"],
                                                    level)
    with np.errstate(divide="ignore", invalid="ignore"):
        t["share"] = t["plugin"] / t["post"]
        t["share_lo"] = t["plugin"] / t["post_hi"]
        t["share_hi"] = t["plugin"] / t["post_lo"]
    return t


def _add_cumulative(t, cov, level):
    """Variance explained by orders <= k (order 0 excluded), both estimators.
    The posterior interval needs the full cross-order covariance."""
    t = t.copy()
    pl = np.clip(t["plugin"].to_numpy(dtype=float), 0.0, None)
    po = t["post"].to_numpy(dtype=float)
    t["cum_plugin"] = np.r_[np.nan, np.cumsum(pl[1:]) / pl[1:].sum()]
    t["cum_post"] = np.r_[np.nan, np.cumsum(po[1:]) / po[1:].sum()]
    if cov is not None:
        ve = vcm.variance_explained(po, cov, level)
        pad = np.full(len(t) - 1 - len(ve["sd"]), np.nan)
        t["cum_post_sd"] = np.r_[np.nan, ve["sd"], pad]
        t["cum_post_lo"] = np.r_[np.nan, ve["lo"], pad]
        t["cum_post_hi"] = np.r_[np.nan, ve["hi"], pad]
        vu = vcm.variance_unexplained(po, cov, level)
        t["unex_post_lo"] = np.r_[np.nan, vu["lo"]]
        t["unex_post_hi"] = np.r_[np.nan, vu["hi"]]
        t["tail_abs_post"] = np.r_[np.nan, vu["tail_abs"]]
        t["tail_abs_post_sd"] = np.r_[np.nan, vu["tail_abs_sd"]]
    else:
        for c in ("cum_post_sd", "cum_post_lo", "cum_post_hi", "unex_post_lo",
                  "unex_post_hi", "tail_abs_post", "tail_abs_post_sd"):
            t[c] = np.nan
    return t


def _figure_table(means, var, tr_akak, level, safety, cov=None):
    """Everything the figures draw, one row per order."""
    t = pd.DataFrame({"order": means["order"].to_numpy()})
    for s, d in (("prior_tr_AK", "prior"), ("V_k_plugin", "plugin"),
                 ("V_k_posterior_mean", "post"), ("tr_A_Sigma", "tr_A_Sigma"),
                 ("tr_Gi_M", "tr_Gi_M"), ("data_informed_frac", "informed"),
                 ("err_plugin", "err_plugin"),
                 ("err_posterior", "err_posterior"),
                 ("resolved_plugin", "resolved_plugin"),
                 ("resolved_posterior", "resolved_post"),
                 ("resolved_informed", "resolved_informed")):
        t[d] = means[s].to_numpy()
    t["prior_sd"] = np.sqrt(np.clip(2.0 * np.asarray(tr_akak), 0.0, None))
    t["prior_lo"], t["prior_hi"] = vcm.gamma_interval(t["prior"], t["prior_sd"],
                                                      level)
    t = _add_variance_parts(t, var, level)
    t = _add_cumulative(t, cov, level)
    for col, err, name in (("prior", None, "prior"),
                           ("plugin", "err_plugin", "plugin"),
                           ("post", "err_posterior", "post")):
        frac, absv = vcm.spectrum_tail(t[col].to_numpy())
        t[f"tail_{name}"] = np.r_[np.nan, frac]
        if err is None:
            t[f"tail_{name}_resolved"] = np.r_[False, absv > 0]
        else:
            _, eabs = vcm.spectrum_tail(t[err].to_numpy())
            t[f"tail_{name}_resolved"] = np.r_[False, absv > safety * eabs]
    return t


def stage_real(args, out_dir: Path) -> dict:
    banner("E5c  order spectrum of the fitted model")
    df = load_segregants(Path(args.data_dir) / "qtls_li_hq.csv")
    X_full = genotype_matrix(df["seq"])
    rng = np.random.default_rng(args.seed)
    n = min(args.n_condition, len(df)) if args.n_condition else len(df)
    if n < len(df):
        keep = np.sort(rng.choice(len(df), size=n, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
        print(f"\n  conditioning on a {n:,}-segregant subsample of "
              f"{len(X_full):,}")
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
        print("  built from the Connectedness theta (--kappa-from-theta); the "
              "theta -> rho link is asserted, so check the line below.")
    else:
        kappa = _to_np(vcm.kernel_factors_from_ktc(sfa.ktc, sfa.L, sfa.A))
    kappa = np.asarray(kappa, dtype=float)
    idx = np.asarray(X, dtype=int)
    m8 = min(8, len(df))
    Kchk = np.ones((m8, m8))
    for p in range(sfa.L):
        Kchk *= kappa[p][idx[:m8, p][:, None], idx[:m8, p][None, :]]
    print(f"  reconstructed K on {m8} training sequences: diagonal "
          f"[{Kchk.diagonal().min():.4f}, {Kchk.diagonal().max():.4f}], "
          f"off-diagonal max {np.abs(Kchk - np.diag(Kchk.diagonal())).max():.4f}")
    if not np.all(np.isfinite(Kchk)):
        print("  !! reconstructed kernel is not finite; stopping")
        return {}

    eta = _to_np(sfa.ktc.Qy).reshape(-1).astype(float)
    R = _to_np(sfa.ktc.S).astype(float)
    print(f"  eta {eta.shape}, covar_cache R {R.shape} (rank {R.shape[1]})")

    vc = vcm.VarianceComponents(pi, kappa, device="cuda" if args.gpu else "cpu")
    print(f"  array backend: {vc.bk.name}")

    sub("identity checks on the real model")
    ident = vc.identity_checks(idx, eta)
    for k in ("sum_k Vk (lift)", "E_pi[f^2] (direct)", "err_total",
              "V_0 (lift)", "(E_pi f)^2 (direct)", "err_order0", "Var_pi(f)"):
        print(f"  {k:<24} {ident[k]:+.8e}")
    print(f"  [{'PASS' if ident['passed'] else 'FAIL'}]")

    sub(f"posterior means through the GPyTorch cache, N = {len(df):,}")
    out = vc.components(idx, eta, R=R, block=args.block, progress=True,
                        error_estimate=True, input_rel_err=2.0 ** -24)
    out = vcm.resolution_per_order(out, safety=args.resolution_safety)
    out = _print_spectrum(out, f"spectrum, GPyTorch cache (rank "
                               f"{R.shape[1]}), N = {len(df):,}")
    write_csv(out, out_dir / "E5_components.csv")
    print("  NOTE: this route inherits the float32 dual weights and the "
          f"rank-{R.shape[1]} cache. It is kept for the record; the figure "
          "uses the exact sub-problem when --component-variance-n is set.")

    fig_means, fig_var, fig_N, fig_cov = out, None, len(df), None
    source = f"GPyTorch cache, rank {R.shape[1]}"
    if args.component_variance_n:
        ex, vt, nv, fig_cov = _exact_rows(args, df, idx, kappa, vc, sfa,
                                          eta, out)
        write_csv(ex, out_dir / "E5_components_exact.csv")
        write_csv(vt, out_dir / "E5_component_variances.csv")
        kc = fig_cov.shape[0]
        cdf = pd.DataFrame(fig_cov, columns=[f"k{j}" for j in range(kc)])
        cdf.insert(0, "order", np.arange(kc))
        write_csv(cdf, out_dir / "E5_component_covariance.csv")
        fig_means, fig_var, fig_N = ex, vt, nv
        source = "exact float64 posterior"

    tab = _figure_table(fig_means, fig_var, vc.trace_AKAK(),
                        args.interval_level, args.resolution_safety,
                        cov=fig_cov)
    meta = {"N": int(fig_N), "N_condition": int(len(df)), "source": source,
            "level": float(args.interval_level), "L": int(vc.L),
            "safety": float(args.resolution_safety),
            "revision": SCRIPT_REVISION}
    write_csv(tab, out_dir / "E5_figure_table.csv")
    (out_dir / "E5_figure_meta.json").write_text(json.dumps(meta, indent=2))
    _fig_components(tab, meta, args, out_dir)
    _fig_tail(tab, meta, args, out_dir)
    print(f"\n  all {vc.L + 1} orders are in E5_figure_table.csv; the figure "
          f"draws orders <= {args.max_order_report}; fig_E5_tail.png draws "
          f"orders <= {args.max_order_tail}.")
    return {"components": out, "identities": ident, "figure_table": tab,
            "n_condition": len(df), "rank": int(R.shape[1]), "fig_N": fig_N}


def stage_replot(args, out_dir: Path) -> None:
    """Redraw from the CSVs of an earlier run.

    Reads from --replot-from (default: --out-dir) and writes ONLY to
    --out-dir, so pointing --out-dir at a fresh folder leaves the source run's
    figures and tables untouched.
    """
    src = Path(args.replot_from) if args.replot_from else out_dir
    if not src.is_absolute():
        src = REPO_ROOT / src
    banner(f"E5  replot: reading {src}, writing {out_dir}")
    if src.resolve() == out_dir.resolve():
        print("  [note] source and destination are the same folder; the "
              "figures there will be overwritten")
    tab = pd.read_csv(src / "E5_figure_table.csv")
    meta = json.loads((src / "E5_figure_meta.json").read_text())
    level = float(meta.get("level", 0.95))
    for c in [c for c in tab.columns if c.startswith("resolved")
              or c.endswith("_resolved")]:
        tab[c] = tab[c].astype(bool)
    # Rebuild every derived column (intervals, cumulative fractions) from the
    # source CSVs, so a replot always reflects the CURRENT interval code --
    # e.g. a table written by a run that crashed while plotting.
    vpath = src / "E5_component_variances.csv"
    if vpath.exists():
        tab = _add_variance_parts(tab, pd.read_csv(vpath), level)
    elif "delta_sd" not in tab:
        tab = _add_variance_parts(tab, None, level)
    cpath = src / "E5_component_covariance.csv"
    cov = (pd.read_csv(cpath).drop(columns="order").to_numpy()
           if cpath.exists() else None)
    if cov is None:
        print("  !! no E5_component_covariance.csv: panel (c) is drawn "
              "without the posterior range.")
    tab = _add_cumulative(tab, cov, level)
    write_csv(tab, out_dir / "E5_figure_table.csv")
    (out_dir / "E5_figure_meta.json").write_text(json.dumps(
        dict(meta, replotted_from=str(src), replot_revision=SCRIPT_REVISION),
        indent=2))
    _fig_components(tab, meta, args, out_dir)
    _fig_tail(tab, meta, args, out_dir)


# =========================================================================== #
# the figure
# =========================================================================== #

def _log_limits(arrays, pad_lo=0.5, pad_hi=1.2):
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    v = v[np.isfinite(v) & (v > 0)]
    return (10 ** (np.floor(np.log10(v.min())) - pad_lo),
            10 ** (np.ceil(np.log10(v.max())) + pad_hi))


def _grouped_log_bars(ax, k, groups, ybot, w=0.27):
    """groups: list of (offset, values, lo, hi, colour, label, mask)."""
    for off, v, lo, hi, colour, label, mask in groups:
        v = np.asarray(v, dtype=float)
        m = np.asarray(mask, dtype=bool) & np.isfinite(v) & (v > 0)
        yerr = None
        if lo is not None:
            lo = np.asarray(lo, dtype=float)[m]
            hi = np.asarray(hi, dtype=float)[m]
            g = np.isfinite(lo) & np.isfinite(hi)
            yerr = np.clip(np.vstack([np.where(g, v[m] - lo, 0.0),
                                      np.where(g, hi - v[m], 0.0)]), 0.0, None)
        ax.bar(k[m] + off, v[m] - ybot, w, bottom=ybot, color=colour,
               label=label, yerr=yerr, capsize=2.2, zorder=3,
               error_kw=dict(lw=0.8, ecolor="0.15", zorder=4))


def _log_errorbar(ax, x, y, lo, hi, **kw):
    """Asymmetric whiskers on a log axis, dropping any that are not drawable."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    g = np.isfinite(y) & (y > 0) & np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
    if g.any():
        ax.errorbar(x[g], y[g], yerr=np.clip(np.vstack([y[g] - lo[g],
                                                        hi[g] - y[g]]),
                                             0.0, None),
                    fmt="none", **kw)


def _fig_components(tab: pd.DataFrame, meta: dict, args, out_dir: Path) -> None:
    """2 x 2; (a), (b) and (d) on a log y-axis, (c) linear.

      (a) V_k per order as points and lines: prior, plug-in, posterior mean
          with its 95% range; plus the aggregate V_{k > K} of every order
          beyond those shown
      (b) fraction of each V_k retained by the plug-in
      (c) variance explained by orders <= k, with the 95% range from the
          cross-order covariance (linear axis: the curve runs 0.89 -> 1 and
          would be flat on a log scale)
      (d) share of the prior order-k variance resolved by the data
    Order 0 (the squared mean, not a variance component) is excluded.
    """
    from matplotlib.lines import Line2D
    _style()
    lvl = int(round(100 * float(meta.get("level", 0.95))))
    N = int(meta["N"])
    L = int(meta["L"])
    t1 = tab[tab["order"] >= 1].reset_index(drop=True)
    rq_all = t1["resolved_post"].to_numpy(bool)
    last_q = int(t1["order"][rq_all].max()) if rq_all.any() else 2
    Ka = max(2, min(int(args.max_order_report), last_q))
    ta = t1[t1["order"] <= Ka].reset_index(drop=True)
    k = ta["order"].to_numpy()
    have_sd = bool(np.isfinite(ta["post_sd"].to_numpy(dtype=float)).any())
    rp = ta["resolved_plugin"].to_numpy(bool)
    rq = ta["resolved_post"].to_numpy(bool)

    fig, grid = plt.subplots(2, 2, figsize=(12.6, 9.2))
    fig.subplots_adjust(wspace=0.26, hspace=0.42, top=0.88, bottom=0.07,
                        left=0.08, right=0.975)
    axes = [grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]]

    # ---- (a) order spectrum, points and lines, plus V_{k > Ka} ---------- #
    ax = axes[0]
    xa = float(Ka) + 1.3                              # position of V_{k>Ka}
    beyond = t1[t1["order"] > Ka]
    agg_prior = float(beyond["prior"].sum())
    agg_post = float(beyond["post"].sum())
    agg_plug = float(np.clip(beyond["plugin"], 0, None).sum())
    row_k = t1[t1["order"] == Ka]
    plug_ok = bool(row_k["tail_plugin_resolved"].iloc[0]) if len(row_k) else False
    post_ok = bool(row_k["tail_post_resolved"].iloc[0]) if len(row_k) else False
    agg_sd = (float(row_k["tail_abs_post_sd"].iloc[0])
              if len(row_k) and "tail_abs_post_sd" in row_k else np.nan)

    ax.plot(k, ta["prior"], "o-", color=C_NEU, ms=7, mfc="white", mew=1.4,
            lw=1.3, zorder=2)
    ax.plot(k[rp], ta["plugin"].to_numpy()[rp], "s-", color=C_ACC, ms=5.5,
            lw=1.4, zorder=3)
    ax.plot(k[rq], ta["post"].to_numpy()[rq], "^-", color=C_RM, ms=6,
            lw=1.4, zorder=4)
    if have_sd:
        _log_errorbar(ax, k[rq], ta["post"].to_numpy()[rq],
                      ta["post_lo"].to_numpy()[rq], ta["post_hi"].to_numpy()[rq],
                      ecolor=C_RM, elinewidth=1.2, capsize=3.5, capthick=1.2,
                      zorder=5)
    ax.plot([xa], [agg_prior], "o", color=C_NEU, ms=7, mfc="white", mew=1.4,
            zorder=2)
    if plug_ok and agg_plug > 0:
        ax.plot([xa], [agg_plug], "s", color=C_ACC, ms=5.5, zorder=3)
    if post_ok and agg_post > 0:
        ax.plot([xa], [agg_post], "^", color=C_RM, ms=6, zorder=4)
        if np.isfinite(agg_sd) and agg_sd > 0:
            lo_, hi_ = vcm.gamma_interval(np.array([agg_post]),
                                          np.array([agg_sd]), meta.get("level",
                                                                       0.95))
            _log_errorbar(ax, [xa], [agg_post], lo_, hi_, ecolor=C_RM,
                          elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=5)
    ax.axvline(Ka + 0.65, color="0.6", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(list(k) + [xa])
    ax.set_xticklabels([str(int(v)) for v in k] + [f"$>${Ka}"])
    ax.set_xlim(k.min() - 0.5, xa + 0.5)
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("$V_k$")
    handles = [Line2D([], [], color=C_NEU, marker="o", mfc="white", mew=1.4,
                      ms=7, lw=1.3, label="prior"),
               Line2D([], [], color=C_ACC, marker="s", ms=5.5, lw=1.4,
                      label="plug-in $f^{MAP}$"),
               Line2D([], [], color=C_RM, marker="^", ms=6, lw=1.4,
                      label="posterior mean"
                            + (f" ({lvl}% range)" if have_sd else ""))]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left")
    _panel(ax, "a", "Order spectrum")

    # ---- (b) fraction retained ------------------------------------------ #
    ax = axes[1]
    tb_ = ta[ta["resolved_plugin"].astype(bool)
             & ta["resolved_post"].astype(bool)]
    kb = tb_["order"].to_numpy()
    rb = (tb_["plugin"] / tb_["post"]).to_numpy()
    okb = np.isfinite(rb) & (rb > 0)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", zorder=2)
    ax.plot(kb[okb], rb[okb], "o-", ms=5.5, color=C_ACC, lw=1.5, zorder=4)
    for kk, rr in zip(kb[okb], rb[okb]):
        ax.annotate(f"{rr:.2f}" if rr >= 0.01 else f"{rr:.0e}", (kk, rr),
                    xytext=(7, 3), textcoords="offset points", fontsize=7.5,
                    color=C_ACC)
    ax.set_yscale("log")
    if okb.any():
        ax.set_ylim(float(rb[okb].min()) / 8, 3.0)
    ax.set_xticks(k)
    ax.set_xlim(k.min() - 0.6, k.max() + 0.6)
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("$V_k(f^{MAP})\\,/\\,E[V_k\\,|\\,y]$")
    _panel(ax, "b", "Fraction retained by the plug-in")

    # ---- (c) variance explained by order <= k (linear axis) ------------ #
    # Kept on a LINEAR axis: the cumulative fraction runs 0.89 -> 1, which is
    # a few hundredths of a decade and flat on a log axis.
    ax = axes[2]
    cp = ta["cum_plugin"].to_numpy(dtype=float)
    cq = ta["cum_post"].to_numpy(dtype=float)
    have_c = "cum_post_lo" in ta and np.isfinite(ta["cum_post_lo"]).any()
    ax.plot(k, cp, "o-", ms=5, color=C_ACC, lw=1.5, zorder=3,
            label="plug-in $f^{MAP}$")
    ax.plot(k, cq, "s-", ms=5, color=C_RM, lw=1.5, zorder=4,
            label="posterior" + (f" ({lvl}% range)" if have_c else ""))
    ylo = float(np.nanmin(np.r_[cp, cq]))
    if have_c:
        lo = ta["cum_post_lo"].to_numpy(dtype=float)
        hi = ta["cum_post_hi"].to_numpy(dtype=float)
        g = np.isfinite(lo) & np.isfinite(hi)
        ax.errorbar(k[g], cq[g],
                    yerr=np.clip(np.vstack([cq[g] - lo[g], hi[g] - cq[g]]),
                                 0.0, None),
                    fmt="none", ecolor=C_RM, elinewidth=1.2, capsize=4,
                    capthick=1.2, zorder=5)
        if g.any():
            ylo = min(ylo, float(np.nanmin(lo[g])))
    pad = 0.08 * (1.0 - ylo)
    ax.set_ylim(max(0.0, ylo - pad), 1.0 + 0.5 * pad)
    ax.set_xticks(k)
    ax.set_xlim(k.min() - 0.4, k.max() + 0.4)
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("cumulative fraction of $\\mathrm{Var}_\\pi(f)$")
    ax.legend(fontsize=7.5, loc="lower right")
    _panel(ax, "c", "Variance explained by order $\\leq k$")

    # ---- (d) what the data constrain ------------------------------------ #
    ax = axes[3]
    inf = 100.0 * ta["informed"].to_numpy(dtype=float)
    ri = ta["resolved_informed"].to_numpy(bool) & (inf > 0)
    ax.bar(k[ri], inf[ri], 0.6, color=C_INF, zorder=3)
    for kk, v in zip(k[ri], inf[ri]):
        ax.text(kk, v * 1.6, f"{v:.1f}%" if v >= 0.1 else f"{v:.0e}%",
                ha="center", va="bottom", fontsize=7.5, color="0.2")
    ax.set_yscale("log")
    if ri.any():
        ax.set_ylim(float(inf[ri].min()) / 6, 2000.0)
    ax.set_xticks(k)
    ax.set_xlim(k.min() - 0.6, k.max() + 0.6)
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("% of prior order-$k$ variance\nresolved by the data")
    ax.text(0.97, 0.95,
            r"$1-\dfrac{\mathrm{tr}(A_k\Sigma)}{\mathrm{tr}(A_kK)}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(fc="white", ec="0.75", lw=0.5, pad=3))
    _panel(ax, "d", "What the data constrain")

    fig.suptitle("Functional-ANOVA variance components over all "
                 f"$2^{{{L}}}$ sequences\n"
                 f"GP conditioned on {N:,} measured segregants for exact "
                 "posterior mean and variance of each $V_k$",
                 fontsize=11.5, y=0.985)
    save_fig(fig, out_dir, "fig_E5_variance_components")


def _fig_tail(tab: pd.DataFrame, meta: dict, args, out_dir: Path) -> None:
    """Supplementary: fraction of Var_pi(f) above each order (reverse cumsum)."""
    _style()
    t1 = tab[tab["order"] >= 1]
    td = t1[t1["order"] <= int(args.max_order_tail)]
    kd = td["order"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    tp = td["tail_prior"].to_numpy(dtype=float)
    ok = np.isfinite(tp) & (tp > 0)
    ax.plot(kd[ok], tp[ok], "-", lw=6, alpha=0.28, color=C_NEU, zorder=1,
            solid_capstyle="round", label="prior")
    mq = td["tail_post_resolved"].to_numpy(bool)
    ax.plot(kd[mq], td["tail_post"].to_numpy()[mq], "^-", ms=4, lw=1.3,
            color=C_RM, zorder=3, label="exact posterior mean")
    mp = td["tail_plugin_resolved"].to_numpy(bool)
    ax.plot(kd[mp], td["tail_plugin"].to_numpy()[mp], "s--", ms=4, lw=1.3,
            color=C_ACC, zorder=4, label="plug-in $f^{MAP}$")
    ax.set_yscale("log")
    ax.set_xlim(0.5, int(args.max_order_tail) + 0.5)
    ax.set_xlabel("interaction order $k$")
    ax.set_ylabel("fraction of $\\mathrm{Var}_\\pi(f)$ above order $k$")
    ax.legend(fontsize=7.5, loc="upper right")
    _panel(ax, "", "How much variance lies above each order")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_E5_tail")


# =========================================================================== #
# CLI
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="validate,timing",
                    help="comma-separated: validate, timing, real, replot")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--replot-from", type=Path, default=None,
                    help="replot stage only: folder of an earlier run to read "
                         "the CSVs from. Figures are written to --out-dir, so "
                         "the source folder is never modified.")
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
                   default=[24, 48, 96, 192, 384, 768])

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
                   help="conditioning-set size for the GPyTorch model; "
                        "0 = all 19,833")
    r.add_argument("--root-decomp-size", type=int, default=None)
    r.add_argument("--background", default="empirical",
                   choices=["empirical", "uniform"])
    r.add_argument("--block", type=int, default=32)
    r.add_argument("--component-variance-n", type=int, default=0,
                   help="size of the EXACT sub-problem on which rows 3-7 are "
                        "computed together for the figure. 0 = skip (figure "
                        "then has no posterior intervals). Set equal to "
                        "--n-condition to use every conditioning sequence.")
    r.add_argument("--variance-budget-hours", type=float, default=4.0,
                   help="wall-clock budget for rows 6-7. The cost is measured "
                        "on a real block first; if the projection exceeds the "
                        "budget, N is reduced to fit and the log says so.")
    r.add_argument("--variance-noise-floor", type=float, default=1e-4,
                   help="per-point noise floor in the exact rebuild "
                        "(GPyTorch's min_fixed_noise in float32)")
    r.add_argument("--component-variance-rank", type=int, nargs="+",
                   default=[], help="DEPRECATED, ignored")
    r.add_argument("--block-bivariate", type=int, default=16)
    r.add_argument("--variance-max-order", type=int, default=10,
                   help="compute posterior variances/covariances for orders "
                        "0..K only. Exact for those orders; makes the "
                        "variance pass ~58x cheaper than all 84 orders and is "
                        "what makes N = 19,833 feasible. Must be >= the "
                        "orders shown in the figure.")
    r.add_argument("--frob-chunk-gb", type=float, default=6.0,
                   help="GPU memory for one chunk of the Frobenius "
                        "accumulator; the term is streamed over chunks, so "
                        "this bounds memory at any N")
    r.add_argument("--resolution-safety", type=float, default=10.0,
                   help="a value is drawn only if it exceeds this multiple of "
                        "its own per-order error estimate")
    r.add_argument("--interval-level", type=float, default=0.95)
    r.add_argument("--max-order-report", type=int, default=8,
                   help="highest order in panels (a)-(c); also capped at the "
                        "last resolved posterior order")
    r.add_argument("--max-order-tail", type=int, default=15,
                   help="highest order in panel (d)")
    r.add_argument("--kappa-from-theta", action="store_true")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tees = []
    if not a.no_log:
        import atexit
        suffix = "_replot" if stages == {"replot"} else ""
        for name, fname in (("stdout", f"run{suffix}.log"),
                            ("stderr", f"run{suffix}.err")):
            tt = _Tee(getattr(sys, name), out_dir / fname)
            setattr(sys, name, tt)
            tees.append((name, tt))

        def _restore():
            for nm, tee in tees:
                setattr(sys, nm, tee.stream)
                tee.close()
        atexit.register(_restore)

    banner(f"E5  variance components -- stages {sorted(stages)}")
    print(f"  script   {Path(__file__).resolve()}")
    print(f"  revision {SCRIPT_REVISION}")
    print(f"           {REVISION_NOTES}")
    print(f"  module   {Path(vcm.__file__).resolve()}")
    mrev = getattr(vcm, "MODULE_REVISION", "<none>")
    print(f"  module revision {mrev}")
    if mrev != REQUIRED_MODULE_REVISION:
        print(f"\n!! gkt_variance_components.py is stale (found {mrev}, need "
              f"{REQUIRED_MODULE_REVISION}). Copy the current module and rerun.")
        return 3
    print(f"  out   {out_dir}")
    summary: dict = {"stages": sorted(stages), "revision": SCRIPT_REVISION}

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
        if not r:
            return 1
        summary["n_condition"] = r["n_condition"]
        summary["figure_N"] = r["fig_N"]
        summary["cache_rank"] = r["rank"]
        summary["identities_passed"] = r["identities"]["passed"]
        c = r["components"]
        summary["cum_frac_plugin_le2"] = float(c["cum_frac_plugin"].iloc[2])
        summary["cum_frac_posterior_le2"] = float(
            c["cum_frac_posterior"].iloc[2])
    if "replot" in stages and "real" not in stages:
        stage_replot(a, out_dir)

    if stages != {"replot"}:
        (out_dir / "E5_summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        print(f"  wrote {out_dir / 'E5_summary.json'}")
    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())