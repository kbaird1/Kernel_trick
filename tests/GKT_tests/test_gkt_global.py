#!/usr/bin/env python
"""
test_gkt_global.py   (module name: test_gkt_global)
===================================================
Verification of the GLOBAL (dataset-level) interpretation functionality added to
``shap_kernel_trick_computer.py``:

    * ``global_shap_importance``   -- mean ``|phi_i|`` over a target sample (first-order).
    * ``global_dividend_summary``  -- from dividends ``a(T)(x)`` over a target sample:
          ``mean_abs``     = mean_x |a(T)|          (global interaction importance)
          ``mean_signed``  = mean_x a(T)            (QC: ~0 for |T|>=1 when x ~ pi)
          ``sobol_unnorm`` = Var_x a(T)             (unnormalized Sobol sensitivity index)

The test fits an EpiK GP on a KNOWN synthetic landscape (additive + a pure injected pair
and triple), draws targets UNIFORMLY (so ``pi`` == the target-sampling distribution and the
Sobol interpretation is exact), and compares the sample estimates to the CLOSED-FORM truth
of the injected function.

Closed-form truth (uniform background, pure-ANOVA injected components)
----------------------------------------------------------------------
Because every injected component is a pure ANOVA term, its dividend as a function of the
target IS that component, so:
    * mean_x a(T) = 0 exactly for |T| >= 1  (ANOVA components are mean-zero)   -> QC check.
    * Var_x a(T) = Sobol(T):
          singleton p    : add_mag^2   * mean_c a[p,c]^2
          injected tuple T: inter_mag^2 * mean_cells W_T^2
          any other tuple : 0
So the injected pair / triple must appear as the dominant Sobol indices at their orders,
and all non-injected tuples must have ~0 Sobol -- the global, non-smearing analog of the
local dividend recovery.

Claims:
    - dividend_values_batch == looping dividend_values (machine precision) -- batch helper.
    - mean_signed ~ 0 for every displayed tuple (QC that x ~ pi and the ANOVA identity holds).
    - sample Sobol correlates strongly with truth Sobol; injected tuples are the dominant
      Sobol index at their own order (soft: sampling + GP-fit limited).
    - global_shap_importance elevates the positions that participate in the injected
      structure (soft, reported).

Sizes: dividends themselves are poly-time (the GKT), so the only cost is the target sample
size x number of tuples; L is kept modest (default 5, ACGT) mainly so the closed-form truth
enumerations stay trivial.

USAGE
-----
    python test_gkt_global.py --gpu
    python test_gkt_global.py --n-samples 400 --inter-mag 6
    python test_gkt_global.py --no-plot

Must sit in tests/ alongside test_gkt_brute.py / test_gkt_interactions.py (shared path
setup + synthetic-landscape builder).
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_gkt_brute import (
    ShapKernelTrickComputer,
    onehot,
    move_ktc_to_device,
    resolve_device,
    DT,
)
from test_gkt_interactions import build_interacting_ktc, all_tuples


# --------------------------------------------------------------------------- #
# Closed-form truth Sobol indices of the injected function (uniform background)
# --------------------------------------------------------------------------- #

def _truth_sobol(info, U):
    """Unnormalized Sobol index Var_uniform[f_U] for tuple U (pure-ANOVA landscape).

    Singleton p     -> add_mag^2   * mean_c a[p, c]^2
    Injected tuple T -> inter_mag^2 * mean over cells of W_T^2
    Otherwise        -> 0
    (Uses that each injected component is zero-mean under uniform, so Var == E[.^2].)
    """
    U = tuple(sorted(int(p) for p in U))
    if len(U) == 1:
        p = U[0]
        return float(info["add_mag"]) ** 2 * float(info["add"][p, :].pow(2).mean())
    if U in info["W"]:
        return float(info["inter_mag"]) ** 2 * float(info["W"][U].pow(2).mean())
    return 0.0


# =========================================================================== #
# Global interpretation test
# =========================================================================== #

def run_global_test(alphabet="ACGT", L=5, seed=0, device="cpu",
                    inject_pair=(1, 3), inject_triple=(0, 2, 4),
                    inter_mag=5.0, add_mag=1.0, kernel="jenga", n_iter=150,
                    n_samples=300, atol=1e-6, out_dir=None):
    A = len(set(alphabet))
    dev = torch.device(device)
    ip, it = tuple(sorted(inject_pair)), tuple(sorted(inject_triple))
    injected = {ip, it}

    print(f"\n{'='*78}\nGLOBAL INTERPRETATION TEST  (importance + Sobol from dividends)")
    print(f"  L={L}  alphabet='{alphabet}'  pair={ip}  triple={it}  "
          f"inter_mag={inter_mag}  add_mag={add_mag}  n_samples={n_samples}  device={device}")
    print("=" * 78)

    ktc, info = build_interacting_ktc(L, alphabet, pairs=[ip], triples=[it],
                                      add_mag=add_mag, inter_mag=inter_mag,
                                      seed=seed, kernel=kernel, n_iter=n_iter)
    if device != "cpu":
        move_ktc_to_device(ktc, device)
    ktc.set_precision(use_double=True)
    shap = ShapKernelTrickComputer(ktc=ktc)

    # Uniform background AND uniform target sampling -> Sobol interpretation is exact.
    pi = torch.full((L, A), 1.0 / A, dtype=DT, device=dev)
    gen = torch.Generator().manual_seed(seed + 7)
    tgt_idx = torch.randint(0, A, (n_samples, L), generator=gen)
    xs = [onehot(row, A, device=dev) for row in tgt_idx]

    # Tuples to summarize: all singletons, all pairs, all triples (covers both injected).
    singles = all_tuples(L, 1)
    pairs = all_tuples(L, 2)
    triples = all_tuples(L, 3)
    inter_tuples = pairs + triples          # dividend summary over interactions of order >=2
    all_disp = singles + pairs + triples

    report = []

    def check(nm, ok, detail=""):
        report.append((nm, bool(ok), detail)); return ok

    all_pass = True

    # (0) batch helper == loop (machine precision)
    T_list_b, A_batch = shap.dividend_values_batch(xs[:6], pi, interactions=inter_tuples)
    A_loop = torch.stack([shap.dividend_values(x, pi, interactions=inter_tuples)[1]
                          for x in xs[:6]])
    be = float((A_batch - A_loop).abs().max())
    all_pass &= check("dividend_values_batch == loop", be < atol, f"max abs err={be:.2e}")

    # (1) global dividend summary over interactions (with Sobol)
    T_list, summ = shap.global_dividend_summary(xs, pi, interactions=inter_tuples,
                                                normalize_sobol=True)
    mean_abs = summ["mean_abs"].detach().cpu().numpy()
    mean_signed = summ["mean_signed"].detach().cpu().numpy()
    sobol = summ["sobol_unnorm"].detach().cpu().numpy()

    # (2) QC: mean_signed ~ 0 for every interaction tuple (ANOVA components mean-zero).
    #     Threshold is scaled by the typical dividend magnitude and the sample size.
    scale = float(np.abs(mean_abs).max()) + 1e-12
    max_signed = float(np.abs(mean_signed).max())
    all_pass &= check("mean_signed ~ 0 for interactions (x ~ pi QC)",
                      max_signed < 0.15 * scale,
                      f"max|mean_signed|={max_signed:.2e}  (<= 0.15 * max|a| = {0.15*scale:.2e})")

    # (3) truth Sobol vs sample Sobol: correlation + injected dominance at each order.
    truth_sobol = np.array([_truth_sobol(info, T) for T in T_list])
    if truth_sobol.std() > 0 and sobol.std() > 0:
        corr = float(np.corrcoef(sobol, truth_sobol)[0, 1])
    else:
        corr = float("nan")
    report.append(("sample Sobol vs truth Sobol (GP+sample limited)", None,
                   f"corr={corr:.4f}"))

    orders = [len(T) for T in T_list]
    for k, injT in ((2, ip), (3, it)):
        same = [i for i, o in enumerate(orders) if o == k]
        top = T_list[same[int(np.argmax(sobol[same]))]]
        all_pass &= check(f"injected order-{k} tuple {injT} is the top Sobol at its order",
                          tuple(top) == injT, f"top Sobol order-{k} = {tuple(top)}")

    # (4) injected tuples' Sobol dominates non-injected of the same order (ratio report)
    for k, injT in ((2, ip), (3, it)):
        same = [i for i, o in enumerate(orders) if o == k]
        inj_i = T_list.index(injT)
        others = [i for i in same if T_list[i] != injT]
        inj_val = float(sobol[inj_i])
        other_max = float(np.max(sobol[others])) if others else 0.0
        report.append((f"order-{k}: injected Sobol vs next-largest", None,
                       f"injected {injT}={inj_val:.3e}, next={other_max:.3e}, "
                       f"ratio={inj_val/(other_max+1e-12):.1f}x"))

    # (5) first-order global importance: positions in the injected structure elevated.
    idx, imp = shap.global_shap_importance(xs, pi)
    imp = imp.detach().cpu().numpy()
    inj_positions = set(ip) | set(it)
    inj_mean = float(np.mean([imp[p] for p in range(L) if p in inj_positions]))
    non_mean = float(np.mean([imp[p] for p in range(L) if p not in inj_positions])) \
        if any(p not in inj_positions for p in range(L)) else float("nan")
    report.append(("global_shap_importance elevates injected positions", None,
                   f"mean|phi| injected={inj_mean:.3e}, non-injected={non_mean:.3e}"))

    for nm, ok, detail in report:
        flag = "DIAG" if ok is None else ("PASS" if ok else "FAIL")
        print(f"    [{flag}] {nm}" + (f"   ({detail})" if detail else ""))

    plot_data = dict(
        L=L, A=A, device=device, n_samples=n_samples, injected=injected,
        T_list=T_list, orders=orders, sobol=sobol, truth_sobol=truth_sobol,
        mean_abs=mean_abs, positions=list(range(L)), importance=imp,
    )
    if out_dir is not None:
        make_global_plot(plot_data, out_dir)
    return all_pass, plot_data


def make_global_plot(pd, out_dir):
    """Three panels: per-position first-order importance; Sobol truth-vs-sample; mean|a|."""
    os.makedirs(out_dir, exist_ok=True)
    T_list, orders = pd["T_list"], pd["orders"]
    labels = ["\u00d7".join(map(str, T)) for T in T_list]
    x = np.arange(len(T_list))

    fig = plt.figure(figsize=(max(12, 0.32 * len(T_list)), 10), dpi=120)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.4, 1.2], hspace=0.45)

    # (1) per-position first-order global importance
    ax0 = fig.add_subplot(gs[0])
    pos = pd["positions"]
    colors = ["#d95f02" if p in set().union(*pd["injected"]) else "#8aa0b6" for p in pos]
    ax0.bar(pos, pd["importance"], color=colors)
    ax0.set_xticks(pos)
    ax0.set_xlabel("position i"); ax0.set_ylabel("mean |phi_i|")
    ax0.set_title(f"First-order global importance (orange = participates in injected "
                  f"structure)  [L={pd['L']}, A={pd['A']}, N={pd['n_samples']}, "
                  f"device={pd['device']}]", fontsize=11, fontweight="bold")

    # (2) Sobol: truth vs sample estimate
    ax1 = fig.add_subplot(gs[1])
    w = 0.4
    ax1.bar(x - w/2, pd["truth_sobol"], w, color="#111111", label="truth Sobol (injected f)")
    ax1.bar(x + w/2, pd["sobol"],       w, color="#2b5c8f", label="sample Sobol (GKT dividends)")
    for T in pd["injected"]:
        if T in T_list:
            ax1.axvspan(T_list.index(T) - 0.5, T_list.index(T) + 0.5,
                        color="gold", alpha=0.25, zorder=0)
    for i in range(1, len(orders)):
        if orders[i] != orders[i - 1]:
            ax1.axvline(i - 0.5, color="0.7", ls="--", lw=1.0)
    ax1.set_ylabel("Sobol index  Var_x a(T)")
    ax1.set_title("Global Sobol sensitivity per interaction (injected = gold): the pure "
                  "pair and triple dominate; other tuples ~0 (no smearing)", fontsize=11)
    ax1.legend(fontsize=9, loc="best")
    ax1.set_xticks(x); ax1.set_xticklabels([])

    # (3) mean|a(T)| importance
    ax2 = fig.add_subplot(gs[2])
    ax2.bar(x, pd["mean_abs"], color="#1b7837")
    for T in pd["injected"]:
        if T in T_list:
            ax2.axvspan(T_list.index(T) - 0.5, T_list.index(T) + 0.5,
                        color="gold", alpha=0.25, zorder=0)
    for i in range(1, len(orders)):
        if orders[i] != orders[i - 1]:
            ax2.axvline(i - 0.5, color="0.7", ls="--", lw=1.0)
    ax2.set_ylabel("mean_x |a(T)|")
    ax2.set_xlabel("interaction tuple T  (pairs | triples)")
    ax2.set_title("Global interaction importance  mean_x |a(T)|", fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=90, fontsize=7)

    p = os.path.join(out_dir, "global_interpretation.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"saved figure -> {p}")


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true",
                    help="run on CUDA if visible (else CPU w/ warning).")
    ap.add_argument("--alphabet", default="ACGT")
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--kernel", default="jenga", choices=["jenga", "connectedness"])
    ap.add_argument("--n-iter", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--inject-pair", default="1,3")
    ap.add_argument("--inject-triple", default="0,2,4")
    ap.add_argument("--inter-mag", type=float, default=5.0)
    ap.add_argument("--add-mag", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=300,
                    help="number of UNIFORM target sequences to average over "
                         "(larger -> tighter Sobol / mean_signed estimates)")
    args = ap.parse_args()

    device = resolve_device(args.gpu)
    print("=" * 78)
    print("GLOBAL INTERPRETATION  (mean|phi| importance + Sobol from dividends)")
    print("=" * 78)
    print(f"torch {torch.__version__}   device={device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    out_dir = None
    if not args.no_plot:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "output", "global_test")

    all_pass, _ = run_global_test(
        alphabet=args.alphabet, L=args.L, seed=args.seed, device=device,
        inject_pair=tuple(int(v) for v in args.inject_pair.split(",")),
        inject_triple=tuple(int(v) for v in args.inject_triple.split(",")),
        inter_mag=args.inter_mag, add_mag=args.add_mag, kernel=args.kernel,
        n_iter=args.n_iter, n_samples=args.n_samples, out_dir=out_dir)

    print("\n==>", "GLOBAL INTERPRETATION TEST PASS" if all_pass
          else "GLOBAL INTERPRETATION TEST FAILED")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)