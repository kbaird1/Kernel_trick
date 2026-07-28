#!/usr/bin/env python
"""
test_gkt_dividends.py   (module name: test_gkt_dividends)
=========================================================
Verification of the **Harsanyi dividend** path (== full-order Faith-Shap; the *pure*,
non-smearing interaction) added to ``shap_kernel_trick_computer.py``
(``dividend_values``, backed by ``_shap_z_interaction`` / ``_shap_R_interaction`` with
the ``[1, 0, 0, ...]`` R-weight). It mirrors ``test_gkt_interactions.py`` but targets the
dividend index, and adds the ``test_gkt_brute.py`` style runtime / operation-count /
speedup benchmark so we confirm the NEW path also runs in polynomial (not exponential)
time.

For each scenario an EpiK GP is fitted on a landscape with a KNOWN injected dependence
structure, then dividends are compared FOUR ways:

    * TRUTH_CF  -- closed-form dividend of the injected function. Because the injected
                   components are PURE ANOVA terms (all marginals zero), the dividend is
                   trivial: ``add_mag * a[p, x_p]`` for a singleton, ``inter_mag * W_T[x_T]``
                   for an injected tuple ``T``, and EXACTLY ZERO for every other tuple.
    * TRUTH_EN  -- the same dividend of the injected function computed by INDEPENDENT
                   subset enumeration (``_truth_dividend``); must equal TRUTH_CF.
    * BF        -- dividend of the fitted GP posterior mean via
                   ``ShapBruteForce.bf_dividend_values`` (enumeration).
    * GKT       -- dividend of the same fitted model via the polynomial layer (tested code).

Core claims per scenario:
    - GKT == BF to machine precision, for BOTH the dividend mean and the dividend
      posterior covariance (both explain the same fitted model, by independent routes).
    - TRUTH_EN == TRUTH_CF (validates the truth we compare against).
    - NON-SMEARING (the headline): for a pure order-k injected tuple, the dividend on every
      PROPER SUBSET is ZERO in truth and ~0 in GKT (GP-fit-limited) -- whereas SII on those
      same subsets is NONZERO. This is the clean-vs-smeared contrast, shown numerically and
      in the figure's second row.
    - GKT recovers the truth's dominant structure at each injected order.

The ladder mirrors the interactions test:
    A additive only     B single pair     C single 3-way     D additive+pair+3-way
plus E (adds a 4-way) when L>=5.

Sizes: brute force / truth enumeration are ~O((1+A)^L) for the mean and ~O(A^{2L}) for the
covariance, so keep L small (default 5, ACGT -> exact and fast). Covariance is checked on a
reduced tuple set to bound the O(A^{2L}) cost.

USAGE
-----
    python test_gkt_dividends.py --gpu
    python test_gkt_dividends.py --inject-pair 1,3 --inject-triple 0,2,4 --inter-mag 8
    python test_gkt_dividends.py --benchmark --bench-L-max 6      # poly-vs-exp timing sweep
    python test_gkt_dividends.py                                  # CPU (falls back w/ warning)

Must sit in the SAME tests/ directory as test_gkt_brute.py and test_gkt_interactions.py:
importing them runs the path setup that makes epik / modules importable, and reuses the
synthetic-landscape builder + truth machinery so the two interaction tests share one model.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from time import perf_counter

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the SHAP computer, brute-force reference, helpers, and float64 dtype.
from test_gkt_brute import (
    ShapKernelTrickComputer,
    ShapBruteForce,
    onehot,
    move_ktc_to_device,
    resolve_device,
    build_real_ktc,
    DT,
)
# Reuse the SAME synthetic interacting landscape + its truth machinery as the SII test,
# so dividends and SII are compared on identical models.
from test_gkt_interactions import (
    build_interacting_ktc,
    _truth_f,
    _truth_sii,
    all_tuples,
    make_targets,
)


# --------------------------------------------------------------------------- #
# Truth dividends of the exact injected function
# --------------------------------------------------------------------------- #

def _truth_dividend(info, U, x, L, A):
    """Enumerated Harsanyi dividend ``a(U) = Delta_U f_true(empty)`` (uniform pi).

        a(U) = sum_{W subseteq U} (-1)^{|U|-|W|} E_uniform[f_true | X_W = x_W].

    Independent of both GKT and BF: it evaluates the exact injected ``f_true`` and applies
    the discrete-derivative-at-``S=empty`` definition. For the pure injected landscape this
    equals the functional-ANOVA component (see :func:`_truth_dividend_closedform`).
    """
    U = tuple(sorted(U)); k = len(U)
    inv = 1.0 / A

    def cond_exp(S):                                   # E_uniform[f_true | X_S = x_S]
        S = set(S); free = [p for p in range(L) if p not in S]
        tot = 0.0
        for combo in itertools.product(range(A), repeat=len(free)):
            z = list(x)
            for p, c in zip(free, combo):
                z[p] = c
            tot += (inv ** len(free)) * _truth_f(info, z)
        return tot

    acc = 0.0
    for wsz in range(k + 1):
        for W in itertools.combinations(U, wsz):
            acc += (-1.0) ** (k - len(W)) * cond_exp(tuple(W))
    return acc


def _truth_dividend_closedform(info, U, x):
    """Closed-form dividend of the injected function (pure-ANOVA structure).

    Singleton -> ``add_mag * a[p, x_p]``; injected tuple ``T`` -> ``inter_mag * W_T[x_T]``;
    anything else -> ``0`` exactly. This is the whole point of the dividend: a pure k-way
    term contributes to its own set and to NOTHING smaller.
    """
    U = tuple(sorted(int(p) for p in U))
    if len(U) == 1:
        p = U[0]
        return info["add_mag"] * float(info["add"][p, int(x[p])])
    if U in info["W"]:
        return info["inter_mag"] * float(info["W"][U][tuple(int(x[p]) for p in U)])
    return 0.0


# =========================================================================== #
# One scenario: fit -> TRUTH / BF / GKT dividends (+ SII contrast) -> check -> plot
# =========================================================================== #

def run_dividend_scenario(name, desc, pairs, triples, add_mag, inter_mag, L, alphabet,
                          seed, device, kernel="jenga", n_iter=150, quads=(), atol=1e-6,
                          out_dir=None):
    """Fit a known landscape, then compare TRUTH vs BF vs GKT dividends (and SII contrast).

    Returns ``(all_pass, plot_data)``.
    """
    A = len(set(alphabet))
    dev = torch.device(device)
    pairs = [tuple(sorted(p)) for p in pairs]
    triples = [tuple(sorted(t)) for t in triples]
    quads = [tuple(sorted(q)) for q in quads]
    injected = set(pairs) | set(triples) | set(quads)

    print(f"\n{'-'*78}\nSCENARIO {name}: {desc}")
    print(f"  inject pairs={pairs}  triples={triples}  quads={quads}  "
          f"add_mag={add_mag}  inter_mag={inter_mag}")

    ktc, info = build_interacting_ktc(L, alphabet, pairs=pairs, triples=triples,
                                      quads=quads, add_mag=add_mag, inter_mag=inter_mag,
                                      seed=seed, kernel=kernel, n_iter=n_iter)
    if device != "cpu":
        move_ktc_to_device(ktc, device)
    ktc.set_precision(use_double=True)
    shap = ShapKernelTrickComputer(ktc=ktc)
    bf = ShapBruteForce(shap)

    pi = torch.full((L, A), 1.0 / A, dtype=DT, device=dev)       # uniform matches purity

    # representative target: argmax cell of each injected |W_T| so the signal is legible
    x = make_targets(L, A, 1, seed)[0].clone()
    for T in injected:
        W = info["W"][T]
        cell = np.unravel_index(int(W.abs().argmax()), tuple(W.shape))
        for p, c in zip(T, cell):
            x[p] = int(c)
    x_oh = onehot(x, A, device=dev)
    tgt_str = "".join(alphabet[c] for c in x.tolist())

    # display tuples: all positions plus every order up to the highest injected order
    max_order = min(L, max([3] + [len(T) for T in injected]))
    per_order = {k: all_tuples(L, k) for k in range(1, max_order + 1)}
    tuples = [T for k in range(1, max_order + 1) for T in per_order[k]]
    orders = [len(T) for T in tuples]

    # ---- dividends: GKT, BF, and both truths --------------------------------- #
    def gkt_div(Ts):
        _, a = shap.dividend_values(x_oh, pi, interactions=Ts)
        return a
    gkt = torch.cat([gkt_div(per_order[k]) for k in range(1, max_order + 1)]
                    ).detach().cpu().numpy()
    bfv = torch.cat([bf.bf_dividend_values(x_oh, pi, per_order[k])
                     for k in range(1, max_order + 1)]).detach().cpu().numpy()
    truth_en = np.array([_truth_dividend(info, T, x.tolist(), L, A) for T in tuples])
    truth_cf = np.array([_truth_dividend_closedform(info, T, x) for T in tuples])

    # ---- SII on the same tuples, for the non-smearing contrast --------------- #
    def gkt_sii(Ts):
        _, s = shap.shap_interaction_values(x_oh, pi, interactions=Ts, normalize="sii")
        return s
    sii = torch.cat([gkt_sii(per_order[k]) for k in range(1, max_order + 1)]
                    ).detach().cpu().numpy()
    sii_truth = np.array([_truth_sii(info, T, x.tolist(), L, A) for T in tuples])

    report = []

    def check(nm, ok, detail=""):
        report.append((nm, bool(ok), detail)); return ok

    all_pass = True

    # (1) GKT == BF (dividend mean), core correctness at nonzero values
    e = float(np.abs(gkt - bfv).max())
    all_pass &= check(f"[{name}] GKT == brute force (dividend mean, all tuples)",
                      e < atol, f"max abs err={e:.2e}, |a| up to {np.abs(bfv).max():.2e}")

    # (2) truth enumeration == truth closed form (validates the reference)
    et = float(np.abs(truth_en - truth_cf).max())
    all_pass &= check(f"[{name}] truth dividend: enumeration == closed form",
                      et < 1e-6, f"max abs err={et:.2e}")

    # (3) GKT == BF (dividend covariance) on a reduced tuple set (bound O(A^{2L}))
    cov_tuples = list(per_order[1]) + list(per_order.get(2, []))[:3] + list(injected)
    cov_tuples = [tuple(sorted(T)) for T in dict.fromkeys(cov_tuples)]   # de-dup, keep order
    _, _, cov_gkt = shap.dividend_values(x_oh, pi, interactions=cov_tuples, calc_covar=True)
    cov_bf = bf.bf_dividend_covariance(x_oh, pi, cov_tuples)
    ce = float((cov_gkt.detach().cpu() - cov_bf.detach().cpu()).abs().max())
    all_pass &= check(f"[{name}] GKT == brute force (dividend covariance)",
                      ce < 1e-5, f"max abs err={ce:.2e} over {len(cov_tuples)} tuples")
    sym = float((cov_gkt - cov_gkt.transpose(-1, -2)).abs().max())
    all_pass &= check(f"[{name}] dividend covariance symmetric", sym < atol,
                      f"max asym={sym:.2e}")

    # (4) NON-SMEARING: proper subsets of an injected tuple carry ZERO dividend from that
    #     tuple, while SII on those same subsets is NONZERO.
    #
    #     Subtlety for multi-effect scenarios: dividends are ADDITIVE over the injected
    #     effects, so a subset S of T is only zero in truth if S is not ALSO an injected
    #     effect in its own right. Two cases make a proper subset legitimately nonzero and
    #     have nothing to do with smearing:
    #       - singletons when add_mag != 0 (every position carries an order-1 main effect);
    #       - any subset that equals another injected interaction tuple.
    #     We therefore test only proper subsets that are NOT themselves injected effects.
    #     (In the pure-interaction scenarios B/C with add_mag == 0 this excludes nothing and
    #     the check is exactly "all proper subsets == 0", as before.)
    def _is_own_effect(S):
        if len(S) == 1 and add_mag != 0.0:   # order-1 main effect present at every position
            return True
        return S in injected                 # another injected interaction of its own order
    for T in sorted(injected, key=len, reverse=True):
        if len(T) < 2:
            continue
        subs = [S for k in range(1, len(T)) for S in itertools.combinations(T, k)]
        subs = [tuple(sorted(S)) for S in subs]
        test_subs = [S for S in subs if not _is_own_effect(S)]
        # If every proper subset is itself an injected effect (e.g. a pair under add_mag>0),
        # there is nothing for smearing to leak onto; skip rather than vacuously pass.
        if not test_subs:
            report.append((f"[{name}] non-smearing subsets of {T} all self-injected (skipped)",
                           None, "no non-injected proper subset to test"))
            continue
        sub_idx = [tuples.index(S) for S in test_subs]
        sii_sub_idx = [tuples.index(S) for S in subs]   # SII contrast over ALL subsets
        div_sub_truth = float(np.abs(truth_cf[sub_idx]).max())
        sii_sub_truth = float(np.abs(sii_truth[sii_sub_idx]).max())
        all_pass &= check(
            f"[{name}] dividend does NOT smear onto non-injected subsets of {T} (truth==0)",
            div_sub_truth < 1e-9,
            f"max|non-injected subset dividend truth|={div_sub_truth:.2e}  "
            f"vs max|subset SII truth|={sii_sub_truth:.2e}")
        # GKT (fitted model): subset dividends small relative to the injected tuple's own
        inj_pos = tuples.index(T)
        div_sub_gkt = float(np.abs(gkt[sub_idx]).max())
        ratio = div_sub_gkt / (abs(gkt[inj_pos]) + 1e-12)
        report.append((f"[{name}] GKT subset dividends of {T} small vs own (GP-limited)",
                       None, f"max|subset|/|{T}| = {ratio:.2%} "
                             f"(subset {div_sub_gkt:.3f}, own {gkt[inj_pos]:+.3f})"))

    # (5) structure recovery: GKT's dominant dividend at each injected order == truth's
    for k in sorted({len(T) for T in injected}):
        same = [i for i, o in enumerate(orders) if o == k]
        r_top = tuples[same[int(np.abs(gkt[same]).argmax())]]
        t_top = tuples[same[int(np.abs(truth_cf[same]).argmax())]]
        all_pass &= check(f"[{name}] GKT recovers truth's top order-{k} dividend",
                          r_top == t_top, f"GKT top={r_top}, truth top={t_top}")

    # (6) per-injected-tuple recovery report (GKT vs truth; ratio < 100% is GP shrinkage)
    for T in sorted(injected, key=len):
        pos = tuples.index(T)
        frac = (f"{gkt[pos] / truth_cf[pos] * 100:.0f}% of truth"
                if abs(truth_cf[pos]) > 1e-9 else "n/a")
        report.append((f"[{name}] injected {T} dividend recovered", None,
                       f"truth={truth_cf[pos]:+.3f}, GKT={gkt[pos]:+.3f} ({frac})"))

    for nm, ok, detail in report:
        flag = "DIAG" if ok is None else ("PASS" if ok else "FAIL")
        print(f"    [{flag}] {nm}" + (f"   ({detail})" if detail else ""))

    plot_data = dict(name=name, desc=desc, L=L, A=A, device=device, tgt=tgt_str,
                     tuples=tuples, orders=orders, injected=injected,
                     truth=truth_cf, bf=bfv, gkt=gkt, sii=sii, sii_truth=sii_truth)
    if out_dir is not None:
        make_dividend_plot(plot_data, out_dir)
    return all_pass, plot_data


def make_dividend_plot(pd, out_dir):
    """Two-row figure: (top) dividend truth/BF/GKT; (bottom) SII truth/GKT -- the contrast."""
    os.makedirs(out_dir, exist_ok=True)
    tuples, orders = pd["tuples"], pd["orders"]
    labels = ["\u00d7".join(map(str, T)) for T in tuples]
    x = np.arange(len(tuples))

    fig, (ax_d, ax_s) = plt.subplots(2, 1, figsize=(max(11, 0.34 * len(tuples)), 9.2),
                                     dpi=120, sharex=True)
    w = 0.27

    # top: dividends (the clean, non-smearing index)
    ax_d.bar(x - w, pd["truth"], w, color="#111111", label="truth (injected f)")
    ax_d.bar(x,     pd["bf"],    w, color="#d95f02", label="brute force (GP)")
    ax_d.bar(x + w, pd["gkt"],   w, color="#2b5c8f", label="GKT (GP)")
    ax_d.axhline(0, color="0.5", lw=0.7)
    ax_d.set_ylabel("Harsanyi dividend  a(T)\n(full-order Faith-Shap)")
    ax_d.set_title(f"Scenario {pd['name']}: {pd['desc']}\n"
                   f"DIVIDENDS -- pure interaction, no smearing  "
                   f"(L={pd['L']}, A={pd['A']}, target '{pd['tgt']}', "
                   f"device={pd['device']}; injected = gold)",
                   fontsize=12, fontweight="bold")
    ax_d.legend(fontsize=9, loc="best")

    # bottom: SII on the same tuples (smears onto subsets)
    ax_s.bar(x - w/2, pd["sii_truth"], w, color="#111111", label="truth SII (injected f)")
    ax_s.bar(x + w/2, pd["sii"],       w, color="#7570b3", label="GKT SII (GP)")
    ax_s.axhline(0, color="0.5", lw=0.7)
    ax_s.set_ylabel("Shapley Interaction Index\n(SII -- smears down)")
    ax_s.set_title("SII on the SAME tuples: a pure high-order term spills onto its subsets "
                   "(contrast with the clean dividends above)", fontsize=11)
    ax_s.legend(fontsize=9, loc="best")

    order_names = {1: "order 1 (positions)", 2: "order 2 (pairs)",
                   3: "order 3 (triples)", 4: "order 4 (quads)"}
    for ax in (ax_d, ax_s):
        for i in range(1, len(orders)):
            if orders[i] != orders[i - 1]:
                ax.axvline(i - 0.5, color="0.7", ls="--", lw=1.0)
        for T in pd["injected"]:
            ax.axvspan(tuples.index(T) - 0.5, tuples.index(T) + 0.5,
                       color="gold", alpha=0.25, zorder=0)
    ax_s.set_xticks(x); ax_s.set_xticklabels(labels, rotation=90, fontsize=7)
    ax_s.set_xlabel("   |   ".join(order_names.get(k, f"order {k}")
                                   for k in sorted(set(orders))))
    fig.tight_layout()
    p = os.path.join(out_dir, f"dividends_scenario_{pd['name']}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"saved figure -> {p}")


# =========================================================================== #
# Operation-count model + benchmark for the DIVIDEND path
# =========================================================================== #

def flops_bruteforce_dividend(L, A, t, tuples):
    """Op-count for ShapBruteForce.bf_dividend_values on a fixed tuple set.

    Per tuple U: a(U) = sum_{W subseteq U} (+/-) E_uniform[f | X_W = x_W]; each conditional
    expectation enumerates A^(L-|W|) sequences at f-eval cost ~ (L+1)*t. The |W|=0 term
    (A^L per tuple) dominates -> exponential in L.
    """
    f_eval = (L + 1) * t
    total = 0.0
    for U in tuples:
        k = len(U)
        for wsz in range(k + 1):
            total += math.comb(k, wsz) * (A ** (L - wsz)) * f_eval
    return total


def flops_gkt_dividend(L, A, t, tuples):
    """Op-count for dividend_values on a fixed tuple set (channels + per-tuple poly).

    Channels ~ L*(A^2+A)*t (shared). Per tuple: a polynomial product over L degree-<=1
    factors (~L convolutions, each O(L*t)) then apply_R ~ L*t -> ~ L^2*t. No A^L factor.
    """
    channels = L * (A * A + A) * t
    per_tuple = sum((L * L) * t + L * t for _ in tuples)
    return channels + per_tuple


def _time_call(fn, repeats=5, warmup=1, device="cpu"):
    is_cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        best = min(best, perf_counter() - t0)
    return best


def run_dividend_benchmark(L_values, alphabet="ACGT", ptrain=0.5, n_iter=60,
                           kernel="jenga", targets=3, seed=0, repeats=5, device="cpu"):
    """Sweep L; time GKT vs brute-force DIVIDENDS for a fixed tuple set {(0,1),(0,1,2)}.

    The tuple set is held constant across L (needs L>=3) so the plot shows pure L-scaling:
    polynomial for the GKT, exponential (A^L) for the brute force.
    """
    A = len(set(alphabet))
    dev = torch.device(device)
    fixed_tuples = [(0, 1), (0, 1, 2)]
    rows = []
    print(f"\n[device: {device}]  dividend tuples = {fixed_tuples}")
    print(f"{'L':>3} {'t':>5} {'GKT (ms)':>12} {'BF (ms)':>12} "
          f"{'speedup':>10} {'GKT flops':>12} {'BF flops':>12}")
    for L in L_values:
        ktc = build_real_ktc(L=L, alphabet=alphabet, ptrain=ptrain,
                             n_iter=n_iter, seed=seed, kernel=kernel)
        if device != "cpu":
            move_ktc_to_device(ktc, device)
        ktc.set_precision(use_double=True)
        shap = ShapKernelTrickComputer(ktc=ktc)
        bf = ShapBruteForce(shap)
        t = ktc._train_x.reshape(-1, L, A).shape[0]

        gen = torch.Generator().manual_seed(seed + 1)
        pi = torch.rand(L, A, generator=gen, dtype=DT)
        pi = (pi / pi.sum(dim=1, keepdim=True)).to(dev)
        all_seqs = list(itertools.product(range(A), repeat=L))
        tsel = torch.randperm(len(all_seqs), generator=gen)[:targets].tolist()
        xs = [onehot(torch.tensor(all_seqs[i]), A, device=dev) for i in tsel]

        gkt_times, bf_times = [], []
        for x_oh in xs:
            gkt_times.append(_time_call(
                lambda: shap.dividend_values(x_oh, pi, interactions=fixed_tuples),
                repeats, device=device))
            bf_times.append(_time_call(
                lambda: bf.bf_dividend_values(x_oh, pi, fixed_tuples),
                repeats, device=device))
        gkt_s = sorted(gkt_times)[len(gkt_times) // 2]
        bf_s = sorted(bf_times)[len(bf_times) // 2]

        fl_gkt = flops_gkt_dividend(L, A, t, fixed_tuples)
        fl_bf = flops_bruteforce_dividend(L, A, t, fixed_tuples)
        rows.append(dict(L=L, t=t, gkt_s=gkt_s, bf_s=bf_s, fl_gkt=fl_gkt, fl_bf=fl_bf))
        print(f"{L:>3} {t:>5} {gkt_s*1e3:>12.3f} {bf_s*1e3:>12.3f} "
              f"{bf_s/gkt_s:>9.1f}x {fl_gkt:>12.2e} {fl_bf:>12.2e}")

    return dict(A=A, targets=targets, device=device, tuples=fixed_tuples,
                L=[r["L"] for r in rows], t=[r["t"] for r in rows],
                gkt_s=[r["gkt_s"] for r in rows], bf_s=[r["bf_s"] for r in rows],
                fl_gkt=[r["fl_gkt"] for r in rows], fl_bf=[r["fl_bf"] for r in rows])


def make_dividend_benchmark_plot(bench, out_dir):
    out = os.path.join(out_dir, "dividends_benchmark.png")
    L = bench["L"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=110)

    ax = axes[0]
    ax.plot(L, [s * 1e3 for s in bench["bf_s"]], "o-", color="#d95f02", label="Brute force")
    ax.plot(L, [s * 1e3 for s in bench["gkt_s"]], "o-", color="#2b5c8f", label="GKT")
    ax.set_yscale("log"); ax.set_xticks(L)
    ax.set_xlabel("L (sequence length)"); ax.set_ylabel("wall-clock per target (ms, log)")
    ax.set_title(f"Dividend runtime  [{bench.get('device', 'cpu')}]"); ax.legend()

    ax = axes[1]
    ax.plot(L, bench["fl_bf"], "s--", color="#d95f02", label="Brute force  ~A^L")
    ax.plot(L, bench["fl_gkt"], "s--", color="#2b5c8f", label="GKT  ~poly(L)")
    ax.set_yscale("log"); ax.set_xticks(L)
    ax.set_xlabel("L (sequence length)"); ax.set_ylabel("analytical flops (log)")
    ax.set_title("Modeled operation count"); ax.legend()

    ax = axes[2]
    meas = [bench["bf_s"][k] / bench["gkt_s"][k] for k in range(len(L))]
    modeled = [bench["fl_bf"][k] / bench["fl_gkt"][k] for k in range(len(L))]
    ax.plot(L, meas, "o-", color="#1b7837", label="measured (wall-clock)")
    ax.plot(L, modeled, "s--", color="#762a83", label="modeled (flops)")
    ax.set_yscale("log"); ax.set_xticks(L)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("L (sequence length)"); ax.set_ylabel("brute force / GKT  (x, log)")
    ax.set_title("Dividend speedup"); ax.legend()

    plt.suptitle(f"Dividends (full-order Faith-Shap): exponential -> polynomial "
                 f"(A={bench['A']}, tuples={bench['tuples']}, "
                 f"device={bench.get('device', 'cpu')})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"saved benchmark figure -> {out}")


# =========================================================================== #
# Scenario ladder
# =========================================================================== #

def run_all_scenarios(alphabet="ACGT", L=5, seed=0, device="cpu",
                      inject_pair=(1, 3), inject_triple=(0, 2, 4),
                      inter_mag=5.0, add_mag=1.0, kernel="jenga", n_iter=150,
                      out_dir=None):
    ip, it = tuple(sorted(inject_pair)), tuple(sorted(inject_triple))
    print(f"\n{'='*78}\nDIVIDEND RECOVERY SCENARIOS  (truth vs brute force vs GKT)")
    print(f"  L={L}  alphabet='{alphabet}'  pair={ip}  triple={it}  "
          f"inter_mag={inter_mag}  device={device}")
    print("=" * 78)

    scenarios = [
        ("A_additive", "additive only (no interactions)",       [],   [],   add_mag, 0.0),
        ("B_pairwise", f"single pairwise interaction {ip}",      [ip], [],   0.0,     inter_mag),
        ("C_triple",   f"single 3-way interaction {it}",         [],   [it], 0.0,     inter_mag),
        ("D_combined", f"additive + pairwise {ip} + 3-way {it}", [ip], [it], add_mag, inter_mag),
    ]
    all_pass = True
    for nm, desc, prs, trs, am, im in scenarios:
        ok, _ = run_dividend_scenario(nm, desc, prs, trs, am, im, L, alphabet, seed, device,
                                      kernel=kernel, n_iter=n_iter, out_dir=out_dir)
        all_pass = all_pass and ok

    if L >= 5:
        eP, eT, eQ = (2, 3), (0, 1, 2), (0, 1, 3, 4)
        ok, _ = run_dividend_scenario(
            "E_all_orders",
            f"additive + pairwise {eP} + 3-way {eT} + 4-way {eQ}",
            [eP], [eT], add_mag, inter_mag, L, alphabet, seed, device,
            kernel=kernel, n_iter=n_iter, quads=[eQ], out_dir=out_dir)
        all_pass = all_pass and ok
    else:
        print("\n[note] skipping scenario E (needs L>=5 for a 4th-order interaction).")

    return all_pass


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true",
                    help="run GKT + brute force on CUDA if visible (else CPU w/ warning).")
    ap.add_argument("--alphabet", default="ACGT")
    ap.add_argument("--L", type=int, default=5, help="keep small: truth/BF enumerate ~(1+A)^L.")
    ap.add_argument("--kernel", default="jenga", choices=["jenga", "connectedness"])
    ap.add_argument("--n-iter", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--inject-pair", default="1,3")
    ap.add_argument("--inject-triple", default="0,2,4")
    ap.add_argument("--inter-mag", type=float, default=5.0)
    ap.add_argument("--add-mag", type=float, default=1.0)
    # benchmark
    ap.add_argument("--benchmark", action="store_true",
                    help="run the dividend GKT-vs-brute-force timing/flop sweep over L")
    ap.add_argument("--bench-L-min", type=int, default=3)
    ap.add_argument("--bench-L-max", type=int, default=6)
    ap.add_argument("--bench-targets", type=int, default=3)
    ap.add_argument("--bench-repeats", type=int, default=5)
    args = ap.parse_args()

    device = resolve_device(args.gpu)
    print("=" * 78)
    print("DIVIDEND (FULL-ORDER FAITH-SHAP) TESTS  (truth vs brute force vs GKT)")
    print("=" * 78)
    print(f"torch {torch.__version__}   device={device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    out_dir = None
    if not args.no_plot:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "output", "dividends_test")

    if args.benchmark:
        A = len(set(args.alphabet))
        L_values = list(range(args.bench_L_min, args.bench_L_max + 1))
        projected = A ** args.bench_L_max
        print(f"\nBenchmark sweep L={L_values}, A={A}")
        if projected > 5e5:
            print(f"  [warn] brute-force dividend at L={args.bench_L_max} scales ~A^L "
                  f"= {projected:.1e} evals/tuple; lower --bench-L-max if slow.")
        bench = run_dividend_benchmark(
            L_values, alphabet=args.alphabet, kernel=args.kernel,
            targets=args.bench_targets, seed=args.seed, repeats=args.bench_repeats,
            device=device)
        if not args.no_plot:
            os.makedirs(out_dir, exist_ok=True)
            make_dividend_benchmark_plot(bench, out_dir)
        print("\n==> DIVIDEND BENCHMARK COMPLETE")
        return True

    all_pass = run_all_scenarios(
        alphabet=args.alphabet, L=args.L, seed=args.seed, device=device,
        inject_pair=tuple(int(v) for v in args.inject_pair.split(",")),
        inject_triple=tuple(int(v) for v in args.inject_triple.split(",")),
        inter_mag=args.inter_mag, add_mag=args.add_mag, kernel=args.kernel,
        n_iter=args.n_iter, out_dir=out_dir)

    print("\n==>", "ALL DIVIDEND SCENARIOS PASS" if all_pass
          else "SOME DIVIDEND SCENARIOS FAILED")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)