#!/usr/bin/env python
"""
test_gkt_interactions.py   (module name: test_gkt_interactions)
===============================================================
Recovery-scenario tests for the pairwise / higher-order interaction SHAP path in
``generalized_kernel_trick_computer.py`` (``shap_interaction_values``, backed by
``_shap_z_interaction`` / ``_shap_R_interaction``).

Each scenario fits an EpiK GP on a landscape with a KNOWN injected dependence
structure, then compares the interaction SHAP three ways:

    * TRUTH -- the Shapley Interaction Index (SII) of the EXACT injected function,
               computed by independent subset enumeration (``_truth_sii``). This is the
               ground-truth "structure of the dependence".
    * BF    -- SII of the fitted GP posterior mean via ``shap_brute_force.ShapBruteForce``.
    * GKT   -- SII of the same fitted model via the polynomial layer (the code tested).

Claims per scenario:
    - GKT == BF to machine precision (both explain the same fitted model) -- the core
      correctness check, now exercised at LARGE interaction values, not just ~0.
    - the injected tuple is the dominant interaction at its own order in the recovered
      values.
    - recovered vs TRUTH agree up to GP fit error (reported; limited by how faithfully
      the GP learned the landscape, not by the SHAP code).

The ladder of scenarios:
    A  additive only            (order-1 signal, interactions ~0 -- the baseline)
    B  single pairwise (i,j)    (one pure 2-way term)
    C  single 3-way (i,j,k)     (one pure 3-way term)
    D  additive + pairwise + 3-way   (the realistic mix)

A subtlety worth expecting in the graphs: SII is NOT the ANOVA decomposition. A pure
k-way term has NONZERO SII on every SUBSET of its positions and exactly zero elsewhere,
so scenario C's triple lights up the triple bar AND its sub-pairs / singletons. The
TRUTH curve captures this spill-down exactly, so truth and recovered show the same
pattern.

Sizes: brute force and the truth enumeration are ~O((1+A)^L); keep L small (default 5,
alphabet ACGT -> exact and fast). This file computes interaction MEANS only (no
covariance), so there is no O(A^{2L}) kernel.

USAGE
-----
    # inside a GPU allocation (srun ... --gres=gpu:1 --pty bash), env activated:
    python test_gkt_interactions.py --gpu
    python test_gkt_interactions.py --gpu --inject-pair 1,3 --inject-triple 0,2,4
    python test_gkt_interactions.py --gpu --inter-mag 8 --n-iter 250   # help a stubborn fit
    python test_gkt_interactions.py                                    # CPU (falls back w/ warning)

This file must sit in the SAME tests/ directory as test_gkt_brute.py: importing it runs
that module's top-level sys.path setup, which makes epik / modules importable.
"""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
import torch

# Importing test_gkt_brute also runs its top-level path configuration, so epik / modules
# resolve inside build_interacting_ktc. We reuse its SHAP computer, brute-force reference,
# one-hot helper, device mover, device resolver, and float64 dtype.
from test_gkt_brute import (
    ShapKernelTrickComputer,
    ShapBruteForce,
    onehot,
    move_ktc_to_device,
    resolve_device,
    DT,
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def all_tuples(L, k):
    """All size-``k`` position tuples over ``range(L)`` (sorted, as the GKT expects)."""
    return [tuple(t) for t in itertools.combinations(range(L), k)]


def make_targets(L, A, n, seed):
    """``n`` random target sequences as index rows (CPU); ``onehot`` places them later."""
    gen = torch.Generator().manual_seed(seed + 1)
    return [row for row in torch.randint(0, A, (n, L), generator=gen)]


# =========================================================================== #
# Interaction-landscape builder
# --------------------------------------------------------------------------- #
# Injects a KNOWN pure interaction into the ground truth before fitting, so the SHAP
# interaction values can then be checked against a truth we constructed.
# =========================================================================== #

def _pure_interaction(W):
    """Project a tensor onto its pure top-order ANOVA component (uniform weights).

    Zeroing the mean along every axis leaves a tensor whose marginal over any proper
    subset of axes is zero -- a term that is genuinely order-|axes| with no lower-order
    part. For 2-D this is ordinary double-centering; the one-pass-per-axis loop is exact
    at any order because the per-axis centering projectors commute and are idempotent.
    """
    for ax in range(W.dim()):
        W = W - W.mean(dim=ax, keepdim=True)
    return W


def build_interacting_ktc(L=5, alphabet="ACGT", pairs=(), triples=(), quads=(),
                          add_mag=1.0, inter_mag=5.0, ptrain=0.75, n_iter=150,
                          learning_rate=0.1, seed=0, kernel="jenga", noise=0.01):
    """Fit an EpiK GP on ``additive + injected pure interactions`` and return its KTC.

    Landscape: ``y = add_mag * sum_p a[p, x_p] + inter_mag * sum_T W_T[x_T]``, with ``a``
    a per-position pure (zero-mean) additive effect and each ``W_T`` a pure |T|-way
    interaction (all marginals zero, via :func:`_pure_interaction`) for T in
    ``pairs`` / ``triples`` / ``quads``. The components are built in float64 so the truth
    SII is exact; the GP is trained in float32 as usual. Dense sampling (``ptrain``) and
    low ``noise`` keep the posterior mean close to the true landscape so an injected
    interaction survives into the fit. Returns ``(ktc, info)`` where ``info`` records the
    injected tuples AND the component tensors (for the truth SII).
    """
    from epik.epik.kernel import JengaKernel, ConnectednessKernel
    from epik.epik.model import EpiK
    from epik.epik.utils import encode_seqs, split_training_test
    from modules.kernel_trick_computer_refactored import KernelTrickComputer

    torch.manual_seed(seed)
    np.random.seed(seed)
    A = len(set(alphabet))
    syms = sorted(set(alphabet))

    all_seqs = ["".join(p) for p in itertools.product(syms, repeat=L)]
    X = encode_seqs(all_seqs, alphabet=alphabet)                 # (N, L*A) one-hot
    N = len(all_seqs)
    idx = X.reshape(N, L, A).argmax(dim=-1)                      # (N, L) char indices

    # pure additive part (float64): y_add[n] = sum_p a[p, x_{n,p}]
    a = _pure_interaction(torch.randn(L, A, dtype=torch.float64))   # zero per-position mean
    y = add_mag * a[torch.arange(L).unsqueeze(0), idx].sum(dim=1)   # (N,)

    info = {"pairs": [], "triples": [], "quads": [], "inter_mag": inter_mag,
            "add_mag": add_mag, "L": L, "A": A, "add": a.detach().cpu(), "W": {}}

    def inject(T, key_list):
        T = tuple(int(v) for v in T)
        W = _pure_interaction(torch.randn(*([A] * len(T)), dtype=torch.float64))
        cols = tuple(idx[:, p] for p in T)
        nonlocal y
        y = y + inter_mag * W[cols]
        key = tuple(sorted(T))
        key_list.append(key); info["W"][key] = W.detach().cpu()

    for T in pairs:
        inject(T, info["pairs"])
    for T in triples:
        inject(T, info["triples"])
    for T in quads:
        inject(T, info["quads"])

    y_var = torch.full((N,), float(noise))
    train_x, train_y, test_x, test_y, train_y_var = split_training_test(
        X, y, y_var, ptrain=ptrain, dtype=torch.float32
    )
    kernel_cls = {"jenga": JengaKernel, "connectedness": ConnectednessKernel}[kernel]
    model = EpiK(kernel_cls(n_alleles=A, seq_length=L), track_progress=True)
    model.set_data(train_x, train_y, train_y_var)
    model.fit(n_iter=n_iter, learning_rate=learning_rate)
    model.predict(test_x, calc_variance=True)                   # populate caches
    return KernelTrickComputer(model, alphabet, L), info


# =========================================================================== #
# Ground-truth SII of the exact injected function (independent of GKT and BF)
# =========================================================================== #

def _truth_f(info, x):
    """Exact injected landscape value f_true(x) for an index sequence ``x``."""
    L = info["L"]
    y = info["add_mag"] * sum(float(info["add"][p, x[p]]) for p in range(L))
    for T, W in info["W"].items():
        y = y + info["inter_mag"] * float(W[tuple(int(x[p]) for p in T)])
    return y


def _truth_sii(info, U, x, L, A):
    """Enumerated SII of the EXACT injected function for tuple ``U`` (uniform pi).

    Evaluates ``f_true`` directly and applies the same subset / discrete-derivative
    formula used everywhere else, so it is independent of both GKT and BF. Cheap because
    ``f_true`` is a handful of table look-ups. (For a pure top-order term this reproduces
    the closed form ``SII_T = inter_mag * W_T[x_T]``; the enumeration also captures the
    nonzero SII spill-down onto subsets of an injected tuple.)
    """
    U = tuple(sorted(U)); k = len(U)
    inv = 1.0 / A
    rest = [p for p in range(L) if p not in U]

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
    for r in range(len(rest) + 1):
        gw = math.exp(math.lgamma(r + 1) + math.lgamma(L - k - r + 1)
                      - math.lgamma(L - k + 2))
        for S in itertools.combinations(rest, r):
            d = 0.0
            for wsz in range(k + 1):
                for W in itertools.combinations(U, wsz):
                    d += (-1.0) ** (k - len(W)) * cond_exp(tuple(S) + tuple(W))
            acc += gw * d
    return acc


# =========================================================================== #
# One scenario: fit -> compute TRUTH / BF / GKT -> check -> plot
# =========================================================================== #

def run_scenario(name, desc, pairs, triples, add_mag, inter_mag, L, alphabet,
                 seed, device, kernel="jenga", n_iter=150, quads=(), atol=1e-6,
                 out_dir=None):
    """Fit a known landscape, then compare TRUTH vs BF vs GKT interaction SHAP.

    Returns ``(all_pass, plot_data)``. Hard checks: GKT == BF (machine precision); the
    truth enumerator matches the top-order closed form; and GKT recovers the SAME
    dominant interaction as the truth at each injected order (robust to SII spill-down,
    where a subset of an injected tuple can carry the largest value). Reported (soft):
    per-tuple recovery and the overall recovered-vs-truth agreement, both limited by GP
    fit quality, not the SHAP code.
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

    # representative target: set each injected tuple's positions to the argmax cell of
    # |W_T| so the injected signal is strong and legible; fill the rest deterministically.
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

    # ---- three signed SII curves for this target -------------------------- #
    def gkt_of(Ts):
        _, phi = shap.shap_interaction_values(x_oh, pi, interactions=Ts, normalize="sii")
        return phi
    gkt = torch.cat([gkt_of(per_order[k]) for k in range(1, max_order + 1)]
                    ).detach().cpu().numpy()
    bfv = torch.cat([bf.bf_shap_interaction_values(x_oh, pi, per_order[k], convention="sii")
                     for k in range(1, max_order + 1)]).detach().cpu().numpy()
    truth = np.array([_truth_sii(info, T, x.tolist(), L, A) for T in tuples])

    report = []

    def check(nm, ok, detail=""):
        report.append((nm, bool(ok), detail)); return ok

    all_pass = True
    # (1) GKT == BF everywhere (core correctness, now at nonzero values)
    e = float(np.abs(gkt - bfv).max())
    all_pass &= check(f"[{name}] GKT == brute force (all displayed tuples, SII)",
                      e < atol, f"max abs err={e:.2e}, at |phi| up to {np.abs(bfv).max():.2e}")

    # (2) truth top-order identity: enumerated truth SII(injected) == inter_mag * W_T[x_T].
    #     RELATIVE tolerance: the enumeration is a heavily-cancelling signed sum, so a
    #     3-way term accumulates ~1e-8 float64 error at |phi| ~ several -- expected, not a bug.
    idabs = idrel = 0.0
    for T in injected:
        expected = info["inter_mag"] * float(info["W"][T][tuple(int(x[p]) for p in T)])
        er = abs(float(truth[tuples.index(T)]) - expected)
        idabs = max(idabs, er); idrel = max(idrel, er / (abs(expected) + 1e-12))
    if injected:
        all_pass &= check(f"[{name}] truth SII(injected) == inter_mag * W[x]",
                          idrel < 1e-6, f"max rel err={idrel:.2e} (abs {idabs:.2e})")

    # (3) structure recovery: at each injected order, GKT's dominant interaction matches
    #     TRUTH's dominant interaction. The injected tuple is NOT always the largest at
    #     its order -- SII spills DOWN onto subsets of an injected tuple, so a pair shared
    #     by an injected triple and quad can exceed a standalone injected pair. The right
    #     recovery check is "GKT reproduces the truth's ranking", not "the injected tuple
    #     is mechanically on top".
    for k in sorted({len(T) for T in injected}):
        same = [i for i, o in enumerate(orders) if o == k]
        r_top = tuples[same[int(np.abs(gkt[same]).argmax())]]
        t_top = tuples[same[int(np.abs(truth[same]).argmax())]]
        all_pass &= check(f"[{name}] GKT recovers truth's top order-{k} interaction",
                          r_top == t_top, f"GKT top={r_top}, truth top={t_top}")

    # (3b) report each injected tuple's recovery (GKT vs truth). A ratio < 100% is GP
    #      shrinkage of (esp. high-order) structure, NOT a SHAP error -- GKT == BF above
    #      confirms the code; this line quantifies how well the GP learned that term.
    for T in sorted(injected, key=len):
        pos = tuples.index(T)
        frac = (f"{gkt[pos] / truth[pos] * 100:.0f}% of truth"
                if abs(truth[pos]) > 1e-9 else "n/a")
        report.append((f"[{name}] injected {T} recovered", None,
                       f"truth={truth[pos]:+.3f}, GKT={gkt[pos]:+.3f} ({frac})"))

    # (4) recovered vs truth (SOFT: limited by GP fit, reported not asserted)
    rel = float(np.abs(gkt - truth).max() / (np.abs(truth).max() + 1e-30))
    corr = float(np.corrcoef(gkt, truth)[0, 1]) if truth.std() > 0 else float("nan")
    report.append((f"[{name}] recovered vs truth (GP-fit-limited)", None,
                   f"corr={corr:.4f}, max|GKT-truth|/max|truth|={rel:.2%}"))

    for nm, ok, detail in report:
        flag = "DIAG" if ok is None else ("PASS" if ok else "FAIL")
        print(f"    [{flag}] {nm}" + (f"   ({detail})" if detail else ""))

    plot_data = dict(name=name, desc=desc, L=L, A=A, device=device, tgt=tgt_str,
                     tuples=tuples, orders=orders, injected=injected,
                     truth=truth, bf=bfv, gkt=gkt)
    if out_dir is not None:
        make_scenario_plot(plot_data, out_dir)
    return all_pass, plot_data


def make_scenario_plot(pd, out_dir):
    """Grouped bar chart per scenario: TRUTH vs BF vs GKT SII over all displayed tuples."""
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except Exception as e:  # noqa: BLE001
        print(f"[note] skipping scenario plot (matplotlib unavailable): {e}")
        return
    os.makedirs(out_dir, exist_ok=True)

    tuples, orders = pd["tuples"], pd["orders"]
    labels = ["\u00d7".join(map(str, T)) for T in tuples]
    x = _np.arange(len(tuples))

    fig, ax = plt.subplots(figsize=(max(11, 0.34 * len(tuples)), 6.0), dpi=120)
    w = 0.27
    ax.bar(x - w, pd["truth"], w, color="#111111", label="truth (injected f)")
    ax.bar(x,     pd["bf"],    w, color="#d95f02", label="brute force (GP)")
    ax.bar(x + w, pd["gkt"],   w, color="#2b5c8f", label="GKT (GP)")
    ax.axhline(0, color="0.5", lw=0.7)

    # order dividers at each order boundary + highlight injected tuples
    for i in range(1, len(orders)):
        if orders[i] != orders[i - 1]:
            ax.axvline(i - 0.5, color="0.7", ls="--", lw=1.0)
    for T in pd["injected"]:
        ax.axvspan(tuples.index(T) - 0.5, tuples.index(T) + 0.5,
                   color="gold", alpha=0.25, zorder=0)

    order_names = {1: "order 1 (positions)", 2: "order 2 (pairs)",
                   3: "order 3 (triples)", 4: "order 4 (quads)"}
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("interaction SHAP (SII)")
    ax.set_xlabel("   |   ".join(order_names.get(k, f"order {k}")
                                 for k in sorted(set(orders))))
    ax.set_title(f"Scenario {pd['name']}: {pd['desc']}\n"
                 f"truth vs brute force vs GKT  (L={pd['L']}, A={pd['A']}, "
                 f"target '{pd['tgt']}', device={pd['device']}; injected = gold)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    p = os.path.join(out_dir, f"scenario_{pd['name']}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"saved figure -> {p}")


def run_all_scenarios(alphabet="ACGT", L=5, seed=0, device="cpu",
                      inject_pair=(1, 3), inject_triple=(0, 2, 4),
                      inter_mag=5.0, add_mag=1.0, kernel="jenga", n_iter=150,
                      out_dir=None):
    """Run the additive / pairwise / triple / combined ladder; return overall pass."""
    ip, it = tuple(sorted(inject_pair)), tuple(sorted(inject_triple))
    print(f"\n{'='*78}\nRECOVERY SCENARIOS  (truth vs brute force vs GKT)")
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
        ok, _ = run_scenario(nm, desc, prs, trs, am, im, L, alphabet, seed, device,
                             kernel=kernel, n_iter=n_iter, out_dir=out_dir)
        all_pass = all_pass and ok

    # Scenario E: additive baseline + 2nd + 3rd + 4th order together. Needs a 4th-order
    # term, so it uses a DEDICATED non-nested position set (pair (2,3), triple (0,1,2),
    # quad (0,1,3,4) -- they share position 2, which the quad omits) rather than the
    # --inject-* tuples, so each order has a unique dominant injected tuple. Requires L>=5.
    if L >= 5:
        eP, eT, eQ = (2, 3), (0, 1, 2), (0, 1, 3, 4)
        ok, _ = run_scenario(
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
                    help="run the GKT and brute-force paths on CUDA if visible "
                         "(falls back to CPU with a warning).")
    ap.add_argument("--alphabet", default="ACGT", help="symbol set (default 'ACGT').")
    ap.add_argument("--L", type=int, default=5,
                    help="sequence length; keep small -- truth/BF enumerate ~(1+A)^L.")
    ap.add_argument("--kernel", default="jenga", choices=["jenga", "connectedness"])
    ap.add_argument("--n-iter", type=int, default=150, help="GP fit iterations.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--inject-pair", default="1,3",
                    help="positions of the injected pairwise interaction (e.g. '1,3').")
    ap.add_argument("--inject-triple", default="0,2,4",
                    help="positions of the injected 3-way interaction (e.g. '0,2,4').")
    ap.add_argument("--inter-mag", type=float, default=5.0,
                    help="magnitude of injected interactions (raise if a high-order term "
                         "is not recovered -- that reflects GP fit, not the SHAP code).")
    ap.add_argument("--add-mag", type=float, default=1.0,
                    help="magnitude of the additive part (scenarios A and D).")
    args = ap.parse_args()

    device = resolve_device(args.gpu)
    print("=" * 78)
    print("INTERACTION SHAP RECOVERY  (truth vs brute force vs GKT)")
    print("=" * 78)
    print(f"torch {torch.__version__}   device={device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    out_dir = None
    if not args.no_plot:
        import os
        # <project>/tests/output/interactions_test/  (this file lives in tests/)
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "output", "interactions_test")

    all_pass = run_all_scenarios(
        alphabet=args.alphabet, L=args.L, seed=args.seed, device=device,
        inject_pair=tuple(int(v) for v in args.inject_pair.split(",")),
        inject_triple=tuple(int(v) for v in args.inject_triple.split(",")),
        inter_mag=args.inter_mag, add_mag=args.add_mag, kernel=args.kernel,
        n_iter=args.n_iter, out_dir=out_dir)

    print("\n==>", "ALL RECOVERY SCENARIOS PASS" if all_pass
          else "SOME RECOVERY SCENARIOS FAILED")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)