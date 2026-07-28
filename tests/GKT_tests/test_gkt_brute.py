"""
test_gkt_vs_bruteforce.py   (module name: test_gkt_brute)
=========================================================
End-to-end verification of the SHAP *mean* path in
``shap_kernel_trick_computer.py`` against a brute-force reference that lives in its
OWN module (``shap_brute_force.ShapBruteForce``, constructed from the SHAP computer).

Both paths explain the *same* posterior-mean function (they read the same cached
``site_kernels`` / ``_train_x`` / ``Qy`` off the real ``KernelTrickComputer``), but
compute by completely different routes: the GKT uses the polynomial layer (imported
from ``generalized_kernel_trick``), while the ``ShapBruteForce.bf_*`` methods enumerate
subsets directly. Agreement is therefore evidence the SHAP-mean implementation is
correct rather than a modelling coincidence.

The sequence space is kept small because the brute force is O((1 + |A|)^L).

Requires the same stack as the notebook: torch, epik, gpytorch, linear_operator,
pykeops. Run:
    python test_gkt_brute.py                       # L=4, alphabet=ABC (CPU)
    python test_gkt_brute.py --gpu                 # same checks, run on CUDA if available
    python test_gkt_brute.py --L 5 --alphabet ABC --targets 8
    python test_gkt_brute.py --kernel connectedness --n-iter 150
    python test_gkt_brute.py --benchmark --bench-L-max 6         # timing/flop sweep (CPU)
    python test_gkt_brute.py --benchmark --gpu --bench-L-max 6   # sweep on the GPU

Exercises the mean side of the SHAP computer (g-weights, channels, mean rows,
shap_values, indices subset, shap_values_batch, base_value, local accuracy),
prints a per-check report, and writes ``output/gkt_vs_bruteforce.png``.

With ``--benchmark`` it instead sweeps L, timing GKT vs brute-force wall-clock and
comparing against an analytical operation-count model (exponential (1+A)^L for the
brute force vs polynomial for the GKT), writing
``output/gkt_vs_bruteforce_benchmark.png``.

Module layout note
------------------
The GKT/SHAP code was split into two modules: ``generalized_kernel_trick`` (the reusable
polynomial layer) and ``shap_kernel_trick_computer`` (``ShapKernelTrickComputer``). This
file imports the computer from the latter; the old ``generalized_kernel_trick_computer``
shim has been removed. Sibling tests (``test_gkt_gpu.py``, ``test_gkt_interactions.py``)
import the computer *through this module's namespace*, so they needed no change.

GPU (``--gpu``)
--------------
Moves the fitted ``KernelTrickComputer`` and all its cached tensors onto CUDA
*before* constructing ``ShapKernelTrickComputer`` / ``ShapBruteForce`` (both capture
``ktc.device`` at init), then runs the same GKT and brute-force paths there. Both
paths agree across devices, so the checks stay meaningful; the only reason to keep
``L`` small is the brute force's O((1+A)^L) cost, which is device-independent. Timing
is CUDA-synchronised so wall-clock reflects real kernel completion, not launch time.
If ``--gpu`` is passed without a visible CUDA device the run falls back to CPU with a
warning. (For a dedicated CPU-vs-GPU study of the GKT path alone, see test_gkt_gpu.py.)
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard library + third-party imports (no ordering constraints)
# --------------------------------------------------------------------------- #
import argparse
import itertools
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

# matplotlib is imported here with the non-interactive Agg backend selected
# BEFORE pyplot, which is required for headless figure saving.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = torch.float64


# --------------------------------------------------------------------------- #
# Path configuration  (must run before any epik import)
# --------------------------------------------------------------------------- #
print("\n" + "="*50)
print("DEBUGGING & CONFIGURING PATHS")
print("="*50)

# 1. Find the project root directory ('kernel_trick').
#    This file now lives in tests/GKT_tests/, so the root is TWO levels up
#    (../../). If you move this file, adjust the number of ".." accordingly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(f"Calculated ROOT_DIR:    {ROOT_DIR}")

# 2. Set up paths for Python to find our internal modules
MODULES_DIR = os.path.join(ROOT_DIR, "modules")

print(f"Modules directory:     {MODULES_DIR}")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

# 3. Import epik_loader and configure the epik package path.
import epik_loader

# 4. epik addresses its own submodules as `epik.epik.*` (see epik/epik/kernel.py:
#    `from epik.epik.utils import ...`), so the name `epik` must resolve to the
#    OUTER wrapper directory kernel_trick/epik, making epik.epik -> the inner
#    package kernel_trick/epik/epik. We therefore point the loader at ROOT_DIR:
#    it appends `/epik` internally, so inner_pkg_dir = kernel_trick/epik (which
#    exists), and `import epik` lands on kernel_trick/epik/__init__.py.
#    NOTE: this requires kernel_trick/epik/__init__.py AND
#    kernel_trick/epik/epik/__init__.py to both exist.
epik_loader.ensure_epik_loaded(epik_code_root=ROOT_DIR)
print("="*50 + "\n")


# --------------------------------------------------------------------------- #
# epik + project-module imports
# These MUST stay below the path configuration above: they only resolve once
# ensure_epik_loaded() has put the right directory on sys.path. The GKT computer
# now lives in shap_kernel_trick_computer; its polynomial layer is in
# generalized_kernel_trick; the brute-force reference is in shap_brute_force.
# --------------------------------------------------------------------------- #
from epik.epik.kernel import JengaKernel, ConnectednessKernel
from epik.epik.model import EpiK
from epik.epik.utils import encode_seqs, split_training_test

from modules.kernel_trick_computer_refactored import KernelTrickComputer
from modules.shap_kernel_trick_computer import ShapKernelTrickComputer
from modules.shap_brute_force import ShapBruteForce


# =========================================================================== #
# Build the REAL KernelTrickComputer from a small fitted EpiK model
# =========================================================================== #

def build_real_ktc(
    L=4,
    alphabet="ABC",
    ptrain=0.5,
    n_iter=100,
    learning_rate=0.1,
    seed=0,
    kernel="jenga",
):
    """Fit a small EpiK model on a simulated additive landscape and return a real
    ``KernelTrickComputer``.

    The whole A^L space is enumerated, given an additive ground-truth fitness plus
    noise, split into train/test, and used to fit an EpiK GP. A single ``predict``
    call populates the mean/covar caches the KTC reads.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    A = len(set(alphabet))
    all_seqs = ["".join(p) for p in itertools.product(alphabet, repeat=L)]
    X = encode_seqs(all_seqs, alphabet=alphabet)

    # simulated additive fitness  y = X @ effects  (+ heteroscedastic noise var)
    effects = torch.rand(L * A)
    y = X @ effects
    y_var = torch.rand(len(all_seqs))

    train_x, train_y, test_x, test_y, train_y_var = split_training_test(
        X, y, y_var, ptrain=ptrain, dtype=torch.float32
    )

    kernel_cls = {"jenga": JengaKernel, "connectedness": ConnectednessKernel}[kernel]
    model = EpiK(kernel_cls(n_alleles=A, seq_length=L), track_progress=True)
    model.set_data(train_x, train_y, train_y_var)
    model.fit(n_iter=n_iter, learning_rate=learning_rate)
    model.predict(test_x, calc_variance=True)   # populates prediction_strategy caches

    return KernelTrickComputer(model, alphabet, L)


def onehot(x_idx, A, device=None):
    """One-hot encode a length-L index sequence to an ``(L, A)`` float64 tensor.

    ``device`` places the result (default CPU); callers that run on the GPU pass the
    target device so targets/pi live alongside the moved ``ktc`` caches.
    """
    L = len(x_idx)
    oh = torch.zeros(L, A, dtype=DT, device=device)
    oh[torch.arange(L, device=device), torch.as_tensor(x_idx, device=device)] = 1.0
    return oh


# =========================================================================== #
# Device placement
# =========================================================================== #

def move_ktc_to_device(ktc, device):
    """Move a ``KernelTrickComputer``'s model + all cached tensors onto ``device``.

    Must run BEFORE constructing ``ShapKernelTrickComputer`` / ``ShapBruteForce``:
    both capture ``ktc.device`` at init (and the GKT builds ``self.g`` there), and
    ``ShapBruteForce`` allocates its enumeration tensors on ``ktc.device``. Some GKT
    paths cast cached tensors with ``.to(dtype)`` only (no device move), so every
    tensor attribute on ``ktc`` must already be on the right device. Returns the list
    of moved attribute names.
    """
    if hasattr(ktc, "model") and hasattr(ktc.model, "to"):
        try:
            ktc.model = ktc.model.to(device)
        except Exception as e:  # noqa: BLE001 -- best effort; GKT uses cached tensors
            print(f"  [note] could not move live model to {device}: {e}")

    moved = []
    for name, val in list(vars(ktc).items()):
        if torch.is_tensor(val):
            setattr(ktc, name, val.to(device))
            moved.append(name)

    ktc.device = device
    return moved


def resolve_device(want_gpu: bool) -> str:
    """Return 'cuda' if the GPU was requested AND available, else 'cpu' (with a warning)."""
    if not want_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    print("[warn] --gpu requested but no CUDA device is visible; running on CPU. "
          "Are you inside a GPU allocation (srun ... --gres=gpu:1)?")
    return "cpu"


# =========================================================================== #
# Verification battery  (GKT mean path vs. the ShapBruteForce reference)
# =========================================================================== #

def run_checks(ktc, n_targets=8, seed=0, atol=1e-6, device="cpu"):
    if device != "cpu":
        move_ktc_to_device(ktc, device)         # place caches before building computers
    ktc.set_precision(use_double=True)          # exact float64 for both methods
    shap = ShapKernelTrickComputer(ktc=ktc)     # GKT (polynomial) path
    bf = ShapBruteForce(shap)                   # enumeration reference (own module)

    L, A, alphabet = ktc.seq_length, ktc.n_alleles, ktc.alphabet
    dev = torch.device(device)

    gen = torch.Generator().manual_seed(seed + 1)   # CPU generator; move samples after
    pi = torch.rand(L, A, generator=gen, dtype=DT)
    pi = (pi / pi.sum(dim=1, keepdim=True)).to(dev)

    all_seqs = list(itertools.product(range(A), repeat=L))
    tsel = torch.randperm(len(all_seqs), generator=gen)[:n_targets].tolist()
    targets = [torch.tensor(all_seqs[i]) for i in tsel]   # index rows (CPU); onehot places them

    report = []

    def check(name, cond, detail=""):
        report.append((name, bool(cond), detail))

    # 1. g-weights identity: sum_s C(L-1,s) g(s) == 1
    g = shap.g
    ident = sum(math.comb(L - 1, s) * float(g[s]) for s in range(L))
    check("g_weights identity  sum C(L-1,s) g(s) == 1", abs(ident - 1.0) < 1e-12,
          f"value={ident:.12f}")

    # 2. channels have the right shape
    c0, c1 = shap._mean_channels(onehot(targets[0], A, device=dev), pi, 1.0)
    t = ktc._train_x.reshape(-1, L, A).shape[0]
    check("_mean_channels shapes (L, t)", c0.shape == (L, t) and c1.shape == (L, t),
          f"c0={tuple(c0.shape)}, c1={tuple(c1.shape)}, t={t}")

    # 3. shap_values (GKT) vs brute force, over all targets  (the core check)
    gkt_all, bf_all = [], []
    for x_idx in targets:
        x_oh = onehot(x_idx, A, device=dev)
        gkt_all.append(shap.shap_values(x_oh, pi))
        bf_all.append(bf.bf_shap_values(x_oh, pi))
    gkt_all = torch.stack(gkt_all); bf_all = torch.stack(bf_all)
    max_err = float((gkt_all - bf_all).abs().max())
    check("shap_values == brute force (all targets)", max_err < atol,
          f"max abs err={max_err:.2e}")

    # 4. indices subset matches the full computation
    idx = list(range(0, L, 2))
    sub = shap.shap_values(onehot(targets[0], A, device=dev), pi, indices=idx)
    full = shap.shap_values(onehot(targets[0], A, device=dev), pi)
    check("indices subset matches full", torch.allclose(sub, full[idx], atol=atol),
          f"idx={idx}")

    # 5. shap_values_batch == per-target loop
    xs = [onehot(x, A, device=dev) for x in targets[:4]]
    batched = shap.shap_values_batch(xs, pi)
    looped = torch.stack([shap.shap_values(x, pi) for x in xs])
    check("shap_values_batch == loop", torch.allclose(batched, looped, atol=atol))

    # 6. base_value (GKT) == brute-force base value == E_pi[f]
    phi0 = float(shap.base_value(pi))
    phi0_bf = float(bf.bf_base_value(onehot(targets[0], A, device=dev), pi))
    check("base_value == E_pi[f]", abs(phi0 - phi0_bf) < atol,
          f"gkt={phi0:.6f}, bf={phi0_bf:.6f}")

    # 7. local accuracy: f(x) == phi0 + sum_i phi_i   (GKT diagnostic)
    la = float(shap.local_accuracy_residual(onehot(targets[0], A, device=dev), pi))
    check("local_accuracy_residual ~ 0 (GKT)", abs(la) < 1e-5, f"residual={la:.2e}")

    # 8. local accuracy via the independent brute-force path
    la_bf = float(bf.bf_local_accuracy_residual(onehot(targets[0], A, device=dev), pi))
    check("local_accuracy_residual ~ 0 (brute force)", abs(la_bf) < 1e-5,
          f"residual={la_bf:.2e}")

    # 9. COVARIANCE: GKT (calc_covar=True) vs brute-force Theorem-1 assembly.
    #    Both use ktc.S (covar_cache root, S S^T = Q + Qy Qy^T); the GKT builds
    #    R = m^i K m^j via the bivariate polynomial layer, the brute force via
    #    explicit full-space rows. Independent paths -> agreement is real evidence.
    cov_gkt_all, cov_bf_all = [], []
    for x_idx in targets:
        x_oh = onehot(x_idx, A, device=dev)
        _, cg = shap.shap_values(x_oh, pi, calc_covar=True)
        cb = bf.bf_shap_covariance(x_oh, pi)
        cov_gkt_all.append(cg); cov_bf_all.append(cb)
    cov_gkt_all = torch.stack(cov_gkt_all); cov_bf_all = torch.stack(cov_bf_all)
    cov_max_err = float((cov_gkt_all - cov_bf_all).abs().max())
    check("shap covariance == brute force (all targets)", cov_max_err < 1e-5,
          f"max abs err={cov_max_err:.2e}")

    # 10. covariance symmetry (basic sanity on the GKT output)
    sym_err = float((cov_gkt_all - cov_gkt_all.transpose(-1, -2)).abs().max())
    check("GKT covariance symmetric", sym_err < atol, f"max asym={sym_err:.2e}")

    # Move everything the plot/report touches back to CPU (numpy needs host tensors).
    return dict(report=report,
                gkt=gkt_all.detach().cpu(), bf=bf_all.detach().cpu(),
                targets=targets, pi=pi.detach().cpu(),
                L=L, A=A, alphabet=alphabet, t=t, max_err=max_err, device=device,
                cov_gkt=cov_gkt_all.detach().cpu(), cov_bf=cov_bf_all.detach().cpu(),
                cov_max_err=cov_max_err)


# =========================================================================== #
# Plot
# =========================================================================== #

def make_plot(res, out_dir):
    out = os.path.join(out_dir, "gkt_vs_bruteforce.png")

    gkt = res["gkt"].numpy(); bfv = res["bf"].numpy()
    L, alphabet = res["L"], res["alphabet"]
    targets = res["targets"]
    n_targets = len(targets)
    pos = range(L)

    corr = float(torch.corrcoef(torch.stack([res["gkt"].ravel(), res["bf"].ravel()]))[0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=110)
    axes = axes.ravel()

    # (1) parity
    ax = axes[0]
    x = bfv.ravel(); y = gkt.ravel()
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#d95f02",
            lw=1.5, label="y = x")
    ax.scatter(x, y, s=45, color="#2b5c8f", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Brute-force SHAP  $\\phi_i$")
    ax.set_ylabel("GKT SHAP  $\\phi_i$")
    ax.set_title(f"Parity: all target x position pairs\n"
                 f"max abs err = {res['max_err']:.1e},  r = {corr:.6f}")
    ax.legend(loc="upper left"); ax.set_aspect("equal", adjustable="datalim")

    # (2) per-position bars for one target
    ax = axes[1]
    w = 0.4
    ax.bar([p - w / 2 for p in pos], bfv[0], w, label="Brute force", color="#d95f02", alpha=0.85)
    ax.bar([p + w / 2 for p in pos], gkt[0], w, label="GKT", color="#2b5c8f", alpha=0.85)
    seq_str = "".join(alphabet[c] for c in targets[0].tolist())
    ax.set_xticks(list(pos)); ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("position i"); ax.set_ylabel("SHAP  $\\phi_i$")
    ax.set_title(f"Per-position SHAP, target = '{seq_str}'"); ax.legend()

    # (3) per-target error (log)
    ax = axes[2]
    err = (gkt - bfv).__abs__()
    for ti in range(n_targets):
        ax.plot(list(pos), err[ti], marker="o", ms=4, lw=1, alpha=0.7)
    ax.set_yscale("log"); ax.set_xticks(list(pos))
    ax.axhline(1e-9, color="green", ls="--", lw=1, label="1e-9")
    ax.set_xlabel("position i"); ax.set_ylabel("|GKT - brute force|  (log)")
    ax.set_title("Mean SHAP: abs error per target (log)"); ax.legend()

    # (4) covariance parity: all entries of cov[i,j] across all targets
    ax = axes[3]
    if "cov_gkt" in res:
        cg = res["cov_gkt"].reshape(-1).numpy()
        cb = res["cov_bf"].reshape(-1).numpy()
        lo, hi = min(cg.min(), cb.min()), max(cg.max(), cb.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#d95f02",
                lw=1.5, label="y = x")
        ax.scatter(cb, cg, s=30, color="#1b7837", alpha=0.7, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Brute-force cov$[i,j]$"); ax.set_ylabel("GKT cov$[i,j]$")
        ax.set_title(f"Covariance parity (all i,j, all targets)\n"
                     f"max abs err = {res['cov_max_err']:.1e}")
        ax.legend(loc="upper left"); ax.set_aspect("equal", adjustable="datalim")
    else:
        ax.axis("off")

    dev_note = f"  [device: {res.get('device', 'cpu')}]"
    plt.suptitle("GKT vs brute-force SHAP: mean + covariance "
                 "(real KernelTrickComputer, simulated EpiK model)" + dev_note,
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"saved figure -> {out}")


# =========================================================================== #
# Analytical operation-count model
# --------------------------------------------------------------------------- #
# These are closed-form op-count estimates derived from the ALGORITHMS (not a
# profiler), so they show algorithmic scaling (exponential vs polynomial) rather
# than implementation/backend noise. Each is auditable against the math in the
# notes and the code; the exact constants don't matter for the scaling story,
# only the presence/absence of the (1+A)^L factor.
# =========================================================================== #

def flops_bruteforce(L: int, A: int, t: int, n_indices: int = None) -> float:
    """Op-count for the brute-force SHAP mean (ShapBruteForce.bf_shap_values).

    phi_i = sum_{S subset of the other L-1 positions} g(|S|)(f_x(S u i) - f_x(S)).
    A conditional expectation over `free` free positions visits A^free sequence
    assignments, each costing one f-evaluation ~ (L+1)*t flops (L per-position
    kernel mults over t training points + t multiply-adds against Qy).

      evals_per_index = sum_{s=0}^{L-1} C(L-1,s) * (A^(L-s) + A^(L-s-1))

    The A^(L-s) term dominates: sum_s C(L-1,s) A^(L-s) = A*(1+A)^(L-1), giving the
    (1+A)^L growth stated in the module docstring.

    NOTE: this is a mild UNDER-estimate of the real code, which recomputes the full
    per-position product on every bf_posterior_mean call (no partial-product memo),
    so the measured speedup is a lower bound on the true one.
    """
    if n_indices is None:
        n_indices = L
    f_eval_cost = (L + 1) * t
    evals_per_index = 0.0
    for s in range(L):
        c = math.comb(L - 1, s)
        evals_per_index += c * (A ** (L - s) + A ** (L - s - 1))
    return n_indices * evals_per_index * f_eval_cost


def flops_gkt(L: int, A: int, t: int, n_indices: int = None) -> float:
    """Op-count for the GKT SHAP mean (_mean_channels + _shap_z_mean, prefix/suffix).

    (1) channels ~ L*(A^2 + A)*t.
    (2) prefix/suffix products (shared across indices) ~ 2*L^2*t.
    (3) per index i: one convolution prefix[i](deg i) x suffix[i](deg L-1-i)
        ~ (i+1)*(L-i)*t, plus apply_R ~ L*t.
    Dominant terms are polynomial in L with NO A^L factor -- that's the whole point.
    """
    if n_indices is None:
        n_indices = L
    channels = L * (A * A + A) * t
    prefix_suffix = 2.0 * (L ** 2) * t
    per_index = 0.0
    for i in range(L):
        per_index += (i + 1) * (L - i) * t + L * t
    per_index *= (n_indices / L)
    return channels + prefix_suffix + per_index


# =========================================================================== #
# Benchmark: GKT vs brute force across a sweep of L
# =========================================================================== #

def _time_call(fn, repeats: int, warmup: int = 1, device: str = "cpu") -> float:
    """Best-of-`repeats` wall-clock seconds for `fn` (after `warmup` untimed calls).

    Best-of (min) is used rather than mean: it is the least contaminated by OS
    scheduling / GC pauses, and we want the intrinsic cost, not the noise. On CUDA we
    synchronise around each timed call, because kernel launches are asynchronous and
    ``perf_counter`` alone would measure launch time, not completion.
    """
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


def run_benchmark(L_values, alphabet="ABC", ptrain=0.5, n_iter=60,
                  kernel="jenga", targets=3, seed=0, repeats=5, device="cpu"):
    """Sweep L; for each, fit a model and time GKT vs brute-force SHAP means.

    Returns a dict of parallel lists suitable for make_benchmark_plot. Records both
    measured wall-clock (median-of-targets, best-of-repeats per target) and the
    analytical op-counts for GKT and brute force. With ``device='cuda'`` both paths
    run (and are timed, synchronised) on the GPU.
    """
    A = len(set(alphabet))
    dev = torch.device(device)
    rows = []
    print(f"\n[device: {device}]")
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

        # time per target, take the median target to represent "typical" cost
        gkt_times, bf_times = [], []
        for x_oh in xs:
            gkt_times.append(_time_call(lambda: shap.shap_values(x_oh, pi), repeats, device=device))
            bf_times.append(_time_call(lambda: bf.bf_shap_values(x_oh, pi), repeats, device=device))
        gkt_s = sorted(gkt_times)[len(gkt_times) // 2]
        bf_s = sorted(bf_times)[len(bf_times) // 2]

        fl_gkt = flops_gkt(L, A, t)
        fl_bf = flops_bruteforce(L, A, t)
        rows.append(dict(L=L, t=t, gkt_s=gkt_s, bf_s=bf_s,
                         fl_gkt=fl_gkt, fl_bf=fl_bf))
        print(f"{L:>3} {t:>5} {gkt_s*1e3:>12.3f} {bf_s*1e3:>12.3f} "
              f"{bf_s/gkt_s:>9.1f}x {fl_gkt:>12.2e} {fl_bf:>12.2e}")

    return dict(
        A=A, targets=targets, device=device,
        L=[r["L"] for r in rows], t=[r["t"] for r in rows],
        gkt_s=[r["gkt_s"] for r in rows], bf_s=[r["bf_s"] for r in rows],
        fl_gkt=[r["fl_gkt"] for r in rows], fl_bf=[r["fl_bf"] for r in rows],
    )


def make_benchmark_plot(bench, out_dir):
    out = os.path.join(out_dir, "gkt_vs_bruteforce_benchmark.png")
    L = bench["L"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=110)

    # (1) wall-clock, log-y
    ax = axes[0]
    ax.plot(L, [s * 1e3 for s in bench["bf_s"]], "o-", color="#d95f02",
            label="Brute force")
    ax.plot(L, [s * 1e3 for s in bench["gkt_s"]], "o-", color="#2b5c8f", label="GKT")
    ax.set_yscale("log"); ax.set_xticks(L)
    ax.set_xlabel("L (sequence length)"); ax.set_ylabel("wall-clock per target (ms, log)")
    ax.set_title(f"Measured runtime  [{bench.get('device', 'cpu')}]"); ax.legend()

    # (2) analytical op-count, log-y  (the clean scaling story)
    ax = axes[1]
    ax.plot(L, bench["fl_bf"], "s--", color="#d95f02", label="Brute force  ~(1+A)^L")
    ax.plot(L, bench["fl_gkt"], "s--", color="#2b5c8f", label="GKT  ~poly(L)")
    ax.set_yscale("log"); ax.set_xticks(L)
    ax.set_xlabel("L (sequence length)"); ax.set_ylabel("analytical flops (log)")
    ax.set_title("Modeled operation count"); ax.legend()

    # (3) speedup (measured vs modeled), log-y
    ax = axes[2]
    meas = [bench["bf_s"][k] / bench["gkt_s"][k] for k in range(len(L))]
    modeled = [bench["fl_bf"][k] / bench["fl_gkt"][k] for k in range(len(L))]
    ax.plot(L, meas, "o-", color="#1b7837", label="measured (wall-clock)")
    ax.plot(L, modeled, "s--", color="#762a83", label="modeled (flops)")
    ax.set_yscale("log"); ax.set_xticks(L)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("L (sequence length)"); ax.set_ylabel("brute force / GKT  (x, log)")
    ax.set_title("Speedup"); ax.legend()

    plt.suptitle(f"GKT vs brute-force: exponential -> polynomial "
                 f"(A={bench['A']}, targets={bench['targets']}, "
                 f"device={bench.get('device', 'cpu')})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"saved benchmark figure -> {out}")


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--alphabet", type=str, default="ABC")
    ap.add_argument("--ptrain", type=float, default=0.5)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--kernel", choices=["jenga", "connectedness"], default="jenga")
    ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="run the GKT and brute-force paths on CUDA if a GPU is visible "
                         "(falls back to CPU with a warning otherwise). Applies to both "
                         "the verification battery and --benchmark.")
    # Benchmark options
    ap.add_argument("--benchmark", action="store_true",
                    help="run the GKT-vs-brute-force timing/flop sweep over L")
    ap.add_argument("--bench-L-min", type=int, default=2)
    ap.add_argument("--bench-L-max", type=int, default=6,
                    help="ceiling for the sweep; brute force is O((1+A)^L), keep modest")
    ap.add_argument("--bench-targets", type=int, default=3)
    ap.add_argument("--bench-repeats", type=int, default=5)
    args = ap.parse_args()

    # Resolve the compute device once, up front.
    device = resolve_device(args.gpu)
    if device == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("device: cpu")

    # Output directory: tests/output/ (this file lives in tests/).
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    # ----- Benchmark mode: sweep L, time both methods, model op-counts ----- #
    if args.benchmark:
        L_values = list(range(args.bench_L_min, args.bench_L_max + 1))
        # Guard: warn if the brute-force ceiling is going to be painful.
        A = len(set(args.alphabet))
        projected = (1 + A) ** args.bench_L_max
        print(f"Benchmark sweep L={L_values}, A={A}, targets={args.bench_targets}")
        if projected > 5e5:
            print(f"  [warn] brute force at L={args.bench_L_max} scales ~ (1+A)^L "
                  f"= {projected:.1e} evals/index; this may be slow. "
                  f"Lower --bench-L-max if needed.")
        bench = run_benchmark(
            L_values, alphabet=args.alphabet, ptrain=args.ptrain,
            n_iter=args.n_iter, kernel=args.kernel,
            targets=args.bench_targets, seed=args.seed, repeats=args.bench_repeats,
            device=device,
        )
        if not args.no_plot:
            make_benchmark_plot(bench, out_dir)
        print("\n==> BENCHMARK COMPLETE")
        return True

    print(f"Fitting EpiK model: L={args.L}, alphabet='{args.alphabet}', "
          f"kernel={args.kernel} ...")
    ktc = build_real_ktc(L=args.L, alphabet=args.alphabet, ptrain=args.ptrain,
                         n_iter=args.n_iter, seed=args.seed, kernel=args.kernel)

    res = run_checks(ktc, n_targets=args.targets, seed=args.seed, device=device)

    print(f"\nconfig: L={res['L']}, A={res['A']}, t={res['t']}, "
          f"targets={args.targets}, device={res['device']}\n")
    all_pass = True
    for name, ok, detail in res["report"]:
        flag = "PASS" if ok else "FAIL"
        extra = f"   ({detail})" if detail else ""
        print(f"  [{flag}] {name}{extra}")
        all_pass = all_pass and ok
    print(f"\nmax |GKT - brute force| across all targets = {res['max_err']:.3e}")
    if "cov_max_err" in res:
        print(f"max |GKT - brute force| covariance         = {res['cov_max_err']:.3e}")

    if not args.no_plot:
        make_plot(res, out_dir)

    print("\n==>", "ALL CHECKS PASS" if all_pass else "SOME CHECKS FAILED")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)