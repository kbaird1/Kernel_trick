#!/usr/bin/env python
"""
test_gkt_gpu.py  --  CPU-vs-GPU comparison for the GKT SHAP path (GKT only).

Does NOT run brute force (that's test_gkt_brute.py's job). This file answers:

    1. Can the GKT SHAP MEAN and COVARIANCE run on a GPU?
    2. Do the GPU results agree numerically with the CPU results?
    3. How does GPU-GKT wall-clock compare to CPU-GKT as L / A scale?

It reuses ShapKernelTrickComputer + onehot from test_gkt_brute.py so we time the
real code. It provides its OWN model builder so it can reach large L.

TWO BUILDERS
------------
* build_real_ktc (imported): enumerates the full A**L sequence space. Exact, but only
  usable for small L (A**L must fit in memory). Fine for A=3, L<=~10.
* build_sampled_ktc (here, used with --sampled): SAMPLES n_train random sequences and
  never enumerates A**L. This is what makes L=15-20, A=4 tractable. The GP fit is
  intentionally cheap/bad (low --n-iter) -- we only need a VALID ktc whose cached
  tensors are populated; fit QUALITY is irrelevant to whether the GKT runs.

WHY LARGE L IS TRACTABLE FOR THE GKT
------------------------------------
The generalized kernel trick computes the exact SHAP mean/covariance WITHOUT
enumerating the A**L sequence space -- it replaces the exponential sum with
polynomial products over the L positions. Its cost scales with the number of
training points t and with L (mean ~O(t*L*A), covariance ~O(t*L^2*A)), NOT with A**L.
So the ONLY thing that blows up at large L is building/fitting the model over the
full space -- which build_sampled_ktc avoids by sampling. The GKT tensors themselves
stay small (e.g. site_kernels (L,A,A), _train_x (t,L,A), Qy (t,), S (t,t)).

MEAN vs COVARIANCE
------------------
Covariance is NOT a separate method: shap_values(x, pi, calc_covar=True) returns
(mean, cov). We time the mean-only call and the mean+cov call separately so you can
see the incremental cost of the covariance (bivariate) path.

PRECISION TRAP
--------------
shap_values(..., use_double=True) internally calls ktc.set_precision(use_double=True),
which recasts caches and forces float64. To time float32 HONESTLY we must pass
use_double=False into the call itself -- setting precision only on the ktc is not
enough because the method would flip it back. This file passes use_double per call.

DEVICE / TIMING NOTES
---------------------
* ShapKernelTrickComputer captures its device at __init__ (self.device = ktc.device,
  self.g built there). So set the device on the ktc and move its cached tensors
  BEFORE constructing the SHAP computer.
* Some GKT paths cast cached tensors with .to(dtype) only (no device move); if left on
  CPU while self.device == cuda you get a device-mismatch. So move EVERY tensor on ktc.
* CUDA kernels are async: perf_counter() alone measures launch time. We synchronize.

USAGE
-----
    # inside a GPU allocation (srun ... --gres=gpu:1 --pty bash), env activated:
    python test_gkt_gpu.py --sampled --alphabet ACGT --L 15 18 20 --precision both
    python test_gkt_gpu.py --sampled --alphabet ACGT --L 20 --n-train 800 --targets 5
    python test_gkt_gpu.py --L 4 6 8            # small: exact full-space builder (default)

Corroborate GPU utilisation in the SAME allocation shell:
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1 > /tmp/gpu.log &
    python test_gkt_gpu.py --sampled --alphabet ACGT --L 18 ; kill %1 ; cat /tmp/gpu.log
"""

import argparse
import itertools
from time import perf_counter

import numpy as np
import torch

# Reuse the SHAP computer + helpers + the full-space builder from the benchmark test.
# Importing this module also runs its top-level sys.path setup, which makes
# epik_loader / the modules package importable -- so this file works from tests/.
from test_gkt_brute import build_real_ktc, ShapKernelTrickComputer, onehot


# --------------------------------------------------------------------------- #
# Sampled builder: fit on RANDOM sequences, never enumerate A**L
# --------------------------------------------------------------------------- #
def build_sampled_ktc(L, alphabet="ACGT", n_train=500, n_query=8,
                      n_iter=5, learning_rate=0.1, seed=0, kernel="jenga"):
    """Fit an EpiK model on n_train RANDOMLY SAMPLED sequences (no A**L enumeration).

    Returns a real KernelTrickComputer, identical in type to build_real_ktc's output,
    but reachable at L=15-20, A=4 because it never materializes the full space. The
    fit is deliberately cheap; only the populated caches matter for the GKT.
    """
    # Imported lazily so the module still imports on machines without epik present.
    from epik.epik.utils import encode_seqs
    from epik.epik.kernel import JengaKernel, ConnectednessKernel
    from epik.epik.model import EpiK
    from modules.kernel_trick_computer_refactored import KernelTrickComputer

    torch.manual_seed(seed)
    np.random.seed(seed)
    A = len(set(alphabet))
    syms = sorted(set(alphabet))

    # Sample n_train + n_query random sequences as (N, L) allele-index rows, then
    # render to strings for encode_seqs. No full-space product is ever formed.
    total = n_train + n_query
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, A, (total, L), generator=gen)          # (total, L)
    seqs = ["".join(syms[i] for i in row.tolist()) for row in idx]

    X = encode_seqs(seqs, alphabet=alphabet)                      # (total, L*A) one-hot
    effects = torch.rand(L * A)                                   # additive ground truth
    y = X @ effects
    y_var = torch.rand(total)

    train_x, train_y = X[:n_train], y[:n_train]
    train_y_var = y_var[:n_train]
    query_x = X[n_train:n_train + n_query]                        # tiny -> cheap variance cache

    kernel_cls = {"jenga": JengaKernel, "connectedness": ConnectednessKernel}[kernel]
    model = EpiK(kernel_cls(n_alleles=A, seq_length=L), track_progress=True)
    model.set_data(train_x, train_y, train_y_var)
    model.fit(n_iter=n_iter, learning_rate=learning_rate)
    # Predict on the SMALL query set only -> avoids the huge test x test covariance
    # that blew up (4.2 TB) when the full space was used as the test set.
    model.predict(query_x, calc_variance=True)
    return KernelTrickComputer(model, alphabet, L)


# --------------------------------------------------------------------------- #
# Device placement
# --------------------------------------------------------------------------- #
def move_ktc_to_device(ktc, device):
    """Move a KernelTrickComputer's model + all cached tensors onto `device`.

    Call BEFORE constructing ShapKernelTrickComputer (it reads ktc.device at init).
    Returns the list of moved tensor-attribute names.
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


def _first_tensor(obj):
    """First tensor from a tensor or a (mean, cov)-style tuple/list."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for e in obj:
            if torch.is_tensor(e):
                return e
    return None


# --------------------------------------------------------------------------- #
# Timing (CUDA-safe)
# --------------------------------------------------------------------------- #
def time_call(fn, device, repeats=5, warmup=2):
    """Best-of-`repeats` wall-clock seconds; synchronise if on CUDA."""
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
            torch.cuda.synchronize()  # wait for kernels to actually finish
        best = min(best, perf_counter() - t0)
    return best


# --------------------------------------------------------------------------- #
# Per-L comparison  (GKT only: CPU vs GPU, mean AND covariance)
# --------------------------------------------------------------------------- #
def build_ktc(sampled, L, alphabet, ptrain, n_train, n_iter, seed, kernel):
    if sampled:
        return build_sampled_ktc(L=L, alphabet=alphabet, n_train=n_train,
                                 n_iter=n_iter, seed=seed, kernel=kernel)
    return build_real_ktc(L=L, alphabet=alphabet, ptrain=ptrain,
                          n_iter=n_iter, seed=seed, kernel=kernel)


def compare_one_L(L, alphabet, sampled, ptrain, n_train, n_iter, kernel, seed,
                  targets, precisions, repeats, have_cuda):
    A = len(set(alphabet))
    ktc = build_ktc(sampled, L, alphabet, ptrain, n_train, n_iter, seed, kernel)

    # t = number of training points actually held by the fitted model (drives GKT cost)
    try:
        t = ktc._train_x.reshape(-1, L, A).shape[0]
    except Exception:  # noqa: BLE001
        t = -1

    # Build query targets (one-hot (L, A)) and a pi background, on CPU first.
    gen = torch.Generator().manual_seed(seed + 1)
    pi_cpu = torch.rand(L, A, generator=gen, dtype=torch.float64)
    pi_cpu = pi_cpu / pi_cpu.sum(dim=1, keepdim=True)
    # random target sequences (index rows -> one-hot); never enumerates A**L
    tgt_idx = torch.randint(0, A, (targets, L), generator=gen)
    xs_cpu = [onehot(row, A) for row in tgt_idx]

    rows = []
    for prec in precisions:
        use_double = (prec == "double")

        # ---- CPU ----------------------------------------------------------- #
        shap_cpu = ShapKernelTrickComputer(ktc=ktc)  # device == cpu here

        def cpu_mean(x):
            return shap_cpu.shap_values(x, pi_cpu, calc_covar=False, use_double=use_double)

        def cpu_meancov(x):
            return shap_cpu.shap_values(x, pi_cpu, calc_covar=True, use_double=use_double)

        cpu_mean_s = min(time_call(lambda x=x: cpu_mean(x), "cpu", repeats) for x in xs_cpu)
        cpu_cov_s = min(time_call(lambda x=x: cpu_meancov(x), "cpu", repeats) for x in xs_cpu)

        # reference values for correctness (first target)
        ref_mean = _first_tensor(cpu_mean(xs_cpu[0]))
        ref_m, ref_c = cpu_meancov(xs_cpu[0])
        ref_mean = ref_mean.detach().cpu() if ref_mean is not None else None
        ref_c = ref_c.detach().cpu() if torch.is_tensor(ref_c) else None

        gpu_mean_s = gpu_cov_s = None
        err_mean = err_cov = None
        gpu_mem_mb = None
        on_gpu = None

        # ---- GPU ----------------------------------------------------------- #
        if have_cuda:
            torch.cuda.reset_peak_memory_stats()
            move_ktc_to_device(ktc, "cuda")
            shap_gpu = ShapKernelTrickComputer(ktc=ktc)  # device == cuda now

            pi_gpu = pi_cpu.to("cuda")
            xs_gpu = [x.to("cuda") for x in xs_cpu]

            def gpu_mean(x):
                return shap_gpu.shap_values(x, pi_gpu, calc_covar=False, use_double=use_double)

            def gpu_meancov(x):
                return shap_gpu.shap_values(x, pi_gpu, calc_covar=True, use_double=use_double)

            # verify output really lands on GPU
            probe_m = _first_tensor(gpu_mean(xs_gpu[0]))
            on_gpu = bool(probe_m is not None and probe_m.is_cuda)

            gpu_mean_s = min(time_call(lambda x=x: gpu_mean(x), "cuda", repeats) for x in xs_gpu)
            gpu_cov_s = min(time_call(lambda x=x: gpu_meancov(x), "cuda", repeats) for x in xs_gpu)

            # correctness: GPU vs CPU on first target
            gm, gc = gpu_meancov(xs_gpu[0])
            if ref_mean is not None and torch.is_tensor(gm):
                err_mean = (gm.detach().cpu() - ref_mean).abs().max().item()
            if ref_c is not None and torch.is_tensor(gc):
                err_cov = (gc.detach().cpu() - ref_c).abs().max().item()

            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            move_ktc_to_device(ktc, "cpu")  # reset for next precision's CPU baseline

        rows.append(dict(
            L=L, A=A, t=t, prec=prec,
            cpu_mean=cpu_mean_s, gpu_mean=gpu_mean_s,
            cpu_cov=cpu_cov_s, gpu_cov=gpu_cov_s,
            err_mean=err_mean, err_cov=err_cov,
            on_gpu=on_gpu, mem=gpu_mem_mb,
        ))

    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, nargs="+", default=[4, 6, 8],
                    help="sequence lengths to sweep")
    ap.add_argument("--alphabet", default="ABC",
                    help="symbol set for synthetic sequences (default 'ABC'; use 'ACGT' for DNA)")
    ap.add_argument("--sampled", action="store_true",
                    help="SAMPLE n_train sequences instead of enumerating A**L "
                         "(required for L greater than ~10-12)")
    ap.add_argument("--n-train", type=int, default=500,
                    help="number of sampled training sequences (only with --sampled)")
    ap.add_argument("--ptrain", type=float, default=0.5,
                    help="train fraction for the full-space builder (ignored with --sampled)")
    ap.add_argument("--n-iter", type=int, default=5,
                    help="GP fit iterations (kept low: fit quality is irrelevant to the GKT)")
    ap.add_argument("--kernel", default="jenga", choices=["jenga", "connectedness"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", type=int, default=3,
                    help="number of target sequences to explain (timing is min over them)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--precision", default="double",
                    choices=["double", "float32", "both"],
                    help="float32 is where the GPU is most likely to help")
    args = ap.parse_args()

    A = len(set(args.alphabet))
    have_cuda = torch.cuda.is_available()

    print("=" * 92)
    print("GKT CPU-vs-GPU COMPARISON  (GKT only: MEAN and COVARIANCE, no brute force)")
    print("=" * 92)
    print(f"torch: {torch.__version__}   cuda available: {have_cuda}")
    if have_cuda:
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("NO GPU VISIBLE -- are you inside a GPU allocation "
              "(srun ... --gres=gpu:1)? Running CPU-only.")
    print()
    print("SYNTHETIC DATA")
    print(f"  alphabet          : '{args.alphabet}'  (unique symbols: {sorted(set(args.alphabet))})")
    print(f"  A (alphabet size) : {A}")
    builder = "sampled (no A**L enumeration)" if args.sampled else "full-space enumeration"
    print(f"  builder           : {builder}")
    if args.sampled:
        print(f"  n_train (sampled) : {args.n_train}")
    else:
        print(f"  sequence space    : A**L = {A}**L enumerated per L "
              f"(only feasible for small L)")
    print(f"  kernel            : {args.kernel}   fit iters: {args.n_iter}")
    print()
    precisions = ["double", "float32"] if args.precision == "both" else [args.precision]
    print(f"precision(s): {precisions}   L: {args.L}   targets: {args.targets}   "
          f"repeats: {args.repeats}")
    print()

    header = (f"{'L':>3} {'A':>2} {'t':>5} {'prec':>7} "
              f"{'CPU mean':>9} {'GPU mean':>9} {'mean sp':>8} "
              f"{'CPU cov':>9} {'GPU cov':>9} {'cov sp':>8} "
              f"{'onGPU':>6} {'err_mean':>10} {'err_cov':>10} {'GPUmem MB':>10}")
    print(header + "    (times in ms)")
    print("-" * len(header))

    all_rows = []
    for L in args.L:
        rows = compare_one_L(
            L, args.alphabet, args.sampled, args.ptrain, args.n_train,
            args.n_iter, args.kernel, args.seed, args.targets,
            precisions, args.repeats, have_cuda,
        )
        for r in rows:
            def ms(x):
                return f"{x*1e3:.3f}" if x is not None else "-"

            def sp(cpu, gpu):
                return f"{cpu/gpu:.2f}x" if (cpu and gpu) else "-"

            onp = "-" if r["on_gpu"] is None else ("yes" if r["on_gpu"] else "NO")
            em = f"{r['err_mean']:.2e}" if r["err_mean"] is not None else "-"
            ec = f"{r['err_cov']:.2e}" if r["err_cov"] is not None else "-"
            mem = f"{r['mem']:.1f}" if r["mem"] is not None else "-"
            print(f"{r['L']:>3} {r['A']:>2} {r['t']:>5} {r['prec']:>7} "
                  f"{ms(r['cpu_mean']):>9} {ms(r['gpu_mean']):>9} {sp(r['cpu_mean'], r['gpu_mean']):>8} "
                  f"{ms(r['cpu_cov']):>9} {ms(r['gpu_cov']):>9} {sp(r['cpu_cov'], r['gpu_cov']):>8} "
                  f"{onp:>6} {em:>10} {ec:>10} {mem:>10}")
        all_rows.extend(rows)

    # ---- verdicts ---------------------------------------------------------- #
    print()
    print("[INFO] 'mean' times the calc_covar=False call; 'cov' times the "
          "calc_covar=True call (mean+covariance together).")
    if have_cuda:
        if any(r["on_gpu"] is False for r in all_rows):
            print("[WARN] Some runs returned CPU tensors -- GPU not actually used there.")
        else:
            print("[PASS] All GKT mean/covariance computations ran on the GPU.")

        errs = [e for r in all_rows for e in (r["err_mean"], r["err_cov"]) if e is not None]
        if errs and max(errs) < 1e-4:
            print(f"[PASS] GPU matches CPU for BOTH mean and covariance "
                  f"(max abs err = {max(errs):.2e}).")
        elif errs:
            print(f"[WARN] GPU vs CPU max abs err = {max(errs):.2e} -- larger than expected.")

        def best_sp(key_cpu, key_gpu):
            vals = [r[key_cpu] / r[key_gpu] for r in all_rows
                    if r[key_gpu] and r[key_cpu]]
            return max(vals) if vals else None

        bm, bc = best_sp("cpu_mean", "gpu_mean"), best_sp("cpu_cov", "gpu_cov")
        if bm:
            print(f"[INFO] Best GPU speedup -- mean: {bm:.2f}x   covariance: "
                  f"{bc:.2f}x" if bc else f"[INFO] Best mean speedup: {bm:.2f}x")
        if (bm and bm <= 1.0) and (bc and bc <= 1.0):
            print("[INFO] No GPU speedup at these sizes -- expected when t and L are "
                  "small. The point here is that mean AND covariance stay TRACTABLE "
                  "at large L/A; raise --n-train and L to grow the GPU advantage.")
    else:
        print("[INFO] Ran CPU-only. Re-run inside a GPU allocation to exercise the GPU.")

    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)