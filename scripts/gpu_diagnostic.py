#!/usr/bin/env python
"""
gpu_diagnostic.py

Determine whether the GP fit actually runs on the GPU, and if not, where the
tensors are being stranded.

WHY THE MEMORY READOUT IS NOT PROOF
    The fit logged "Mem(alloc/res)=0.00/0.00MB" every iteration, which is what a
    CPU run looks like -- but KeOps allocates OUTSIDE PyTorch's caching
    allocator, so a LazyTensor kernel can legitimately show ~0 allocated while
    the GPU is saturated. This script checks device placement directly instead.

THE LEAD
    analysis.py:437 reads
        y_t = torch.as_tensor(self.y, dtype=torch.float32)
    with no `device=` argument, so it lands on CPU. If the training inputs are
    built the same way, GPyTorch runs everything on CPU no matter what --gpu
    says. Step 3 below reports where each tensor actually ended up.

THE TIMING ARGUMENT
    At n_condition=4000 the fit took 2.63 s/iter. A 4000x4000 Cholesky is
    ~2.1e10 flops and the jenga kernel evaluation ~1.3e9 -- order 10 ms of real
    GPU work. Three orders of magnitude of slack is not framework overhead.
    Step 4 measures what this GPU should do per iteration.

Usage:
    python scripts/gpu_diagnostic.py
    python scripts/gpu_diagnostic.py --n-condition 4000 --kernel jenga
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "modules", REPO_ROOT / "scripts"):
    sys.path.insert(0, str(p))


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-condition", type=int, default=2000)
    ap.add_argument("--kernel", default="jenga")
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--data", default=str(REPO_ROOT / "data/qtls_li_hq/qtls_li_hq.csv"))
    a = ap.parse_args()

    # ---- 1. is CUDA visible at all ---------------------------------------- #
    hr("1. CUDA visibility")
    print(f"  torch            {torch.__version__}")
    print(f"  built for CUDA   {torch.version.cuda}")
    print(f"  cuda.is_available {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n  -> No CUDA. Either no --gres=gpu:1 on the job, or the torch build")
        print("     is CPU-only. Check: conda list | grep -E 'torch|cudatoolkit'")
        print("     A CPU-only torch build is the single most common cause and no")
        print("     amount of walltime fixes it.")
        return 1
    p = torch.cuda.get_device_properties(0)
    print(f"  device           {p.name}  {p.total_memory / 1e9:.1f} GB  "
          f"sm_{p.major}{p.minor}")

    # ---- 2. does KeOps use the GPU ---------------------------------------- #
    hr("2. KeOps GPU path")
    try:
        import pykeops
        print(f"  pykeops {pykeops.__version__}")
        from pykeops.torch import LazyTensor
        x = torch.randn(1000, 3, device="cuda")
        xi, xj = LazyTensor(x[:, None, :]), LazyTensor(x[None, :, :])
        out = ((xi - xj) ** 2).sum(-1).sum(dim=1)
        torch.cuda.synchronize()
        print(f"  LazyTensor reduction ran on {out.device} -- KeOps GPU path OK")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  !! KeOps GPU test failed: {exc}")
        print("     pykeops compiles at runtime and needs the cuda module loaded")
        print("     (activate_env loads cuda/12.9.0 for exactly this reason).")

    # ---- 3. WHERE DOES THE MODEL ACTUALLY LIVE ---------------------------- #
    # This is the decisive test: build the analysis object the same way the fit
    # does, then ask every tensor which device it is on.
    hr("3. device placement inside the fitted model")
    try:
        from analysis import SequenceFunctionAnalysis, build_config_from_args, add_common_args
    except ImportError as exc:                                # noqa: BLE001
        print(f"  !! cannot import analysis: {exc}")
        return 1

    sub = argparse.ArgumentParser()
    add_common_args(sub, data_required=False)
    args = sub.parse_args([
        "--data", a.data, "--kernel", a.kernel, "--gpu",
        "--n-condition", str(a.n_condition), "--n-iter", str(a.n_iter),
        "--out-dir", str(REPO_ROOT / "output/gpu_diagnostic"),
    ])
    cfg = build_config_from_args(args)
    print(f"  cfg.device (if present): {getattr(cfg, 'device', '<no attr>')}")

    t0 = time.time()
    sfa = SequenceFunctionAnalysis(cfg).run_fit_only()
    dt = time.time() - t0
    print(f"  fit of {a.n_iter} iters took {dt:.1f}s "
          f"({dt / max(1, a.n_iter):.2f} s/iter)")

    def _walk(obj, prefix="", depth=0, seen=None, out=None):
        """Recursively find tensors and nn.Modules inside an arbitrary wrapper.

        sfa.model is an EpiK object, not a gpytorch Module, so named_parameters()
        does not exist -- the real GPyTorch model lives somewhere inside it.
        Walk attributes instead of assuming an interface.
        """
        import torch.nn as nn
        if seen is None:
            seen, out = set(), []
        if id(obj) in seen or depth > 3:
            return out
        seen.add(id(obj))
        if torch.is_tensor(obj):
            out.append((prefix, "tensor", tuple(obj.shape), str(obj.device)))
            return out
        if isinstance(obj, nn.Module):
            for n, q in obj.named_parameters(recurse=True):
                out.append((f"{prefix}.{n}", "param", tuple(q.shape), str(q.device)))
            for n, q in obj.named_buffers(recurse=True):
                out.append((f"{prefix}.{n}", "buffer", tuple(q.shape), str(q.device)))
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:                                  # noqa: BLE001
                continue
            if torch.is_tensor(v):
                out.append((f"{prefix}.{name}", "tensor", tuple(v.shape), str(v.device)))
            elif isinstance(v, nn.Module):
                _walk(v, f"{prefix}.{name}", depth + 1, seen, out)
            elif isinstance(v, (tuple, list)) and v and torch.is_tensor(v[0]):
                out.append((f"{prefix}.{name}[0]", "tensor",
                            tuple(v[0].shape), str(v[0].device)))
        return out

    print(f"\n  sfa.model is {type(sfa.model).__name__} "
          f"(not a gpytorch Module -- walking attributes)")
    found = _walk(sfa.model, "model")
    ktc = getattr(getattr(sfa, "shap", None), "ktc", None)
    if ktc is not None:
        found += _walk(ktc, "ktc")

    if not found:
        print("  !! no tensors found by introspection")
    else:
        print(f"  {'path':<46} {'kind':<7} {'device':<8} shape")
        for path, kind, shape, dev in found[:40]:
            print(f"  {path[:46]:<46} {kind:<7} {dev:<8} {shape}")
        if len(found) > 40:
            print(f"  ... and {len(found) - 40} more")

    devs = {}
    for _, kind, _, dev in found:
        devs[dev] = devs.get(dev, 0) + 1
    print(f"\n  device tally: {devs}")
    n_cpu = sum(v for k, v in devs.items() if k.startswith("cpu"))
    n_gpu = sum(v for k, v in devs.items() if k.startswith("cuda"))

    # ---- 3b. THE DECISIVE TEST: is the GPU busy during a fit? ------------- #
    hr("3b. GPU utilization sampled during a live fit")
    print("  Low torch-allocator memory proves nothing: KeOps LazyTensors never")
    print("  materialize the kernel matrix, so 2.10 MB reserved at n=2000 is")
    print("  expected on a working GPU path. Sampling actual utilization does")
    print("  settle it.\n")

    import subprocess, threading
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                u, m = (x.strip() for x in r.stdout.strip().split(",")[:2])
                samples.append((int(u), int(m)))
            except Exception:                                  # noqa: BLE001
                pass
            stop.wait(0.25)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    SequenceFunctionAnalysis(cfg).run_fit_only()
    fit_dt = time.time() - t0
    stop.set()
    th.join(timeout=2)

    if samples:
        util = [u for u, _ in samples]
        mem = [m for _, m in samples]
        busy = sum(1 for u in util if u > 20) / len(util)
        print(f"  {len(samples)} samples over {fit_dt:.1f}s")
        print(f"  GPU util : mean={sum(util)/len(util):.0f}%  max={max(util)}%  "
              f"fraction of samples >20%: {busy:.0%}")
        print(f"  GPU mem  : max={max(mem)} MiB  (nvidia-smi, includes KeOps)")
        print()
        if max(util) < 10:
            print("  >>> VERDICT: GPU essentially idle during the fit -> running on CPU.")
        elif busy > 0.5:
            print("  >>> VERDICT: GPU is doing the work. The cost is ALGORITHMIC, not")
            print("      placement -- no device fix will speed this up.")
        else:
            print("  >>> VERDICT: GPU used in bursts but idle most of the time ->")
            print("      likely per-iteration CPU work or host<->device copies")
            print("      between GPU kernels. Profile the inner loop.")
    else:
        print("  !! nvidia-smi sampling produced nothing")
        print(f"  falling back on placement tally: {n_gpu} gpu / {n_cpu} cpu tensors")

    # ---- 4. what this GPU should manage per iteration --------------------- #
    hr("4. reference: raw GPU throughput at fit scale")
    n = a.n_condition
    A = torch.randn(n, n, device="cuda", dtype=torch.float32)
    A = A @ A.T + n * torch.eye(n, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    torch.linalg.cholesky(A)
    torch.cuda.synchronize()
    chol = time.time() - t0
    per_iter = fit_dt / max(1, a.n_iter)
    print(f"  {n}x{n} Cholesky on GPU: {chol * 1e3:.1f} ms")
    print(f"  observed cost:           {per_iter * 1e3:.0f} ms/iter "
          "(includes setup + held-out eval; read the tqdm rate for the pure "
          "optimizer step)")
    print(f"  ratio:                   {per_iter / max(chol, 1e-6):.0f}x")
    print("\n  A GP iteration is a few Cholesky-equivalents plus kernel evaluation.")
    print("  A ratio in the hundreds or thousands means the linear algebra is not")
    print("  where the time goes -- suspect CPU execution, per-element Python work")
    print("  in the kernel, or repeated host<->device copies.")

    # ---- 5. extrapolate the full-data walltime ---------------------------- #
    hr("5. projected cost at full data (19,833)")
    print("  Measured so far: 0.73 s/iter at n=2000, 2.63 s/iter at n=4000")
    print("  (both from the tqdm rate, i.e. the optimizer step alone).")
    import math as _m
    expo = _m.log(2.63 / 0.73) / _m.log(2.0)
    proj = 2.63 * (19833 / 4000) ** expo
    print(f"  empirical exponent: n^{expo:.2f}  "
          "(between the O(n^2) kernel build and the O(n^3) solves)")
    print(f"  projected: {proj:.0f} s/iter at n=19833")
    for it in (200, 400, 600):
        print(f"    {it:>3} iters -> {proj * it / 3600:.1f} h  "
              f"(set --time to {proj * it / 3600 * 1.4:.0f}h with 40% margin)")
    print("\n  The exponent will drift upward as the Cholesky term takes over, so")
    print("  treat these as lower bounds and confirm with slurm/00_timing_probe.sh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())