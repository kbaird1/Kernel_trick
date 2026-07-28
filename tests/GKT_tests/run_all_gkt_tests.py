#!/usr/bin/env python
"""
run_all_gkt_tests.py   (module name: run_all_gkt_tests)
=======================================================
One entry point that runs every GKT / SHAP test script and collects a single
pass/fail summary, with all figures landing in ``tests/GKT_tests/output/``.

It drives the five scripts in this directory:
    * test_gkt_brute.py         -- SHAP mean + covariance vs brute force (+ --benchmark)
    * test_gkt_dividends.py     -- Harsanyi dividends vs brute force + truth (+ --benchmark)
    * test_gkt_global.py        -- global importance + Sobol from dividends
    * test_gkt_interactions.py  -- SII recovery: truth vs brute force vs GKT
    * test_gkt_gpu.py           -- CPU-vs-GPU GKT parity/timing (GKT only)

WHY SUBPROCESSES (not imports)
------------------------------
Each test script does its own ``sys.path`` / epik bootstrap at import time and calls
``sys.exit`` in ``__main__``; several also parse ``argparse`` at module load. Importing
them into one process would run those side effects once and fight over global state
(argv, the epik path shim, matplotlib figures). Running each as a separate
``python <script> <args>`` subprocess keeps them isolated, mirrors how you run them by
hand, and lets us capture a clean per-script exit code (0 == all its checks passed).

OUTPUT DIRECTORY
----------------
Every script already writes its PNGs under ``<its dir>/output/...`` (computed from
``os.path.dirname(__file__)``). Because they now live in ``tests/GKT_tests/``, that
resolves to ``tests/GKT_tests/output/`` automatically -- no redirection needed. This
runner just ensures that directory exists up front and prints where things landed:

    tests/GKT_tests/output/gkt_vs_bruteforce.png
    tests/GKT_tests/output/gkt_vs_bruteforce_benchmark.png     (with --benchmark)
    tests/GKT_tests/output/interactions_test/scenario_*.png
    tests/GKT_tests/output/dividends_test/dividends_scenario_*.png
    tests/GKT_tests/output/dividends_test/dividends_benchmark.png   (with --benchmark)
    tests/GKT_tests/output/global_test/global_interpretation.png

USAGE
-----
    python run_all_gkt_tests.py                 # all correctness suites (CPU)
    python run_all_gkt_tests.py --gpu           # pass --gpu through to every script
    python run_all_gkt_tests.py --benchmark     # also run the timing/flop sweeps
    python run_all_gkt_tests.py --only brute dividends
    python run_all_gkt_tests.py --skip gpu      # e.g. no GPU allocation
    python run_all_gkt_tests.py --L 5 --alphabet ACGT --n-iter 200

Common flags (``--L``, ``--alphabet``, ``--n-iter``, ``--seed``, ``--kernel``) are
forwarded only to the scripts that accept them (see ACCEPTS below), so you can tune the
whole battery at once without a script choking on a flag it doesn't define.

EXIT CODE
---------
0 iff every selected script exited 0; otherwise 1. Suitable for CI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from time import perf_counter


HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")


# Which optional flags each script actually defines. A flag is forwarded to a script
# only if listed here, so the runner can accept a superset without breaking any child.
# ("gpu"/"benchmark"/"no_plot" are handled separately below.)
ACCEPTS = {
    "brute":        {"L", "alphabet", "n_iter", "seed", "kernel", "targets"},
    "dividends":    {"L", "alphabet", "n_iter", "seed", "kernel"},
    "global":       {"L", "alphabet", "n_iter", "seed", "kernel"},
    "interactions": {"L", "alphabet", "n_iter", "seed", "kernel"},
    "gpu":          {"L", "alphabet", "n_iter", "seed", "kernel"},
}

SCRIPTS = {
    "brute":        "test_gkt_brute.py",
    "dividends":    "test_gkt_dividends.py",
    "global":       "test_gkt_global.py",
    "interactions": "test_gkt_interactions.py",
    "gpu":          "test_gkt_gpu.py",
}

# Scripts that support --benchmark (a timing/flop sweep instead of correctness checks).
BENCHMARKABLE = {"brute", "dividends"}
# Scripts that accept a --gpu flag. NOTE: test_gkt_gpu.py is EXCLUDED on purpose --
# it runs a CPU-vs-GPU comparison by design and does not define --gpu (it has
# --precision {double,float32,both} instead), so forwarding --gpu makes its argparse
# exit 2. It auto-detects the GPU from the allocation.
GPU_AWARE = {"brute", "dividends", "global", "interactions"}
NOPLOT_AWARE = {"brute", "dividends", "global", "interactions"}   # gpu has no --no-plot

# test_gkt_gpu.py takes --L as a NARGS list ("--L 4 6 8"); the others take a single int.
GPU_L_IS_LIST = True


def build_cmd(key, args, benchmark_mode):
    """Assemble the argv for one child script, forwarding only flags it accepts."""
    script = os.path.join(HERE, SCRIPTS[key])
    cmd = [sys.executable, script]
    acc = ACCEPTS[key]

    # scalar/common flags
    if "L" in acc and args.L is not None:
        if key == "gpu" and GPU_L_IS_LIST:
            cmd += ["--L", str(args.L)]           # single value is a valid 1-elt nargs list
        else:
            cmd += ["--L", str(args.L)]
    if "alphabet" in acc and args.alphabet is not None:
        cmd += ["--alphabet", args.alphabet]
    if "n_iter" in acc and args.n_iter is not None:
        cmd += ["--n-iter", str(args.n_iter)]
    if "seed" in acc and args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if "kernel" in acc and args.kernel is not None:
        cmd += ["--kernel", args.kernel]

    # gpu / plotting / benchmark
    if args.gpu and key in GPU_AWARE:
        cmd += ["--gpu"]
    if args.no_plot and key in NOPLOT_AWARE:
        cmd += ["--no-plot"]
    if benchmark_mode and key in BENCHMARKABLE:
        cmd += ["--benchmark"]

    return cmd


def run_one(key, args, benchmark_mode):
    cmd = build_cmd(key, args, benchmark_mode)
    label = f"{SCRIPTS[key]}" + ("  [--benchmark]" if (benchmark_mode and key in BENCHMARKABLE) else "")
    print("\n" + "#" * 78)
    print(f"# RUNNING: {label}")
    print("#   " + " ".join(cmd))
    print("#" * 78, flush=True)

    t0 = perf_counter()
    # Inherit stdout/stderr so each script's own report streams live.
    proc = subprocess.run(cmd, cwd=HERE)
    dt = perf_counter() - t0

    ok = (proc.returncode == 0)
    print(f"\n# ---> {SCRIPTS[key]} exited {proc.returncode} "
          f"({'PASS' if ok else 'FAIL'}) in {dt:.1f}s")
    return ok, dt, proc.returncode


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SCRIPTS),
                    help="run ONLY these suites (default: all)")
    ap.add_argument("--skip", nargs="+", choices=list(SCRIPTS), default=[],
                    help="skip these suites (e.g. --skip gpu when no GPU allocation)")
    ap.add_argument("--benchmark", action="store_true",
                    help="ALSO run the timing/flop sweeps for the benchmarkable suites "
                         f"({sorted(BENCHMARKABLE)}) after their correctness runs")
    ap.add_argument("--benchmark-only", action="store_true",
                    help="run ONLY the benchmark sweeps (skip correctness runs)")
    ap.add_argument("--gpu", action="store_true",
                    help="forward --gpu to every GPU-aware script")
    ap.add_argument("--no-plot", action="store_true",
                    help="forward --no-plot (skip figure writing) where supported")
    # forwarded knobs (only sent to scripts that accept them; see ACCEPTS)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--alphabet", type=str, default=None)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--kernel", choices=["jenga", "connectedness"], default=None)
    args = ap.parse_args()

    selected = list(args.only) if args.only else list(SCRIPTS)
    selected = [k for k in selected if k not in set(args.skip)]
    # Stable, sensible order: correctness suites first, gpu last.
    order = ["brute", "dividends", "global", "interactions", "gpu"]
    selected = [k for k in order if k in selected]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("GKT / SHAP TEST BATTERY")
    print("=" * 78)
    print(f"python      : {sys.executable}")
    print(f"suites      : {selected}")
    print(f"gpu         : {args.gpu}")
    print(f"benchmark   : {args.benchmark or args.benchmark_only}"
          + ("  (benchmark-only)" if args.benchmark_only else ""))
    print(f"output dir  : {OUTPUT_DIR}")

    results = []   # (label, ok, seconds, rc)

    for key in selected:
        # correctness run (unless benchmark-only)
        if not args.benchmark_only:
            ok, dt, rc = run_one(key, args, benchmark_mode=False)
            results.append((SCRIPTS[key], ok, dt, rc))
        # benchmark run (if requested and supported)
        if (args.benchmark or args.benchmark_only) and key in BENCHMARKABLE:
            ok, dt, rc = run_one(key, args, benchmark_mode=True)
            results.append((SCRIPTS[key] + " [--benchmark]", ok, dt, rc))

    # ---- summary ---------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    all_ok = True
    total = 0.0
    for label, ok, dt, rc in results:
        total += dt
        flag = "PASS" if ok else f"FAIL (exit {rc})"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<34} {dt:7.1f}s   {flag}")
        all_ok = all_ok and ok
    print("-" * 78)
    print(f"  {'ALL SUITES PASSED' if all_ok else 'SOME SUITES FAILED'}"
          f"   ({len(results)} runs, {total:.1f}s total)")
    print(f"  figures under: {OUTPUT_DIR}")
    print("=" * 78)

    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)