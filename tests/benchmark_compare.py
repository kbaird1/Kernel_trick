#!/usr/bin/env python3
"""
Benchmark / comprehensive comparison of the original vs refactored KernelTrickComputer.

This is NOT a unit test. It builds both implementations on one model, runs the full
set of representative operations through each, and reports:
  * correctness    -- max|difference| and Pearson r between the two implementations,
  * performance    -- median wall-clock per operation and the refactored speedup,
and writes graphs + a CSV/Markdown summary to the output directory.

The equality checks reuse the same helpers as the pytest equivalence suite, so the two
never drift. Each scenario is annotated with which refactor change it primarily
exercises.

Usage:
    python benchmark_compare.py [--output-dir output_graphs] [--repeats 20]
                                [--warmup 2] [--batch 8] [--k 6] [--device cpu]

Note: the metrics-matrix scenarios will show a non-trivial difference until the
compute_metrics_matrix ordering fix is applied (see the equivalence suite docstring).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


# ---- sys.path bootstrap (mirrors conftest.py) ------------------------------
# Running this script directly does NOT load conftest.py, so we must put the repo
# root (containing epik/) and modules/ on sys.path ourselves, or `import epik`
# fails with "No module named 'epik'".
def _bootstrap_paths():
    import os
    here = Path(__file__).resolve().parent          # .../tests
    for cand in [here, *here.parents]:
        if (cand / "epik").is_dir() and (cand / "modules").is_dir():
            root = cand
            break
    else:
        root = here.parent
    for p in (str(root / "modules"), str(root), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("KTC_MODULES_DIR", str(root / "modules"))


_bootstrap_paths()


def _sync(device):
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timeit(fn, repeats, warmup, device):
    """Return (median_seconds, last_output)."""
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    _sync(device)
    samples = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), out


def to_parity(result):
    """Reduce an op result to a single tensor for parity comparison."""
    if isinstance(result, tuple):
        return result[-1]          # (mean, cov) -> cov
    return result


def make_scenarios(pis_b, pis_k, subs):
    return [
        ("weights_both_mean",
         lambda a: a.weights(pis_b, 1.0, "both"),
         "baseline gauge-fixed mean"),
        ("weights_linear_cov",
         lambda a: a.weights(pis_b, 1.0, "linear", calc_covar=True),
         "per-pi covariance"),
        ("across_linear_mean",
         lambda a: a.across(pis_b, 1.0, "linear"),
         "Change 3 (batched across-pis mean)"),
        ("across_linear_cov",
         lambda a: a.across(pis_b, 1.0, "linear", calc_covar=True),
         "Change 2+3 (hoisted Y/M; batched mean)"),
        ("ve_matrix",
         lambda a: a.ve_matrix(pis_k),
         "Change 1 (encode + KX memoisation)"),
        ("mse_matrix",
         lambda a: a.mse_matrix(pis_k),
         "Change 1 (encode + KX memoisation)"),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="output_graphs")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--batch", type=int, default=8, help="batch size B for pis")
    p.add_argument("--k", type=int, default=6, help="number of pis for metrics matrices")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-length", type=int, default=5)
    p.add_argument("--alphabet", default="ABC")
    args = p.parse_args(argv)

    # ---- imports (fail loudly; this is a manual tool) --------------------- #
    try:
        import torch  # noqa: F401
        import kt_test_utils as U
    except Exception as e:
        print(f"ERROR: cannot import torch / kt_test_utils: {e}", file=sys.stderr)
        return 1
    try:
        orig_mod, refac_mod = U.load_implementations()
    except Exception as e:
        print(f"ERROR: cannot load implementations: {e}", file=sys.stderr)
        return 1

    # Neutralise epik_loader path-hunting when epik is already importable
    # (mirrors the pytest conftest shim; see README "Known issue" section).
    try:
        import epik  # noqa: F401
        import epik_loader
        epik_loader.ensure_epik_loaded = lambda *a, **k: None
    except Exception:
        pass

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building synthetic model ...")
    try:
        fx = U.build_synthetic_model(seq_length=args.seq_length, alphabet=args.alphabet,
                                     device=args.device)
    except Exception as e:
        print(f"ERROR: could not build model (epik unavailable?): {e}", file=sys.stderr)
        return 1

    orig, refac = U.make_computers(fx, orig_mod, refac_mod)

    pis_b = U.random_pis(fx.seq_length, fx.n_alleles, batch=args.batch, seed=101, device=fx.device)
    pis_k = U.random_pis(fx.seq_length, fx.n_alleles, batch=args.k, seed=202, device=fx.device)
    subs = U.generate_use_case(fx.seq_length, fx.alphabet, num_subseqs=12)[2]

    scenarios = make_scenarios(pis_b, pis_k, subs)

    rows = []
    print(f"\nRunning {len(scenarios)} scenarios "
          f"(repeats={args.repeats}, warmup={args.warmup}, device={fx.device}) ...\n")
    for name, fn, note in scenarios:
        try:
            t_o, out_o = timeit(lambda: fn(orig), args.repeats, args.warmup, fx.device)
            t_r, out_r = timeit(lambda: fn(refac), args.repeats, args.warmup, fx.device)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            rows.append(dict(scenario=name, note=note, orig_ms=float("nan"),
                             refac_ms=float("nan"), speedup=float("nan"),
                             max_abs_diff=float("nan"), pearson_r=float("nan")))
            continue

        po, pr = to_parity(out_o), to_parity(out_r)
        mad = U.max_abs_diff(po, pr)
        r = U.pearson_r(po, pr)
        speedup = (t_o / t_r) if t_r > 0 else float("inf")
        rows.append(dict(scenario=name, note=note,
                         orig_ms=t_o * 1e3, refac_ms=t_r * 1e3, speedup=speedup,
                         max_abs_diff=mad, pearson_r=r))
        print(f"  {name:22s}  orig={t_o*1e3:8.2f} ms  refac={t_r*1e3:8.2f} ms  "
              f"speedup={speedup:5.2f}x  max|Δ|={mad:.2e}  r={r:.5f}")

        # per-scenario parity scatter
        U.save_parity_scatter(po, pr, out_dir / f"bench_parity_{name}.png",
                              title=f"{name}\n({note})")

    _write_summaries(rows, out_dir)
    _plot_timing(rows, out_dir)

    print(f"\nWrote graphs and summaries to: {out_dir}")
    # Surface the known metrics discrepancy without failing the run.
    bad = [row for row in rows if row["scenario"] in ("ve_matrix", "mse_matrix")
           and (row["max_abs_diff"] != row["max_abs_diff"] or row["max_abs_diff"] > 1e-3)]
    if bad:
        print("\nNOTE: metrics matrices differ between implementations. This is the known "
              "compute_metrics_matrix subseq-ordering issue; apply the one-line fix "
              "(subseqs = self.constant_subseq + self.linear_subseqs) and re-run.")
    return 0


def _write_summaries(rows, out_dir: Path):
    import csv
    csv_path = out_dir / "benchmark_summary.csv"
    fields = ["scenario", "note", "orig_ms", "refac_ms", "speedup", "max_abs_diff", "pearson_r"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    md_path = out_dir / "benchmark_summary.md"
    with open(md_path, "w") as f:
        f.write("# KernelTrickComputer benchmark: original vs refactored\n\n")
        f.write("| scenario | exercises | orig (ms) | refac (ms) | speedup | max|Δ| | r |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['scenario']} | {row['note']} | {row['orig_ms']:.2f} | "
                    f"{row['refac_ms']:.2f} | {row['speedup']:.2f}x | "
                    f"{row['max_abs_diff']:.2e} | {row['pearson_r']:.5f} |\n")
        f.write("\n*Speedup = orig / refac (higher is better). max|Δ| and r compare the two "
                "implementations' outputs; a large max|Δ| flags a functional difference.*\n")


def _plot_timing(rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [row for row in rows if row["orig_ms"] == row["orig_ms"]]  # drop NaN
    if not valid:
        return
    names = [row["scenario"] for row in valid]
    o = [row["orig_ms"] for row in valid]
    r = [row["refac_ms"] for row in valid]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4.6))
    ax.bar(x - width / 2, o, width, label="original")
    ax.bar(x + width / 2, r, width, label="refactored")
    ax.set_ylabel("median time (ms)")
    ax.set_yscale("log")
    ax.set_title("Original vs refactored — per-operation timing")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    for xi, row in zip(x, valid):
        sp = row["speedup"]
        if sp == sp:
            ax.annotate(f"{sp:.1f}x", (xi + width / 2, row["refac_ms"]),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_timing.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())