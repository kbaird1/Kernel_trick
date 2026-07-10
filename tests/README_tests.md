# Kernel-trick test suite

A complete verification + benchmarking setup for `KernelTrickComputer`, split into a
**pytest unit suite** (pass/fail regression guard) and a **standalone benchmark script**
(produces timing + comparison graphs).

## Files

| File | Role |
|---|---|
| `kt_test_utils.py` | Shared foundation: loads both implementations side by side, builds the synthetic + optional real-data model, brute-force / projection / sampling references, the API adapter, and plotting helpers. |
| `conftest.py` | pytest fixtures (model, adapters, tolerances, output dir) and CLI options. Cleanly *skips* when a dependency is missing. |
| `test_correctness.py` | **Unit suite.** Verifies the refactored tool against independent ground truth — brute-force enumeration, the exact projection `P`, and posterior sampling — plus one targeted guard per refactor change. Emits a method-agreement correlation graph. |
| `test_equivalence.py` | **Unit suite.** Original vs refactored, equality-first, across the whole public API. Emits refactored-vs-original parity graphs. |
| `benchmark_compare.py` | **Not a unit test.** Times every operation on both implementations, reports speedups and output differences, writes timing + parity graphs and a CSV/Markdown summary. |
| `pytest.ini` | Minimal pytest config + marker registration. |

## Running the unit suite

```bash
pytest -v                      # everything (auto-discovers both test files)
pytest test_correctness.py -v  # just the ground-truth correctness checks
pytest test_equivalence.py -v  # just original-vs-refactored equality
pytest -m "not keops" -v       # skip the slow KeOps-compilation test
```

Graphs are written to `./output_graphs` by default; change with `--output-dir DIR`.
Disable graph emission from pytest with `--no-save-graphs`. Posterior-sampling resolution
for the covariance reference is `--samples N` (default 20000; raise it to tighten the
sampling correlation, lower it for speed).

Tolerances can be overridden without editing code:

```bash
KTC_ATOL=1e-4 KTC_RTOL=1e-4 KTC_VE_RTOL=1e-2 pytest -v
```

Model fit speed (correctness identities are independent of fit quality, so a short fit is fine):

```bash
KTC_FIT_ITERS=25 pytest -v          # default
KTC_DEVICE=cuda pytest -v           # run on GPU if available
```

### Optional real-data fixture

Off by default. Enable with `--real-data` and point the env vars at your data/checkpoint:

```bash
export KTC_REAL_CSV=/path/to/smn1.csv
export KTC_REAL_ALPHABET=ACGU
export KTC_REAL_L=8
export KTC_REAL_KERNEL=ConnectednessKernel     # or JengaKernel / GeneralProductKernel
export KTC_REAL_PARAMS=/path/to/model_params.pth   # optional (GeneralProduct checkpoints)
export KTC_REAL_ENUMERATE=1                    # only if A**L is small enough to enumerate
pytest test_correctness.py -v --real-data
```

Brute-force identities on real data require `KTC_REAL_ENUMERATE=1` (the full sequence space
must be enumerable); otherwise those real-data checks skip.

---

## What runs under `pytest tests/ -rs -q`

The command runs every `test_*` function under `tests/`, keeps the output compact (`-q`),
and prints the reason for any skipped test (`-rs`). It collects **31 items**: 30 run, and 1
is the optional real-data test that skips unless `--real-data` is passed.

### Shared setup (used by every test)

Defined once in `conftest.py` / `kt_test_utils.py`; built a single time per session:

- **`synth`** — one small synthetic GP: a `JengaKernel` over the *entire* sequence space
  (L=5, alphabet "ABC" → 243 sequences), fit briefly. Because the whole space is
  enumerated, `preds` (the posterior mean over all 243 sequences) is available, which makes
  exact brute-force references possible.
- **`computers`** — instantiates *both* `KernelTrickComputer` classes on that same model and
  wraps each in an adapter (`orig`, `refac`) exposing a uniform interface, so the two
  differing public APIs can be driven identically.
- **Brute-force references** — `compute_seq_probs` (probability of each enumerated sequence
  under a `pi`), `predict_from_truncated` (linear-model predictions from weights), and
  `naive_ve_mse` (variance-explained / MSE by direct enumeration).
- **`projection`** — the repo's `ProjectionMatrix`, providing the exact projection `P` used as
  ground truth for the gauge-fixed mean and covariance.

### `test_correctness.py` — refactored tool vs independent ground truth

**TestExpectationsVsBruteForce**
- **`test_e_map_uniform_and_random`** — checks `E_MAP(pi)` (expected posterior mean under
  distribution `pi`). Uniform `pi` must equal `mean(preds)`; random `pi` must equal
  `sum(preds * seq_probs)` from enumerating all 243 sequences. Tolerance `atol=1e-4`.
- **`test_e_map2_uniform_and_random`** — same for `E_MAP_2` (expected *squared* value):
  uniform → `mean(preds²)`, random → `sum(preds² * seq_probs)`.
- **`test_e_map2_keops_matches_dense`** — runs `E_MAP_2` dense and again through the
  KeOps/LazyTensor path (`keops=True`) and asserts they agree. Marked `keops` — this is the
  slow, first-call kernel-compilation test (most of the total runtime).
- **`test_e_trunc`** — builds weights from `gauge_fixed_posterior` on random custom
  subsequences, derives `w_preds` (truncated-model predictions over the full space), then
  checks `E_TRUNC` = `mean(w_preds)` (uniform) and `sum(w_preds * seq_probs)` (random).
- **`test_e_trunc2`** — same construction; `E_TRUNC_2` at uniform `pi` must equal
  `mean(w_preds²)`.
- **`test_e_map_trunc`** — `E_MAP_TRUNC` (cross term) at uniform `pi` must equal
  `mean(w_preds * preds)`.
- **`test_variance_explained_and_mse_vs_naive`** — computes VE and MSE by brute-force
  enumeration (`naive_ve_mse`) and compares to `variance_explained` / `MSE`. MSE is checked
  tightly; VE uses a looser relative tolerance (`ve_rtol`, default 1e-2) because it is a
  ratio and more numerically sensitive.

**TestPosteriorVsProjectionAndSampling**
- **`test_mean_matches_projection`** — for 3 random use-cases, compares the gauge-fixed mean
  (weights from `gauge_fixed_posterior`) against the exact projection `P @ preds`. Direct
  ground-truth check on the mean.
- **`test_covariance_three_way_agreement`** — for 3 use-cases, computes the covariance three
  ways: kernel-trick (the tool), "thm-6" analytic (`P K Pᵀ − PKS·(PKS)ᵀ`), and empirical
  (covariance of many projected posterior samples). Asserts kernel-trick ≈ thm-6 tightly and
  kernel-trick correlates with sampling at r > 0.98, and saves the pairwise correlation
  scatter PNG. (The two benign Cholesky-jitter warnings originate from the sampling step.)

**TestStructuralInvariants**
- **`test_covariance_symmetric_and_psd`** — a per-`pi` linear covariance must be symmetric
  (`cov ≈ covᵀ`) and positive semi-definite (smallest eigenvalue ≥ a small negative tolerance).
- **`test_batched_equals_per_pi`** — computes weights for a batch of 4 `pi`s in one call, then
  confirms each row matches the corresponding single-`pi` call. Batching must not change results.

**TestRefactorChangeGuards** (one targeted guard per optimisation)
- **`test_change1_encode_cache_is_hit`** — calls `_encode_subseqs` twice on the same list and
  asserts the *same tensor object* is returned, proving the encoding memoisation (Change 1).
- **`test_change1_4_kx_cache_populated_and_cleared`** — clears `_kx_cache`, calls `E_MAP` and
  asserts it is populated (Change 1's KX memoisation), then calls `set_precision` and asserts
  it is emptied (Change 4's invalidation); restores precision afterward.
- **`test_change2_block_diagonal_matches_single_pi`** — builds the across-`pi` block covariance
  for 3 `pi`s and asserts each diagonal block equals the standalone single-`pi` covariance.
  Guards the hoisted `Y`/`M` refactor (Change 2).
- **`test_change3_across_mean_equals_assembled_per_pi`** — asserts the batched across-`pi` mean
  equals concatenating the per-`pi` means computed individually. Guards the
  reshape-instead-of-loop change (Change 3).

**TestRealDataExpectations**
- **`test_e_map_on_real_data`** — the one skipped test. Runs only with `--real-data` and the
  `KTC_REAL_*` env vars; repeats the `E_MAP` brute-force check on your real dataset when the
  full space is enumerable.

### `test_equivalence.py` — original vs refactored, equality-first

**TestWeightsEquivalence** (per-`pi` gauge-fixed posterior)
- **`test_mean_only[linear]` / `[constant]` / `[both]`** — three parametrised runs; each
  asserts original and refactored mean/weights match for that subsequence type.
- **`test_mean_only_custom`** — same equality check with a fixed list of custom subsequences.
- **`test_with_covariance[linear]` / `[both]`** — asserts both the mean *and* the covariance
  match between implementations when `calc_covar=True`.

**TestAcrossPisEquivalence** (the batched across-`pi` path)
- **`test_mean_only[linear]` / `[both]`** — original vs refactored across-`pi` mean.
- **`test_with_covariance[linear]` / `[both]`** — original vs refactored across-`pi` mean and
  full block covariance.

**TestExpectationEquivalence**
- **`test_expectations_and_metrics`** — at a uniform `pi` with shared weights, asserts the two
  implementations agree on `E_MAP`, `E_MAP_2`, `E_TRUNC`, `E_TRUNC_2`, `E_MAP_TRUNC`,
  `variance_explained`, and `MSE`.

**TestMetricsMatrixEquivalence** (guards the subseq-ordering fix)
- **`test_ve_matrix`** — original `compute_ve_matrix` vs refactored
  `compute_metrics_matrix(metric='ve')` over a batch of `pi`s.
- **`test_mse_matrix`** — same for MSE.
- **`test_ve_diag`** — same for the diagonal-only variant. These passing confirms the
  `compute_metrics_matrix` subsequence-ordering fix
  (`subseqs = self.constant_subseq + self.linear_subseqs`) is in place; without it, all three fail.

**TestEmitParityGraphs**
- **`test_emit`** — not an assertion test; recomputes a few representative outputs from both
  implementations and writes refactored-vs-original parity scatter PNGs (mean, per-`pi`
  covariance, across-`pi` covariance). Skips if graph emission is disabled.

### Reading the `-rs` output

`-q` collapses results to a compact string where `.` = pass and `s` = skip. `-rs` then lists
each skipped test with its reason at the end — the mechanism that surfaced the earlier
`No module named 'epik'` diagnostics. A clean run reads `30 passed, 1 skipped`, the skip
being `test_e_map_on_real_data` (real-data fixture not requested).

---

## Running the benchmark

```bash
python benchmark_compare.py --repeats 20 --batch 8 --k 6 --output-dir output_graphs
```

Outputs in `output_graphs/`:
- `benchmark_timing.png` — grouped bar chart (original vs refactored, log scale, speedup labels)
- `bench_parity_<scenario>.png` — one refactored-vs-original parity scatter per operation
- `benchmark_summary.csv` / `benchmark_summary.md` — per-scenario time, speedup, max|Δ|, correlation

Scenarios are labelled with which refactor change they primarily exercise (Change 1 =
encode/KX memoisation, Change 2 = hoisted `Y`/`M`, Change 3 = batched across-pis mean).

### What happens under `python tests/benchmark_compare.py --output-dir output_graphs`

This is a standalone script, **not** part of the pytest suite — no pass/fail, no fixtures. Its
job is to measure, not to assert. Step by step:

1. **Path bootstrap.** Because running the file directly does not load `conftest.py`, the
   script first walks up from `tests/` to find the repo root (the folder containing both
   `epik/` and `modules/`) and puts the root and `modules/` on `sys.path`. This is what makes
   `import epik` resolve; without it the run fails with `No module named 'epik'`.
2. **Load both implementations.** It imports the original and refactored
   `kernel_trick_computer` files side by side, then neutralises `epik_loader.ensure_epik_loaded`
   into a no-op (epik is already importable), mirroring the pytest shim.
3. **Build one synthetic model.** The same `JengaKernel` / L=5 / "ABC" fixture the unit suite
   uses (controllable with `--seq-length`, `--alphabet`, `--device`). Both
   `KernelTrickComputer` instances are constructed on this one model, so their cached
   quantities are identical and any timing/parity difference is attributable to the refactor.
4. **Build inputs.** A batch of `--batch` random `pi`s for the per-`pi` and across-`pi`
   scenarios, a separate batch of `--k` random `pi`s for the metrics-matrix scenarios, and a
   fixed list of custom subsequences.
5. **Run each scenario on both implementations.** The scenarios are:
   - **`weights_both_mean`** — per-`pi` gauge-fixed mean, `subseq_type='both'` (baseline).
   - **`weights_linear_cov`** — per-`pi` mean + covariance, `linear` (baseline covariance path).
   - **`across_linear_mean`** — batched across-`pi` mean, `linear` — exercises **Change 3**.
   - **`across_linear_cov`** — across-`pi` mean + block covariance — exercises **Change 2 + 3**.
   - **`ve_matrix`** — the k×k variance-explained matrix — exercises **Change 1** (the original
     re-encodes on every cell of the loop; the refactor memoises), so this is where the largest
     speedup is expected.
   - **`mse_matrix`** — the k×k MSE matrix — also **Change 1**.
6. **Time each call.** For every scenario and each implementation: run `--warmup` untimed
   calls first (to absorb one-off costs like KeOps kernel compilation and cache priming), then
   `--repeats` timed calls, taking the **median** wall-clock time. On CUDA it synchronises
   around each timed call so GPU time is measured correctly.
7. **Compare outputs.** For each scenario it reduces the result to a single tensor (the
   covariance when a call returns `(mean, cov)`, otherwise the returned tensor) and computes
   **max|Δ|** (largest absolute element-wise difference between original and refactored) and
   the **Pearson r** between them. A near-zero max|Δ| means the refactor preserved the numbers;
   a large one flags a functional difference.
8. **Write artefacts** to `--output-dir`:
   - `benchmark_timing.png` — grouped bar chart, original vs refactored per scenario, log-scaled
     y-axis, each refactored bar annotated with its `Nx` speedup.
   - `bench_parity_<scenario>.png` — a parity scatter (refactored vs original) per scenario,
     annotated with r and max|Δ|; points on the `y = x` line mean the two agree.
   - `benchmark_summary.csv` and `benchmark_summary.md` — one row per scenario with original ms,
     refactored ms, speedup (`orig / refac`, higher is better), max|Δ|, and r.
9. **Print a live table** to stdout as it goes (the same numbers as the CSV), and at the end
   emit a reminder if the `ve_matrix` / `mse_matrix` outputs differ — the signal of the known
   `compute_metrics_matrix` subseq-ordering issue, in case the one-line fix was not applied.

Interpreting the result: **speedup > 1 means the refactor is faster**; the memoisation change
(Change 1) should show the biggest gap on `ve_matrix` / `mse_matrix`, while Changes 2–3 show up
on the `across_*` scenarios. **max|Δ| should be ~0 everywhere** — if a scenario shows a large
max|Δ|, the two implementations disagree there and it should be investigated before trusting the
timing. Useful flags: `--repeats` (more = steadier medians), `--batch` and `--k` (larger =
heavier workloads that widen the speedup gaps), and `--device cuda`.