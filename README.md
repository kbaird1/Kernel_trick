# kernel_trick

Exact, tractable statistics of **genotype → phenotype maps** learned by Gaussian
processes with product kernels — without ever enumerating sequence space.

Given a sequence-function model `f` fit by a product-kernel GP, we often want to
interrogate *what it learned*: the expected phenotype under a background, the
contribution of each position, which sets of positions interact, and so on. Every one
of these is a linear functional of `f`, and for a product kernel each such functional
has a closed form that costs `O(poly(L))` instead of `O(A^L)`. This repository provides
that machinery in three layers:

- **Kernel trick** (`KernelTrickComputer`) — exact posterior mean **and** covariance of
  any *position-factorizable* linear statistic of the map.
- **Generalized kernel trick / GKT** (`generalized_kernel_trick`) — the same idea with
  each per-position factor promoted from a scalar to a small *polynomial*, which makes
  statistics that are sums over exponentially many factorizable rows (e.g. SHAP)
  tractable.
- **SHAP** (`ShapKernelTrickComputer`) — the first *consumer* of the GKT: first-order
  Shapley values, higher-order interactions (SII), Harsanyi dividends, and Faith-Shap,
  each with full GP posterior uncertainty.

SHAP is only the first thing built on the framework. The last section shows how to
compute **your own statistic** by supplying a different `M`.

> **EpiK is a dependency, not part of this project.** Fitting the GP (kernels, training)
> is handled by the separate, independently documented `epik` package. This README shows
> only the minimal "fit a model" step the examples need; see the EpiK docs for anything
> about fitting.

---

## Repository layout

The project lives in `modules/`:

| file | role |
|------|------|
| `kernel_trick_computer_refactored.py` | `KernelTrickComputer` — the ordinary kernel trick (mean + covariance of factorizable statistics). Also a memory-lean KeOps variant, `KernelTrickComputerSingletonKeOps`. |
| `generalized_kernel_trick.py` | The GKT **polynomial layer**: model-independent primitives (`g_weights`, `poly_prod_over_positions` (+ bivariate), `apply_R`, `apply_RxR`). |
| `shap_kernel_trick_computer.py` | `ShapKernelTrickComputer` — SHAP/interaction/dividend/Faith-Shap on top of the GKT, plus the `make_background` helper. |
| `shap_brute_force.py` | Independent brute-force reference (direct subset enumeration, no polynomial layer) used to verify every SHAP quantity. |
| `epik_loader.py`, `dataset_loader.py` | Helpers to locate the external EpiK package and load sequence-function datasets. |

`epik/` is the external EpiK package (its own repo/docs). `tests/` has its own
`README_tests.md`; testing is not covered here beyond the verification note at the end.

---

## Background: GPs in sequence space

A sequence-function map `f : A_1 × ... × A_L → ℝ` is treated as a vector `f ∈ ℝ^A`
indexed by all `A = ∏_p |A_p|` sequences. Under a GP prior with covariance kernel `K`
and noise `σ²`, and training data `(X, y)`, the posterior over `f` is Gaussian with

```
f_MAP = K_{*,X} (K_{X,X} + σ² I)^{-1} y,
R     = K_{*,*} - K_{*,X} (K_{X,X} + σ² I)^{-1} K_{X,*}.
```

The tools assume a **product kernel**, whose covariance factorizes over positions,

```
K_{x,y} = ∏_{p=1}^{L} a^p_{x_p, y_p},
```

with each `a^p` a symmetric PSD `|A_p| × |A_p|` site-kernel matrix (read off a fitted
model as `model.kernel.get_site_kernels()`, shape `(L, A, A)`).

---

## 1. The kernel trick — `KernelTrickComputer`

Most maps are too large to write down, so we compute **summary statistics**. The kernel
trick applies to any linear transform `M` of `f` whose rows factorize by position,

```
M[i, x] = ∏_{p=1}^{L} m^{i,p}_{x_p}.
```

Because `f` is Gaussian, `M f` is Gaussian; its posterior mean and covariance need only
`M K_{*,X}` and `M K Mᵀ`, and for a product kernel each of these is a **product of `L`
per-position sums** — no sum over sequence space:

```
(M K)_{i,y}    = ∏_p ( Σ_c  m^{i,p}_c a^p_{c, y_p} ),
(M K Mᵀ)_{i,j} = ∏_p ( Σ_{c,c'} m^{i,p}_c m^{j,p}_{c'} a^p_{c,c'} ).
```

This turns an intractable computation into an `O(L·A²)` one and returns the **exact**
posterior distribution of the statistic.

**Supplying a statistic.** Rows are given as strings over the *extended alphabet*
(`alphabet + "-"`): a real character *fixes* that position, `"-"` *averages* it under a
background `π`. Each string is one factorizable row.

```python
import torch
from kernel_trick_computer_refactored import KernelTrickComputer

ktc = KernelTrickComputer(model, alphabet="ACGU", seq_length=8)   # model: fitted EpiK GP
pi  = torch.full((8, 4), 1.0 / 4)                                  # (L, A), row-stochastic

# posterior mean + covariance of a factorizable statistic
# (fix positions 0 and 3 to 'A','U'; average the other six under pi):
mean, cov = ktc.gauge_fixed_posterior(
    pi, eta=1.0, subseqs=["A--U----"], subseq_type="custom", calc_covar=True)

base = ktc.E_MAP(pi)          # E_pi[f], the background expectation
```

Built-in statistic bases are available via `subseq_type="linear" | "constant" | "both"`,
and there are higher-level metrics (`variance_explained`, `MSE`, `E_MAP`, `E_TRUNC`, …).
Useful options: `across_pis=True` (block covariance across a batch of backgrounds),
`max_batch` / `max_subseq_batch` (memory chunking), and `set_precision(use_double=True)`
(float64 assembly). For very large sequence spaces, `KernelTrickComputerSingletonKeOps`
computes the same quantities lazily via KeOps.

---

## 2. The generalized kernel trick — `generalized_kernel_trick`

The ordinary trick needs each row to be a *scalar* product over positions. A single
conditional expectation `f_x(S) = E[f | X_S = x_S]` has that form — but many statistics
are **weighted sums over exponentially many** such rows and cannot be enumerated.

The GKT removes the scalar restriction. Let each position contribute a small
**polynomial** in a bookkeeping variable `w` rather than a scalar, so the row is a
factorizable polynomial `Q^i_z = ∏_p q^i_{z_p}`, and let `R` be a linear functional on
polynomials. Then

```
(M K)_{i,y}    = R  ( ∏_p ( Σ_c  q^i_c a^p_{c, y_p} ) ),
(M K Mᵀ)_{i,j} = R×R( ∏_p ( Σ_{c,c'} q^i_c × q^j_{c'} a^p_{c,c'} ) ).
```

The scalar "product over positions" becomes a **polynomial product over positions** (a
convolution, evaluated by dynamic programming), and the functional `R` collapses the
resulting coefficient vector back to a scalar — in practice, a dot product with a weight
vector `g(s)`. The covariance path carries two bookkeeping variables and uses `R×R`. An
otherwise exponential sum becomes a degree-`(L-1)` polynomial: **exponential → polynomial**.

The module is the pure, model-independent *evaluation* layer:

- `g_weights(L, k)` — build a weight vector `g_k(s)` (default: the Shapley/SII kernel).
- `poly_prod_over_positions(factors)` (+ `_bivariate`) — the convolution over positions.
- `apply_R(coeffs, g)` / `apply_RxR(coeffs2, g, g2)` — collapse coefficients with `g`.

You rarely call these directly. The *encoding* (what each per-position polynomial factor
is, and which `g` collapses it) is statistic-specific and lives in a **consumer** — of
which the SHAP computer is the reference example.

---

## 3. SHAP values — `ShapKernelTrickComputer` (worked example)

`ShapKernelTrickComputer` *composes* a `KernelTrickComputer` (all kernel/training/
posterior state is read from it) and adds the SHAP-specific per-position factors plus the
polynomial layer. In our setting a "feature" is a sequence position and the coalition
value is the conditional expectation `f_x(S)` under a background `π`.

### Preparing a fitted model (EpiK)

The minimal recipe (mirrors `analysis.py`). EpiK does the fitting; everything after
`predict` is this project.

```python
import sys, torch
sys.path.insert(0, "modules")            # so project modules import by name

import epik_loader
epik_loader.ensure_epik_loaded()         # locate the external EpiK package (see epik_loader.py)

from epik.epik.kernel import JengaKernel
from epik.epik.model  import EpiK
from epik.epik.utils  import encode_seqs
from kernel_trick_computer_refactored import KernelTrickComputer
from shap_kernel_trick_computer import ShapKernelTrickComputer, make_background

# data: equal-length sequences + a scalar phenotype each
seqs = [...]                              # e.g. ["VDGV", "VDGA", ...]
y    = [...]
alphabet = "".join(sorted({c for s in seqs for c in s}))
L, A = len(seqs[0]), len(alphabet)

X   = encode_seqs(seqs, alphabet=alphabet)
y_t = torch.as_tensor(y, dtype=torch.float32)
yv  = torch.zeros_like(y_t)              # per-point measurement variance (0 if unknown)

model = EpiK(JengaKernel(n_alleles=A, seq_length=L))
model.set_data(X, y_t, yv)
model.fit(n_iter=200, learning_rate=0.02)
model.predict(X, calc_variance=False)    # REQUIRED: populates the caches the trick reads

ktc  = KernelTrickComputer(model, alphabet, L)
ktc.set_precision(use_double=True)       # SHAP weights are near-cancelling; use float64
shap = ShapKernelTrickComputer(ktc=ktc)
```

### Computing attributions

```python
x  = "VDGV"                                          # target sequence to explain
pi = make_background("uniform", L, alphabet)         # or 'wildtype' / 'empirical'

# first-order Shapley (mean, and optionally posterior covariance)
phi          = shap.shap_values(x, pi)               # (L,)
phi, phi_cov = shap.shap_values(x, pi, calc_covar=True)
phi0         = shap.base_value(pi)                   # E_pi[f]; local accuracy: f(x) = phi0 + sum phi

# higher-order interactions (Shapley Interaction Index)
T, sii   = shap.shap_interaction_values(x, pi, order=2)          # all pairs
# irreducible epistasis (Harsanyi dividend == full-order Faith-Shap)
T, a     = shap.dividend_values(x, pi, interactions=[(0,3), (1,2,5)])
# efficient explanation up to a maximum order ell (truncated Faith-Shap)
T, fs    = shap.faith_shap_values(x, pi, order_max=3, order=2)

# dataset-level summaries over a sample of targets
idx, imp        = shap.global_shap_importance([x1, x2, x3], pi)        # mean |phi_i|
T, div_summary  = shap.global_dividend_summary([x1, x2, x3], pi)       # mean|a|, Sobol variance
```

### Choosing an interaction index

Every interaction quantity is the **same** position-factorized GKT moment collapsed by a
**different weight vector** — pick by the question you are asking:

| method | question it answers | notes |
|--------|--------------------|-------|
| `shap_values` | how much does each single position contribute to *this* `f(x)`? | first-order Shapley; the unique attribution with local accuracy |
| `shap_interaction_values` | non-additivity, in numbers comparable to the SHAP literature | SII; **smears** a true order-`m` effect onto all its subsets |
| `dividend_values` | which position-*sets* are *irreducibly* epistatic? | Harsanyi dividend = full-order Faith-Shap; **no smearing**; not efficient; `a({i})` is the ANOVA main effect, ≠ `phi_i` |
| `faith_shap_values` | an *efficient* explanation using only orders ≤ `ell` | equals the dividends when the true max order ≤ `ell` (exactly when `ell ≥ L`) |

The background `π` is the reference every attribution is measured against; `make_background`
supports `'uniform'` (product background → dividends carry the functional-ANOVA / Sobol
reading), `'wildtype'` (deviations from a WT sequence), and `'empirical'` (dataset residue
frequencies).

---

## Extending the framework: computing your own statistic (a new `M`)

SHAP is one consumer, not the point. Both tools compute the exact posterior of `M f` for
any linear `M` whose rows factorize over positions,

```
M[row, x] = ∏_p m[row, p, x_p].
```

If your statistic has that form it fits, and one question decides which tool to use:

> **Is your statistic a weighted sum over subsets whose weight depends on the subset
> size (a count)?**
>
> - **No** → your `M` is a factorizable row (or a fixed linear combination of a few).
>   Use the **kernel trick** and hand it the rows.
> - **Yes** → your `M` sums over exponentially many rows with a size-dependent
>   coefficient. Use the **GKT**: per-position polynomial factors + a size-weight `g(s)`.

The reason a count forces the GKT is that a count is the one quantity that does *not*
factor across positions; the bookkeeping polynomial is exactly what tracks it.

### Case 1 — factorizable rows (kernel trick)

Encode each row as an extended-alphabet string (fix a character, or `"-"` to average
under `π`) and read the posterior straight off `gauge_fixed_posterior`. For any linear
functional of several rows, pass them together and combine the returned Gaussian yourself
— weight vector `w` gives mean `w·θ` and variance `wᵀ R w`:

```python
rows = ["A-------", "--------"]                       # two factorizable rows
theta, R = ktc.gauge_fixed_posterior(
    pi, subseqs=rows, subseq_type="custom", calc_covar=True)
w      = torch.tensor([1.0, -1.0], dtype=theta.dtype)
value  = w @ theta                                    # posterior mean of the combination
var    = w @ R @ w                                    # its posterior variance
```

This one interface already covers conditional expectations, gauge-fixed
main/interaction components, truncated-basis projections, and their contrasts — the
building blocks the SHAP and `dividend_values` code is assembled from.

### Case 2 — sums over subsets (GKT)

Two things can vary; change only what you need.

**(a) Same moment, new weight `g(s)`.** If your statistic uses the same per-position
factors as an interaction index but a different size-weight, you only supply a new `g`.
This is the entire difference between the SII and the dividend in `_index_weight_fn`. For
example, the **Banzhaf** index weights every context equally, `g_k(s) = (1/2)^{L-k}`:

```python
def banzhaf_weight(k: int) -> torch.Tensor:
    L = shap.seq_length
    return torch.full((L - k + 1,), 0.5 ** (L - k),
                      device=shap.device, dtype=shap._g_dtype)

x_oh = shap.target_to_onehot(x)
interactions = [(0, 3), (1, 2, 5)]

mK   = shap._shap_z_interaction(x_oh, pi, 1.0, interactions, weight_fn=banzhaf_weight)
vals = mK @ shap.ktc.Qy.to(mK.dtype)                  # posterior means of the new index
# covariance mirrors the public path:
R   = shap._shap_R_interaction(x_oh, pi, 1.0, interactions, weight_fn=banzhaf_weight)
zS  = mK @ shap.ktc.S.to(mK.dtype)
cov = R - zS @ zS.T
```

`_shap_z_interaction` / `_shap_R_interaction` (with `weight_fn`) are the extension
points; to make a new index first-class alongside `"sii"` / `"dividend"`, add one branch
to `_index_weight_fn` returning your `k → g_k` callable.

**(b) New per-position factors.** If the per-position *building blocks* differ from
SHAP's (not just the weight), write a small consumer against the primitives in
`generalized_kernel_trick`. The recipe mirrors `_shap_z_mean` / `_shap_R_cov`:

```
1. Build, per position p, a coefficient vector in w (degree 0 or 1 for a first-order
   statistic) from the kernel channels.
2. mK   = apply_R( poly_prod_over_positions(factors), g )        # per training point
3. mean = mK @ ktc.Qy
4. (covariance) build bivariate blocks, then
   R    = apply_RxR( poly_prod_over_positions_bivariate(blocks), g )
   cov  = R - (mK @ ktc.S) @ (mK @ ktc.S).T
```

`ShapKernelTrickComputer` is the reference consumer; a new statistic of this shape should
live in its own class and reuse the primitives unchanged.

### Verify every new `M` against brute force

The project's correctness standard is agreement with a **fully independent** computation,
not internal consistency. When you add a statistic, add a brute-force reference that
computes it by direct enumeration (over subsets, or over sequence space on a small model)
with no shared machinery, and check the two agree to `~1e-12`. `shap_brute_force.ShapBruteForce`
is the template — it re-derives every SHAP quantity by enumeration and is what the
GKT paths are checked against.

---

## Testing

Both tools ship with test suites verified against fully independent ground truth
(brute-force enumeration, an exact projection matrix, and empirical posterior samples).
See `tests/README_tests.md` for how to run them.