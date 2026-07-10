# kernel_trick

Efficient, exact statistics of **genotype → phenotype maps** learned by Gaussian
processes with product kernels.

This repository implements two related tools for interrogating a fitted
sequence-function map without ever enumerating sequence
space:

- **Kernel trick** — closed-form posterior mean and covariance of any position-factorizable linear statistic of the map.
- **Generalized kernel trick (GKT)** — the same machinery extended from scalar factors to *polynomial* factors, which makes exact **SHAP values** tractable.

---

## Background: GP regression in sequence space

We model a sequence-function map $f : \mathcal{A}_1 \times \cdots \times \mathcal{A}_L \to \mathbb{R}$
as a vector $f \in \mathbb{R}^{A}$ indexed by all $A = \prod_p |\mathcal{A}_p|$
sequences. Under a Gaussian-process prior with covariance kernel $K$ and noise
$\sigma_n^2$, and given training data $(X, y_X)$, the posterior over $f$ is Gaussian
with mean and covariance

$$
f^{\text{MAP}} = K_{*,X}\,\big(K_{X,X} + \sigma_n^2 I\big)^{-1} y_X,
\qquad
R = K_{*,*} - K_{*,X}\big(K_{X,X} + \sigma_n^2 I\big)^{-1} K_{X,*}.
$$

The tools here assume **product kernels**, whose covariance between two sequences
factorizes over positions:

$$
K_{x,y} = \prod_{p=1}^{L} a^{p}_{x_p,\, y_p},
$$

where each $a^{p}$ is a symmetric positive-semidefinite $|\mathcal{A}_p| \times |\mathcal{A}_p|$
site-kernel matrix. In the code these factors are read off a fitted model as
`model.kernel.get_site_kernels()` and stored as a tensor of shape `(L, A, A)`.

---

## 1. The kernel trick

For most maps of interest $f$ cannot be written down explicitly, so instead we
compute **summary statistics**. The kernel trick applies to any linear
transformation $M$ of $f$ whose rows factorize by position:

$$
M_{i,x} = \prod_{p=1}^{L} m^{i,p}_{x_p}.
$$

Because $f$ is Gaussian, $Mf$ is Gaussian too. Writing
$Q = \big(K_{X,X} + \sigma_n^2 I\big)^{-1}$, its posterior is
$Mf \sim \mathcal{N}(\bar{\theta}, R)$ with

$$
\bar{\theta} = M K_{*,X}\, Q y,
\qquad
R = M K M^{\mathsf T} - (M K_{*,X})\big(Q + (Qy)(Qy)^{\mathsf T}\big)(M K_{*,X})^{\mathsf T}.
$$

The trick is that the two expensive pieces never require a sum over sequence space.
Using the product structure of $K$ and the factorization of $M$, each entry is a
**product of $L$ position factors**, where each factor is a single sum over one (or
two) alphabets:

$$
(MK)_{i,y} = \prod_{p=1}^{L} \Bigg( \sum_{c \in \mathcal{A}_p} m^{i,p}_c\, a^{p}_{c,\,y_p} \Bigg),
\qquad
(MKM^{\mathsf T})_{ij} = \prod_{p=1}^{L} \Bigg( \sum_{c,c' \in \mathcal{A}_p} m^{i,p}_c\, m^{j,p}_{c'}\, a^{p}_{c,c'} \Bigg).
$$

This turns an intractable computation into an $O(L\,|\mathcal{A}|^2)$ one and yields
the **exact** posterior distribution of the statistic.


### Using it

The class `KernelTrickComputer`
(`modules/kernel_trick_computer_refactored.py`) wraps a fitted
[EpiK](./epik) GP model. Construct it once, then query statistics. The model must
already have its prediction caches populated (a single `model.predict(...)` call does
this) since the computer reads `covar_cache` and `mean_cache` off the model.

```python
from kernel_trick_computer_refactored import KernelTrickComputer

# model: a fitted EpiK GP exposing model.kernel.get_site_kernels()
ktc = KernelTrickComputer(model, alphabet="ACGU", seq_length=8)

# pis: a background distribution, shape (L, A) or batched (B, L, A), row-stochastic.
# Gauge-fixed posterior of the built-in "linear"/"constant"/"both" statistic bases:
mean          = ktc.gauge_fixed_posterior(pis, eta=1.0, subseq_type="linear")
mean, cov     = ktc.gauge_fixed_posterior(pis, eta=1.0, subseq_type="linear",
                                          calc_covar=True)

# custom M-rows, encoded as extended-alphabet "subseqs" ('-' = averaged/wildcard):
mean = ktc.gauge_fixed_posterior(pis, subseqs=my_subseqs, subseq_type="custom")

# expectations and derived metrics under a background pi:
ktc.E_MAP(pi)                       # E_pi[f]      (posterior mean of the map)
ktc.E_MAP_2(pi)                     # E_pi[f^2]
ktc.variance_explained(pi, weights, subseqs)
ktc.MSE(pi, weights, subseqs)
```

Useful options: `across_pis=True` assembles the full block covariance across a batch
of background distributions; `max_batch` / `max_subseq_batch` chunk large workloads to
bound memory; `set_precision(use_double=True)` runs the assembly in float64. A
KeOps/`LazyTensor` variant (`KernelTrickComputerSingletonKeOps`) trades a slow
first-call compilation for better scaling on large problems.

---

## 2. The generalized kernel trick

The kernel trick above needs each $M$-row to be a *scalar* product over positions. A
single conditional expectation $f_x(S) = \mathbb{E}[f \mid X_S = x_S]$ already has that
form. But many statistics of interest, SHAP values among these, are
**weighted sums over exponentially many** such rows and cannot be enumerated.

The generalized kernel trick removes the scalar restriction. Let $R$ be any *linear
functional of polynomials*, and let each position contribute a small polynomial
$q^{i}_{z_p}$ rather than a scalar, so that the row is a factorizable polynomial
$Q^{i}_z = \prod_p q^{i}_{z_p}$. Then:

$$
(MK)_{i,y} = R\Bigg( \prod_{p} \Big( \sum_{c} q^{i}_c\, a^{p}_{c,\,y_p} \Big) \Bigg),
\qquad
(MKM^{\mathsf T})_{ij} = (R \times R)\Bigg( \prod_{p} \Big( \sum_{c,c'} q^{i}_c \times q^{j}_{c'}\, a^{p}_{c,c'} \Big) \Bigg).
$$

The scalar "product over positions" of the ordinary trick becomes a
**polynomial product over positions**, and a final linear functional $R$ collapses the
resulting polynomial back to a scalar. The covariance path carries *two* bookkeeping
variables, so its per-position factor is bivariate and its collapse is $R \times R$.

These primitives are pure functions in
`modules/generalized_kernel_trick_computer.py` — `g_weights`,
`poly_prod_over_positions` (and its bivariate form), `apply_R`, and `apply_RxR` — and
form the entire conceptual addition over the ordinary trick.

---

## 3. SHAP values

**SHAP** (SHapley Additive exPlanations) attributes a model prediction $f(x)$ to its
individual input features by borrowing the Shapley value from cooperative game
theory: each feature's contribution is its average marginal effect over all orders in
which features could be added. It is the unique attribution satisfying *local
accuracy*, *consistency/monotonicity*, and *missingness*.

In our setting a "feature" is a **sequence position**, and the coalition value is a
conditional expectation under a background distribution $\pi$:

$$
f_x(S) = \mathbb{E}\big[f(X) \mid X_S = x_S\big],
\qquad
\phi_i(f, x) = \sum_{S : i \notin S} \frac{|S|!\,(L - |S| - 1)!}{L!}\,\big(f_x(S \cup i) - f_x(S)\big).
$$

Each $\phi_i$ is the value-add of position $i$, averaged over all subsequences that
agree with the target $x$ on a set $S$ and are drawn from the background $\pi$
elsewhere. The base value is $\phi_0 = f_x(\varnothing) = \mathbb{E}_\pi[f]$, and
local accuracy states $f(x) = \phi_0 + \sum_i \phi_i$.

### Computing SHAP with the generalized kernel trick

The sum over subsets $S$ has $2^{L-1}$ terms which is intractable directly, but exactly the
situation the GKT is for. Introduce a bookkeeping variable $w$ that *counts*
$|S|$. Each position contributes the degree $\le 1$ polynomial

$$
q^{i}_{z_p}(w) =
\begin{cases}
\delta_{x_i = z_i} - \pi^{i}_{z_i} & p = i,\\[2pt]
\delta_{x_p = z_p}\, w + \pi^{p}_{z_p} & p \neq i,
\end{cases}
$$

and the functional $R$ maps $w^{s} \mapsto g(s) = s!\,(L - s - 1)!/L!$. Expanding the
product over positions, the coefficient of $w^s$ collects exactly the size-$s$ subsets,
so a $2^{L-1}$-term sum becomes a degree-$(L-1)$ polynomial evaluated by dynamic
programming. Reducing the **exponential cost into polynomial cost** while still returning
the exact posterior distribution of each $\phi_i$.

`ShapKernelTrickComputer` implements this by *composing* a `KernelTrickComputer`
(everything about the kernel, training data, and posterior assembly is reused
unchanged; only the polynomial layer and the SHAP-specific factors are new):

```python
from kernel_trick_computer_refactored import KernelTrickComputer
from generalized_kernel_trick_computer import ShapKernelTrickComputer

ktc  = KernelTrickComputer(model, alphabet, seq_length)
shap = ShapKernelTrickComputer(ktc=ktc)

x  = "ACGUACGU"                # target sequence to explain
pi = background_distribution   # (L, A), row-stochastic

phi        = shap.shap_values(x, pi)                    # (L,)  first-order SHAP means
phi, cov   = shap.shap_values(x, pi, calc_covar=True)   # means + posterior covariance
phi0       = shap.base_value(pi)                         # E_pi[f]
residual   = shap.local_accuracy_residual(x, pi)         # ~0 when eta == 1

# many explanations against one background:
Phi = shap.shap_values_batch([x1, x2, x3], pi)          # (n_targets, L)
```

The mean and covariance paths are implemented and verified. Interaction SHAP
($\phi_{ij}$) and region SHAP are specified in the source with full recipes but left
as future work.

---

## Testing

Both tools ship with test suites whose defining feature is verification against
**fully independent ground truth**, not just internal consistency. Instructions for
running the tests live in [`tests/README_tests.md`](./tests/README_tests.md); this is
a general overview of what they cover.

**Regular kernel trick.** Two complementary pytest suites plus a benchmark script,
all built on a small, fully-enumerable synthetic model:

- a **correctness suite** checks the tool against three independent references —
  brute-force enumeration over the entire sequence space, the exact projection matrix
  $P$, and the empirical covariance of many posterior samples — along with structural
  invariants (covariance symmetry and positive-semidefiniteness, batching consistency);
- an **equivalence suite** confirms the refactored implementation reproduces a prior
  reference implementation across the whole public API; and
- a **benchmark** times each operation on both implementations and reports speedups
  and output parity.

**Generalized kernel trick / SHAP.** The GKT is verified against a brute-force
reference that enumerates all subsets $S$ directly, with **no polynomial layer**.
Both paths read the same cached kernel/training/posterior quantities but compute by
entirely different routes. The checks cover the SHAP mean and covariance, the
$g$-weight sanity identity $\sum_s \binom{L-1}{s} g(s) = 1$, the local-accuracy
identity, and a scaling benchmark that demonstrates the intended
exponential-to-polynomial speedup as sequence length grows.
