"""
shap_kernel_trick_computer.py
=============================

Compute **SHAP values** (with their full posterior uncertainty) for a
sequence-function map learned by a factorized-kernel Gaussian process, using the
*generalized kernel trick* (GKT).

This module is the SHAP consumer of the reusable GKT polynomial layer in
``generalized_kernel_trick.py`` (which holds the model-independent primitives
``g_weights``, ``poly_prod_over_positions`` (+ bivariate), ``apply_R`` / ``apply_RxR``).
It is the SHAP counterpart to ``kernel_trick_computer_refactored.py`` and is used the
same way on genotype->phenotype data: build a computer from a fitted EpiK GP model, then
ask it for the SHAP attribution of a target sequence against a background distribution
``pi``.

    from kernel_trick_computer_refactored import KernelTrickComputer
    from shap_kernel_trick_computer import ShapKernelTrickComputer

    ktc  = KernelTrickComputer(model, alphabet, seq_length)
    shap = ShapKernelTrickComputer(ktc=ktc)

    x   = "ABACA"                    # target sequence to explain
    pi  = background_distribution    # (L, A), row-stochastic
    phi        = shap.shap_values(x, pi)                    # (L,)  first-order means
    phi, cov   = shap.shap_values(x, pi, calc_covar=True)   # means + posterior covariance
    base       = shap.base_value(pi)                        # phi_0 = E_pi[f]


How it works
------------
A first-order SHAP value is a weighted sum over all ``2^{L-1}`` subsets of the other
positions,

    phi_i(f, x) = sum_{S : i not in S} g(|S|) ( f_x(S u i) - f_x(S) ),
    g(s)        = s! (L - s - 1)! / L!,

where ``f_x(S) = E[ f(X) | X_S = x_S ]`` fixes the target's characters on ``S`` and
averages under ``pi`` elsewhere. Each ``f_x(S)`` is already a position-factorizable
statistic the ordinary kernel trick can evaluate -- but there are exponentially many
of them.

The GKT makes the sum tractable with a bookkeeping variable ``w`` that *counts*
``|S|``. Each position contributes a small polynomial in ``w`` instead of a scalar,

    q^i_{z_p}(w) = { delta_{x_i = z_i} - pi^i_{z_i}       if p == i   (degree 0)
                   { delta_{x_p = z_p} * w + pi^p_{z_p}   if p != i   (degree 1),

and a linear functional ``R`` collapses the resulting polynomial by mapping
``w^s -> g(s)``. Expanding the product over positions, the coefficient of ``w^s``
collects exactly the size-``s`` subsets, so a ``2^{L-1}``-term sum becomes a
degree-``(L-1)`` polynomial evaluated by dynamic programming (repeated convolution
across positions). The posterior covariance uses the same idea with two bookkeeping
variables and the functional ``R x R``.


Module layout
-------------
The GKT "polynomial layer" -- the standalone, model-independent primitives (``g_weights``,
``poly_prod_over_positions`` (+ bivariate), ``apply_R`` / ``apply_RxR``) -- now lives in
its own module, ``generalized_kernel_trick.py``, and is imported below. That layer is the
entire conceptual addition over the ordinary kernel trick, and can be unit-tested in
isolation. This module holds ``ShapKernelTrickComputer``, which *composes* a
``KernelTrickComputer`` (all kernel / training / posterior state is read from it) and adds
the SHAP-specific factor builders and the public API (first-order values, and pairwise /
higher-order interaction values).

The independent brute-force reference used to verify these paths lives in its own module,
``shap_brute_force.ShapBruteForce``. It recomputes every SHAP quantity -- first-order and
interaction -- by direct subset enumeration, without the polynomial layer, so agreement is
genuine evidence of correctness. Build it from the same ``ktc`` (or straight from a
``ShapKernelTrickComputer``)::

    from shap_brute_force import ShapBruteForce
    bf = ShapBruteForce(shap=shap)      # or ShapBruteForce(ktc=ktc)
    torch.allclose(shap.shap_values(x, pi), bf.bf_shap_values(x_oh, pi))

The first-order and interaction mean/covariance paths are implemented and verified
against that reference (see ``test_gkt_brute.py``).


Notes on conventions
--------------------
* ML notion (default): one feature per position, background-averaged under ``pi``.
  The "econ notion" (background concentrated on the wild type; features = non-WT
  positions) is reachable by passing a WT-concentrated ``pi`` and restricting
  ``indices`` -- a convenience switch for it is still to be added.
* ``eta`` scales the background-averaged ("pi") channel only; the fixed channel
  ``a^p_{x_p, .}`` is unscaled. Standard SHAP semantics and the local-accuracy
  identity assume ``eta == 1``; ``eta != 1`` is an experimental knob inherited from
  the kernel-trick code and is not expected to telescope.


Still to do
-----------
* KeOps / LazyTensor variant: this module is dense/torch only. A lazy path mirroring
  ``compute_LTDR_lazy`` in the kernel-trick module is still to be written for scaling
  to large sequence spaces.
* ``shap_region``: region SHAP against a product distribution -- specified, not yet
  implemented.
* An ML/econ-notion convenience switch (see conventions above).

Implemented since v1
--------------------
* ``shap_interaction_values`` / ``shap_pairwise_values``: pairwise ``phi_ij`` and
  higher-order interaction SHAP for arbitrary position tuples, via the same polynomial
  layer (every position in the interaction set ``T`` takes the degree-0 difference
  factor). The interaction index is the Shapley Interaction Index (SII); at ``k == 2``
  with the SHAP normalization this reproduces the ``phi_ij`` of the notes (Section 3.4)
  and Lundberg et al. See ``shap_interaction_values`` for the weight and normalization
  details.

Added in v2 (dividends, global summaries, Faith-Shap stub)
----------------------------------------------------------
* ``dividend_values``: Harsanyi dividends ``a(T) = Delta_T f_x(empty)`` for arbitrary
  position tuples (mean + optional posterior covariance). This is full-order
  (``ell == L``) Faith-Shap -- the *pure* interaction that does not smear higher-order
  epistasis onto its subsets -- computed by the same polynomial layer with the R-weight
  swapped to ``[1, 0, 0, ...]`` (keep only the empty context ``S``). Verified against
  ``ShapBruteForce.bf_dividend_values`` / ``bf_dividend_covariance``.
* ``dividend_values_batch``, ``global_shap_importance``, ``global_dividend_summary``:
  global (dataset-level) summaries. Averaging ``|phi_i|`` / ``|a(T)|`` over a target
  sample gives model-level importance; the variance of ``a(T)`` over targets drawn from a
  product background ``pi`` is the (unnormalized) Sobol sensitivity index for ``T`` (see
  ``global_dividend_summary`` for the precise precondition).
* ``faith_shap_values``: truncated (``ell < L``) Faith-Shap -- SPECIFIED, NOT YET
  IMPLEMENTED (its exact redistribution weights need transcribing/testing against Tsai et
  al. 2023 first, so it raises rather than guessing).

Choosing an interaction index
-----------------------------
All the interaction quantities are the same position-factorized GKT moment collapsed by a
different weight vector (see ``_index_weight_fn``); pick by the question you are asking.

* ``shap_values`` (first-order Shapley): "how much does each single position contribute to
  *this* prediction ``f(x)``, fairly divided?" The unique attribution with local accuracy
  (the ``phi_i`` sum to ``f(x) - phi_0``).
* ``dividend_values`` (Harsanyi dividend == full-order Faith-Shap): "which position-*sets*
  are *irreducibly* epistatic?" ``a(T)`` is the pure joint effect -- a genuine 3-way effect
  contributes to its triple and to *nothing* smaller. Equals the functional-ANOVA
  component ``f_T`` under a product-``pi`` background at ``eta == 1``. NOT efficient (no
  per-order sum to ``f(x) - phi_0``); note the order-1 dividend ``a({i}) = f_x(i) -
  f_x(empty)`` is the ANOVA *main effect*, which differs from the Shapley value ``phi_i``.
* ``shap_interaction_values`` (SII / Lundberg): "non-additivity, in numbers comparable to
  the SHAP-interaction literature." It *averages* the discrete derivative over all contexts
  ``S``, which *smears* a true order-``m`` effect down onto all its lower-order subsets. Do
  not read its pairwise matrix as "pure pairwise epistasis" when high-order epistasis is
  present -- a single triplet lights up all three of its pairs. Use ``dividend_values`` for
  the clean reading.
* ``faith_shap_values`` (truncated Faith-Shap, ``ell < L``): "an *efficient* explanation
  using only interactions up to order ``ell``." Exact (== dividends) when the model's true
  maximum interaction order is ``<= ell``; otherwise it redistributes the higher-order
  epistasis into the ``<= ell`` terms (a different smearing than SII). STUB.
"""

from __future__ import annotations

import itertools
from math import comb, factorial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

# The polynomial layer of the generalized kernel trick (model-independent primitives).
from generalized_kernel_trick import (
    g_weights,
    poly_prod_over_positions,
    poly_prod_over_positions_bivariate,
    apply_R,
    apply_RxR,
)


# =============================================================================
# The SHAP computer
# =============================================================================


class ShapKernelTrickComputer:
    """Compute SHAP values (and posterior covariance) via the generalized kernel trick.

    A composition wrapper around a ``KernelTrickComputer`` (``self.ktc``): all kernel,
    training-data, and posterior quantities are read from ``self.ktc``, and this class
    adds only the polynomial layer (imported from ``generalized_kernel_trick``) plus the
    SHAP-specific factor builders and assembly. Composing rather than subclassing keeps
    the tested kernel-trick implementation untouched and makes the reuse boundary explicit
    (every reused quantity is an attribute access on ``self.ktc``).

    Typical use (mirrors how the kernel trick is used on g2p data)::

        from kernel_trick_computer_refactored import KernelTrickComputer
        ktc  = KernelTrickComputer(model, alphabet, seq_length)
        shap = ShapKernelTrickComputer(ktc=ktc)

        x_onehot = shap.target_to_onehot("ABACA")     # target to explain
        pi       = background_distribution             # (L, A), row-stochastic
        phi          = shap.shap_values(x_onehot, pi)              # (L,)   mean
        phi, cov     = shap.shap_values(x_onehot, pi, calc_covar=True)
        base         = shap.base_value(pi)                          # scalar

    Attributes:
        ktc: the composed ``KernelTrickComputer`` (source of all reused state).
        seq_length, n_alleles, alphabet, device: mirrored from ``ktc``.
        g: the cached SHAP weight vector ``g(s)`` (depends only on ``L``).
    """

    def __init__(
        self,
        model=None,
        alphabet: Optional[str] = None,
        seq_length: Optional[int] = None,
        ktc=None,
        g_dtype: torch.dtype = torch.float64,
    ):
        """Construct from an existing ``KernelTrickComputer``, or build one internally.

        Args:
            model, alphabet, seq_length: if ``ktc`` is None, these are forwarded to
                ``KernelTrickComputer(model, alphabet, seq_length)``.
            ktc: a pre-built ``KernelTrickComputer`` to compose with (preferred; the
                test suite already builds one on the shared fixture).
            g_dtype: dtype for the SHAP weight vector ``g`` (float64 by default).

        Raises:
            ValueError: if neither ``ktc`` nor all of (model, alphabet, seq_length)
                are provided.
        """
        # Compose rather than subclass; the reuse boundary is self.ktc.*
        if ktc is None:
            if model is None or alphabet is None or seq_length is None:
                raise ValueError("Provide either `ktc`, or all of (model, alphabet, seq_length).")
            # Local import so this module stays importable even when the refactored
            # module is resolved via the test-suite path shim (see conftest.py).
            from kernel_trick_computer_refactored import KernelTrickComputer
            ktc = KernelTrickComputer(model, alphabet, seq_length)

        self.ktc = ktc                                  # all kernel / posterior state
        self.seq_length = ktc.seq_length
        self.n_alleles = ktc.n_alleles
        self.alphabet = ktc.alphabet
        self.device = ktc.device

        # SHAP weight vector g(s); cached once since it depends only on L.
        self._g_dtype = g_dtype
        self.g = g_weights(self.seq_length, k=1, device=self.device, dtype=g_dtype)
        # Per-order interaction weight cache g_k(s); seeded with the first-order vector.
        self._g_cache: Dict[int, torch.Tensor] = {1: self.g}

    def _g_for_order(self, k: int) -> torch.Tensor:
        """Return (and cache) the order-``k`` interaction weight vector ``g_k``.

        ``g_1`` is the first-order Shapley weight ``self.g``; higher orders are built on
        demand from :func:`g_weights` and reused across interaction tuples of the same
        size. See :func:`g_weights` for the exact formula.
        """
        g = self._g_cache.get(k)
        if g is None:
            g = g_weights(self.seq_length, k=k, device=self.device, dtype=self._g_dtype)
            self._g_cache[k] = g
        return g

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #

    def target_to_onehot(self, x: Union[str, torch.Tensor]) -> torch.Tensor:
        """Return the target sequence as a one-hot ``(L, A)`` tensor.

        The target is represented as a plain one-hot over the base alphabet (no ``'-'``
        wildcard channel), because a SHAP target is always a fully specified sequence.
        This differs from the extended-alphabet 'subseq' encoding the kernel trick uses
        for its M-rows.

        Args:
            x: either an ``L``-character string over ``self.alphabet``, or an
                already-one-hot tensor of shape ``(L, A)``.

        Returns:
            ``(L, A)`` one-hot tensor on ``self.device`` in the computer's comp dtype.
        """
        A = self.n_alleles
        if torch.is_tensor(x):
            assert x.shape == (self.seq_length, A), f"expected (L, A)=({self.seq_length},{A})"
            return x.to(device=self.device, dtype=self.ktc.comp_dtype)
        assert isinstance(x, str) and len(x) == self.seq_length
        idx = torch.tensor([self.alphabet.index(ch) for ch in x], device=self.device)
        oh = torch.zeros(self.seq_length, A, device=self.device, dtype=self.ktc.comp_dtype)
        oh[torch.arange(self.seq_length), idx] = 1.0
        return oh

    # ------------------------------------------------------------------ #
    # Base value and diagnostics
    # ------------------------------------------------------------------ #

    def base_value(self, pi: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        """Base value ``phi_0 = f_x(empty set) = E_pi[f]``.

        This is exactly the background expectation of the map, which the kernel trick
        already computes as ``E_MAP(pi)``. (``E_MAP`` carries no ``eta``; it uses the
        unscaled fixed-kernel columns, which is consistent with the SHAP background
        channel at ``eta == 1``.)

        Args:
            pi: background distribution ``(L, A)``, row-stochastic.
            eta: accepted for interface symmetry; not used by ``E_MAP``.

        Returns:
            Scalar tensor.
        """
        pi = pi.to(device=self.device, dtype=self.ktc.comp_dtype)
        return self.ktc.E_MAP(pi)

    def local_accuracy_residual(
        self, x: Union[str, torch.Tensor], pi: torch.Tensor, eta: float = 1.0
    ) -> torch.Tensor:
        """Diagnostic residual ``f(x) - phi_0 - sum_i phi_i`` (should be ~0 at ``eta == 1``).

        Relies on two facts reused from the kernel trick: the posterior mean ``f(x)``
        equals ``E_MAP`` evaluated at the delta distribution on ``x`` (a one-hot ``pi``,
        since fixing every position leaves no averaging), and ``phi_0 == E_MAP(pi)``.
        This is the cheapest correctness anchor for the SHAP mean path.

        Args:
            x: target sequence (string or one-hot ``(L, A)``).
            pi: background distribution ``(L, A)``.
            eta: background-channel scale (the identity holds at ``eta == 1``).

        Returns:
            Scalar tensor.
        """
        x_oh = self.target_to_onehot(x)
        f_x = self.ktc.E_MAP(x_oh)                    # delta distribution => f(x)
        phi0 = self.base_value(pi, eta=eta)
        phi = self.shap_values(x_oh, pi, eta=eta, calc_covar=False)
        return f_x - phi0 - phi.sum()

    # ------------------------------------------------------------------ #
    # Per-position factor construction (the SHAP "encoding")
    # ------------------------------------------------------------------ #

    def _mean_channels(
        self, x_onehot: torch.Tensor, pi: torch.Tensor, eta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the two per-position channels for the SHAP *mean*.

        For every position ``p`` and training sequence ``y_t``, the two contractions
        that appear in ``(m^i K)_y`` are:

            pi-channel     c0[p, t] = sum_c pi^p_c a^p_{c, y_{t,p}}   (== compute_zeta_pc . y_t)
            fixed-channel  c1[p, t] = a^p_{x_p, y_{t,p}}              (== site_kernels[p, x_p, .] . y_t)

        The pi-channel is exactly the kernel trick's ``compute_zeta_pc`` (which includes
        the ``eta`` scaling); the fixed-channel pins the specific target character
        ``x_p``, which the ordinary tool never does. ``eta`` scales only the pi-channel.

        Returns:
            ``(c0, c1)``, each of shape ``(L, t)``.
        """
        L, A = self.seq_length, self.n_alleles
        # training design matrix reshaped to (t, L, A)
        Xtr = self.ktc._train_x.reshape(-1, L, A).to(x_onehot.dtype)

        # pi-channel via compute_zeta_pc (batched API -> take batch 0)
        pi_b = pi.to(device=self.device, dtype=x_onehot.dtype).unsqueeze(0)   # (1, L, A)
        zeta = self.ktc.compute_zeta_pc(pi_b, eta)[0]                          # (L, A)

        # fixed-channel rows a^p_{x_p, .} = one-hot(x_p) . site_kernels[p]
        site = self.ktc.site_kernels.to(x_onehot.dtype)                       # (L, A, A)
        Ax = torch.einsum("pc,pca->pa", x_onehot, site)                        # (L, A)

        c0 = torch.einsum("la,tla->lt", zeta, Xtr)                             # (L, t) pi-channel
        c1 = torch.einsum("la,tla->lt", Ax, Xtr)                               # (L, t) fixed-channel
        return c0, c1

    # ------------------------------------------------------------------ #
    # z = m^i K_{*,X}  (mirrors compute_z; polynomial internals)
    # ------------------------------------------------------------------ #

    def _shap_z_mean(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        eta: float,
        indices: Sequence[int],
    ) -> torch.Tensor:
        """Build ``mK = m^i K_{*,X}`` for each requested SHAP index ``i``.

        Mirrors ``compute_z`` (which produced ``M K_{*,X}`` for subseq rows), but the
        reduction over positions is a polynomial product + ``R``-contraction instead of
        ``torch.prod``. For a fixed index ``i`` the per-position factors are:

            * position i   : degree-0 factor ``[ c1[i] - c0[i] ]``   shape (t, 1)
            * position p!=i: degree-1 factor ``[ c0[p], c1[p] ]``    shape (t, 2)

        Their polynomial product over ``p`` is a ``(t, L)`` coefficient tensor;
        ``apply_R`` contracts it with ``g`` to give ``(m^i K)_y`` per training ``y``.

        Args:
            x_onehot: target one-hot ``(L, A)``.
            pi: background distribution ``(L, A)``.
            eta: background-channel scale.
            indices: which SHAP positions to compute (default: all ``L``).

        Returns:
            ``mK`` of shape ``(len(indices), t)``.

        Note:
            Only position ``i``'s factor changes across ``i``. Prefix/suffix products
            of the shared degree-1 ``[c0, c1]`` factors are built once
            (``prefix[i] = prod_{p<i}``, ``suffix[i] = prod_{p>i}``), then combined per
            ``i`` with a single convolution and scaled by ``c1[i] - c0[i]``. This makes
            the per-``i`` cost O(L) amortized and division-free.
        """
        L = self.seq_length
        c0, c1 = self._mean_channels(x_onehot, pi, eta)      # (L, t), (L, t)
        t = c0.shape[1]
        g = self.g.to(c0.dtype)

        # Shared degree-1 factor per position: q_p(w) = c0[p] + c1[p] w -> [c0, c1].
        factors = torch.stack([c0, c1], dim=-1)              # (L, t, 2)

        # prefix[i] = prod_{p<i} q_p ;  suffix[i] = prod_{p>i} q_p   (batched over t).
        ones = c0.new_ones(t, 1)
        prefix = [ones] * L
        for p in range(1, L):
            prefix[p] = poly_prod_over_positions([prefix[p - 1], factors[p - 1]])
        suffix = [ones] * L
        for p in range(L - 2, -1, -1):
            suffix[p] = poly_prod_over_positions([suffix[p + 1], factors[p + 1]])

        mK_rows = []
        for i in indices:
            prod_not_i = poly_prod_over_positions([prefix[i], suffix[i]])    # (t, <=L)
            R_val = apply_R(prod_not_i, g)                                   # (t,)
            mK_rows.append((c1[i] - c0[i]) * R_val)                          # (t,)
        return torch.stack(mK_rows, dim=0)                                   # (len(indices), t)

    # ------------------------------------------------------------------ #
    # R = m^i K m^j  (mirrors compute_R; bivariate internals)
    # ------------------------------------------------------------------ #

    def _cov_scalars(
        self, x_onehot: torch.Tensor, pi: torch.Tensor, eta: float
    ) -> dict:
        """Per-position scalar kernel contractions needed by the covariance.

        With ``a^p`` symmetric (``a^p_{c,c'} = a^p_{c',c}``), each position's covariance
        factor is built from three scalars:

            K_ff[p] = a^p_{x_p, x_p}                          (fixed . fixed)
            K_fa[p] = sum_c' a^p_{x_p, c'} pi^p_{c'}   * eta  (fixed . avg)
            K_aa[p] = sum_{c,c'} pi^p_c a^p_{c,c'} pi^p_c'  * eta^2  (avg . avg)

        ``eta`` multiplies each 'avg' (pi) appearance; ``K_ff`` carries no ``eta``.

        Returns:
            dict of tensors, each shape ``(L,)``.
        """
        L, A = self.seq_length, self.n_alleles
        site = self.ktc.site_kernels.to(x_onehot.dtype)                 # (L, A, A)
        pi = pi.to(device=self.device, dtype=x_onehot.dtype)            # (L, A)

        Kx = torch.einsum("pc,pca->pa", x_onehot, site)                 # a^p_{x_p, .}  (L, A)
        K_ff = torch.einsum("pa,pa->p", Kx, x_onehot)                   # (L,)
        K_fa = eta * torch.einsum("pa,pa->p", Kx, pi)                   # (L,)   fixed.avg
        K_aa = (eta * eta) * torch.einsum("pc,pca,pa->p", pi, site, pi) # (L,)   avg.avg
        return {"K_ff": K_ff, "K_fa": K_fa, "K_aa": K_aa}

    def _shap_R_cov(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        eta: float,
        indices: Sequence[int],
    ) -> torch.Tensor:
        """Build ``R[i,j] = (m^i K m^j)`` via the bivariate GKT.

        Mirrors ``compute_R`` but the per-position factor is a bivariate coefficient
        block in ``(w, w')``; the product over positions is a 2D convolution
        (``poly_prod_over_positions_bivariate``) and the collapse is ``apply_RxR``.
        Using the scalars from ``_cov_scalars``, the per-position factor is:

            p not in {i, j}   (deg (1,1)):  [[ K_aa,          K_fa        ],
                                             [ K_fa,          K_ff        ]]
            p == i, p != j    (deg (0,1)):  [[ K_fa - K_aa,   K_ff - K_fa ]]
            p == j, p != i    (deg (1,0)):  [[ K_fa - K_aa ],
                                             [ K_ff - K_fa ]]
            p == i == j       (deg (0,0)):  [[ K_ff - 2 K_fa + K_aa ]]

        Product over all ``p`` -> coefficient array -> ``apply_RxR`` -> scalar
        ``R[i,j]``.

        Returns:
            ``R`` of shape ``(len(indices), len(indices))``.

        Notes:
            * ``R = m^i K m^j`` has no dependence on training points -- it is the
              prior-kernel quadratic form (a single scalar per ``(i, j)``). The training
              data enters only via the ``(zS)(zS)^T`` term assembled in ``shap_values``.
            * This is the clarity-first per-``(i,j)`` loop. A later optimization can
              mirror the mean path (prefix/suffix products of the shared ``(1,1)`` blocks,
              corrected at positions ``i`` and ``j``); kept naive here to stay checkable
              against brute force.
        """
        L, A = self.seq_length, self.n_alleles
        sc = self._cov_scalars(x_onehot, pi, eta)
        K_ff, K_fa, K_aa = sc["K_ff"], sc["K_fa"], sc["K_aa"]     # each (L,)
        g = self.g.to(K_ff.dtype)

        def block(kind: str, p: int) -> torch.Tensor:
            """Minimal-degree bivariate coefficient block for position ``p``.

            Layout: ``B[a, b]`` multiplies ``w^a (w')^b``. The shape reflects the ACTUAL
            degree in each variable (not zero-padded to (2, 2)) so the product over
            positions never exceeds degree ``(L-1, L-1)`` -- exactly the range ``g``
            covers. Padding to (2, 2) would be numerically harmless but would inflate
            the coefficient-array shape past ``len(g)`` and break ``apply_RxR``'s
            ``g[:D]`` slice.
            """
            ff, fa, aa = K_ff[p], K_fa[p], K_aa[p]
            if kind == "both_free":          # p not in {i, j}  (deg (1,1))
                B = K_ff.new_zeros(2, 2)
                B[0, 0] = aa; B[0, 1] = fa; B[1, 0] = fa; B[1, 1] = ff
            elif kind == "i_only":           # p == i, p != j   (deg (0,1))
                B = K_ff.new_zeros(1, 2)
                B[0, 0] = fa - aa; B[0, 1] = ff - fa
            elif kind == "j_only":           # p == j, p != i   (deg (1,0))
                B = K_ff.new_zeros(2, 1)
                B[0, 0] = fa - aa; B[1, 0] = ff - fa
            elif kind == "both":             # p == i == j      (deg (0,0))
                B = K_ff.new_zeros(1, 1)
                B[0, 0] = ff - 2.0 * fa + aa
            else:
                raise ValueError(kind)
            return B

        n = len(indices)
        R = K_ff.new_zeros(n, n)
        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                factors = []
                for p in range(L):
                    if p == i and p == j:
                        factors.append(block("both", p))
                    elif p == i:
                        factors.append(block("i_only", p))
                    elif p == j:
                        factors.append(block("j_only", p))
                    else:
                        factors.append(block("both_free", p))
                coeffs2 = poly_prod_over_positions_bivariate(factors)   # (<=L+1, <=L+1)
                R[a, b] = apply_RxR(coeffs2, g)
        return R

    # ------------------------------------------------------------------ #
    # Public: first-order SHAP values (mean + optional covariance)
    # ------------------------------------------------------------------ #

    def shap_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        calc_covar: bool = False,
        use_double: bool = True,
    ):
        """Compute first-order SHAP values ``phi_i(f, x)`` and (optionally) their covariance.

        The assembly is identical to the kernel trick's ``_gauge_fixed_posterior``::

            mean = mK @ Qy
            cov  = R - (mK @ S) @ (mK @ S)^T

        Only ``mK`` (= ``m^i K_{*,X}``) and ``R`` (= ``m^i K m^j``) come from the new
        polynomial layer instead of the scalar-product layer.

        Args:
            x: target sequence (string over the alphabet, or one-hot ``(L, A)``).
            pi: background distribution, row-stochastic ``(L, A)``.
            eta: background-channel scale (see module notes on ``eta``). Default 1.0.
            indices: SHAP positions to compute; default all positions ``0..L-1``.
            calc_covar: if True, also return the posterior covariance of the ``phi_i``.
            use_double: run the polynomial/assembly in float64 (recommended, since the
                SHAP weights ``g`` are near-cancelling).

        Returns:
            ``phi`` of shape ``(len(indices),)`` if ``calc_covar`` is False, else
            ``(phi, cov)`` with ``cov`` of shape ``(len(indices), len(indices))``.

        Both paths are verified against the brute-force reference
        (``bf_shap_values`` / ``bf_shap_covariance``).
        """
        # Precision: SHAP weights g are near-cancelling, so float64 by default.
        if use_double:
            self.ktc.set_precision(use_double=True)        # recasts S, Qy, kernels; clears caches
        comp = self.ktc.comp_dtype

        x_oh = self.target_to_onehot(x).to(comp)
        pi = pi.to(device=self.device, dtype=comp)
        # Validate the background like any pi (add-batch then check simplex).
        self.ktc.validate_pis(pi)

        if indices is None:
            indices = list(range(self.seq_length))

        # z = m^i K_{*,X}  (polynomial internals).
        mK = self._shap_z_mean(x_oh, pi, eta, indices)     # (len(indices), t)

        # Mean assembly: contract with Qy (== mean_cache).
        Qy = self.ktc.Qy.to(comp)                          # (t,)
        mean = mK @ Qy                                     # (len(indices),)

        if not calc_covar:
            return mean

        # m^i K m^j via the bivariate GKT.
        R = self._shap_R_cov(x_oh, pi, eta, indices)       # (len(indices), len(indices))

        # Covariance assembly, identical to _gauge_fixed_posterior:
        #   cov = R - (mK S)(mK S)^T,  with S == covar_cache.
        S = self.ktc.S.to(comp)                            # (t, t)
        zS = mK @ S                                        # (len(indices), t)
        cov = R - zS @ zS.T                                # (len(indices), len(indices))
        return mean, cov

    # ------------------------------------------------------------------ #
    # Public: batched over target sequences
    # ------------------------------------------------------------------ #

    def shap_values_batch(
        self,
        xs: Sequence[Union[str, torch.Tensor]],
        pi: torch.Tensor,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        use_double: bool = True,
    ) -> torch.Tensor:
        """Mean SHAP values for many target sequences against one background ``pi``.

        The intended heavy-use axis is "many x, one pi" (global summaries such as mean
        ``|phi_i|`` over a dataset). This currently loops over targets; a vectorized
        version would batch ``_mean_channels`` over targets (the fixed-channel is the
        only x-dependent piece; ``c0`` / ``zeta`` are shared across all ``x``).

        Args:
            xs: target sequences (strings or one-hot tensors).
            pi: background distribution ``(L, A)``.
            eta: background-channel scale.
            indices: SHAP positions; default all.
            use_double: run in float64.

        Returns:
            Tensor of shape ``(len(xs), len(indices))``.
        """
        rows = [self.shap_values(x, pi, eta=eta, indices=indices,
                                 calc_covar=False, use_double=use_double)
                for x in xs]
        return torch.stack(rows, dim=0)

    # ------------------------------------------------------------------ #
    # Interaction bookkeeping: resolve tuples, build rows / covariance
    # ------------------------------------------------------------------ #

    def _resolve_interactions(
        self,
        interactions: Optional[Sequence[Sequence[int]]],
        order: Optional[int],
        indices: Optional[Sequence[int]],
    ) -> List[Tuple[int, ...]]:
        """Normalize ``(interactions, order, indices)`` to a list of position tuples.

        If ``interactions`` is given it is used directly (each entry sorted and
        de-duplicated); otherwise every size-``order`` subset of ``indices`` (default:
        all positions) is enumerated in ``itertools.combinations`` order. The returned
        list defines the ordering of every downstream result vector / matrix, so callers
        should read it back rather than assume an order.
        """
        L = self.seq_length
        if interactions is not None:
            out: List[Tuple[int, ...]] = []
            for T in interactions:
                Tt = tuple(sorted(set(int(p) for p in T)))
                if len(Tt) == 0:
                    raise ValueError("interaction tuples must be non-empty.")
                if any(p < 0 or p >= L for p in Tt):
                    raise ValueError(f"interaction positions must lie in [0, {L}); got {tuple(T)}.")
                out.append(Tt)
            return out
        if order is None or order < 1:
            raise ValueError("provide `interactions`, or an integer `order` >= 1.")
        if order > L:
            raise ValueError(f"order {order} exceeds sequence length {L}.")
        idx = list(range(L)) if indices is None else [int(i) for i in indices]
        return [tuple(T) for T in itertools.combinations(idx, order)]

    def _shap_z_interaction(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        eta: float,
        interactions: Sequence[Tuple[int, ...]],
        weight_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Build ``mK = m^T K_{*,X}`` for each interaction tuple ``T`` (raw index weights).

        Generalizes :meth:`_shap_z_mean` from a single index ``i`` to a set ``T``: every
        position ``p in T`` takes the degree-0 difference factor ``[c1[p] - c0[p]]`` and
        every other position the shared degree-1 factor ``[c0[p], c1[p]]``. Their
        polynomial product over ``p`` is a ``(t, L - |T| + 1)`` coefficient tensor, which
        ``apply_R`` contracts with the order-``|T|`` weight vector.

        This is the position-factorized discrete derivative ``Delta_T f_x(S)`` summed
        against ``g_{|T|}(|S|)`` over all ``S`` disjoint from ``T`` -- exactly the
        interaction analog of the first-order row.

        Args:
            weight_fn: callable mapping the interaction order ``k = |T|`` to the R-weight
                vector (length ``L - k + 1``). This is the *only* thing that distinguishes
                one interaction index from another -- the moment itself is identical. If
                None (default), the Shapley Interaction Index weight ``self._g_for_order``
                is used, so existing SII callers are unchanged. See :meth:`_index_weight_fn`
                for the named choices (``'sii'`` vs ``'dividend'``).

        Returns:
            ``mK`` of shape ``(len(interactions), t)``; these are the *unnormalized* rows
            for the chosen index (the public method rescales them per its normalization).
        """
        L = self.seq_length
        if weight_fn is None:
            weight_fn = self._g_for_order
        c0, c1 = self._mean_channels(x_onehot, pi, eta)          # (L, t), (L, t)
        rows = []
        for T in interactions:
            Tset = set(T)
            g_k = weight_fn(len(Tset)).to(c0.dtype)
            factors = []
            for p in range(L):
                if p in Tset:
                    factors.append((c1[p] - c0[p]).unsqueeze(-1))        # (t, 1) deg 0
                else:
                    factors.append(torch.stack([c0[p], c1[p]], dim=-1))  # (t, 2) deg 1
            coeffs = poly_prod_over_positions(factors)                   # (t, L-|T|+1)
            rows.append(apply_R(coeffs, g_k))                            # (t,)
        return torch.stack(rows, dim=0)                                  # (n, t)

    def _shap_R_interaction(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        eta: float,
        interactions: Sequence[Tuple[int, ...]],
        weight_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Build ``R[a, b] = m^{T_a} K m^{T_b}`` via the bivariate GKT (raw index weights).

        Generalizes :meth:`_shap_R_cov`: the per-position bivariate factor branches on
        ``p in T_a`` (the ``w`` axis) and ``p in T_b`` (the ``w'`` axis) instead of
        ``p == i`` / ``p == j``. It reuses the same three scalars from
        :meth:`_cov_scalars`; the block layout is identical to the first-order case with
        those two membership tests substituted:

            p in T_a and T_b  (deg (0,0)):  [[ ff - 2 fa + aa ]]
            p in T_a only     (deg (0,1)):  [[ fa - aa,  ff - fa ]]
            p in T_b only     (deg (1,0)):  [[ fa - aa ], [ ff - fa ]]
            p in neither      (deg (1,1)):  [[ aa, fa ], [ fa, ff ]]

        The product over positions is contracted with ``apply_RxR`` using the two tuples'
        weight vectors (rows for ``T_a``, columns for ``T_b``) -- they differ when the
        tuples have different sizes.

        Args:
            weight_fn: callable mapping order ``k`` to the R-weight vector; None (default)
                selects the SII weight ``self._g_for_order`` so SII callers are unchanged.
                See :meth:`_index_weight_fn`.

        Returns:
            ``R`` of shape ``(len(interactions), len(interactions))`` (unnormalized).
        """
        L = self.seq_length
        if weight_fn is None:
            weight_fn = self._g_for_order
        sc = self._cov_scalars(x_onehot, pi, eta)
        K_ff, K_fa, K_aa = sc["K_ff"], sc["K_fa"], sc["K_aa"]     # each (L,)

        # per-order weight vectors in the working dtype, cached across the (a, b) loop
        g_by_order: Dict[int, torch.Tensor] = {}

        def gord(k: int) -> torch.Tensor:
            g = g_by_order.get(k)
            if g is None:
                g = weight_fn(k).to(K_ff.dtype)
                g_by_order[k] = g
            return g

        def block(in_a: bool, in_b: bool, p: int) -> torch.Tensor:
            """Minimal-degree bivariate block for position ``p`` (see method docstring)."""
            ff, fa, aa = K_ff[p], K_fa[p], K_aa[p]
            if in_a and in_b:            # deg (0, 0)
                B = K_ff.new_zeros(1, 1)
                B[0, 0] = ff - 2.0 * fa + aa
            elif in_a:                   # deg (0, 1)
                B = K_ff.new_zeros(1, 2)
                B[0, 0] = fa - aa; B[0, 1] = ff - fa
            elif in_b:                   # deg (1, 0)
                B = K_ff.new_zeros(2, 1)
                B[0, 0] = fa - aa; B[1, 0] = ff - fa
            else:                        # deg (1, 1)
                B = K_ff.new_zeros(2, 2)
                B[0, 0] = aa; B[0, 1] = fa; B[1, 0] = fa; B[1, 1] = ff
            return B

        sets = [set(T) for T in interactions]
        n = len(sets)
        R = K_ff.new_zeros(n, n)
        for a, Ta in enumerate(sets):
            ga = gord(len(Ta))
            for b, Tb in enumerate(sets):
                gb = gord(len(Tb))
                factors = [block(p in Ta, p in Tb, p) for p in range(L)]
                coeffs2 = poly_prod_over_positions_bivariate(factors)
                R[a, b] = apply_RxR(coeffs2, ga, gb)
        return R

    # ------------------------------------------------------------------ #
    # Public: pairwise / higher-order interaction SHAP
    # ------------------------------------------------------------------ #

    def shap_interaction_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        interactions: Optional[Sequence[Sequence[int]]] = None,
        order: Optional[int] = 2,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        normalize: str = "shap",
        calc_covar: bool = False,
        use_double: bool = True,
    ):
        """Pairwise and higher-order interaction SHAP (notes, Section 3.4).

        Computes the interaction attribution of a position set ``T = {i_1, ..., i_k}``,

            phi_T(f, x) = sum_{S : S cap T = empty} g_k(|S|) * Delta_T f_x(S),

        where ``Delta_T`` is the discrete k-th derivative of the conditional expectation
        ``f_x(S) = E_pi[ f | X_S = x_S ]``,

            Delta_T f_x(S) = sum_{W subseteq T} (-1)^{|T|-|W|} f_x(S u W),

        which for ``k == 2`` is the familiar four-term
        ``f_x(S u {i,j}) - f_x(S u i) - f_x(S u j) + f_x(S)`` of the notes.

        Shared machinery
        ----------------
        Because ``S`` and ``T`` are disjoint, ``Delta_T f_x(S)`` factorizes over
        positions, so it is exactly the position-factorizable form the GKT exploits. The
        implementation reuses the first-order polynomial layer unchanged; the *only*
        change from :meth:`shap_values` is that EVERY position in ``T`` takes the
        degree-0 difference factor ``c1[p] - c0[p]`` (first order does this for the single
        position ``i``). See :meth:`_shap_z_interaction` / :meth:`_shap_R_interaction`.

        Interaction index and weight
        ----------------------------
        Interactions (unlike first-order Shapley) are not unique, so a weight ``g_k(s)``
        must be chosen. This uses the Shapley Interaction Index (SII; Grabisch-Roubens),

            g_k(s) = s! (L - k - s)! / (L - k + 1)!,

        which reduces to the first-order Shapley weight at ``k == 1`` (so
        ``shap_interaction_values(order=1)`` agrees with :meth:`shap_values`) and, at
        ``k == 2`` with the SHAP normalization below, reproduces the ``phi_ij`` written
        in the notes and in Lundberg et al. (SII was chosen over Shapley-Taylor /
        Faith-Shap to stay numerically comparable with those papers; the latter two add
        an efficiency decomposition at the cost of a top-order boundary rule or a
        cross-order least-squares solve, and can be layered on later on top of the same
        GKT moments.)

        Normalization
        -------------
        * ``normalize='shap'`` (default): divide an order-``k`` value by ``k!``. At
          ``k == 2`` this is the ``1/2`` symmetrization in the notes' formula (the
          interaction shared between ``(i, j)`` and ``(j, i)``); at ``k == 1`` it is a
          no-op, so first-order values are unchanged. This is the convention to use for
          the ``phi_ij`` matrix and for the "single positions + pairwise + higher-order
          epistasis" reading.
        * ``normalize='sii'``: return the raw SII value (no ``k!`` factor).

        Args:
            x: target sequence to explain (string over the alphabet, or one-hot ``(L, A)``).
            pi: background distribution, row-stochastic ``(L, A)``.
            interactions: explicit iterable of position tuples ``T`` (any orders, possibly
                mixed). If None, all size-``order`` subsets of ``indices`` are enumerated.
            order: interaction order to enumerate when ``interactions`` is None (default 2
                == pairwise). Ignored when ``interactions`` is given.
            eta: background-channel scale (see module notes on ``eta``). Default 1.0.
            indices: universe of positions to enumerate over when ``interactions`` is None
                (default: all positions). Ignored when ``interactions`` is given.
            normalize: ``'shap'`` (default) or ``'sii'`` (see above).
            calc_covar: if True, also return the posterior covariance of the interactions.
            use_double: run the polynomial / assembly in float64 (recommended).

        Returns:
            ``(interactions, phi)`` if ``calc_covar`` is False, else
            ``(interactions, phi, cov)``, where ``interactions`` is the resolved list of
            tuples (defining the row/column order), ``phi`` has shape ``(n,)`` and ``cov``
            shape ``(n, n)`` with ``n = len(interactions)``.

        Verify with ``shap_brute_force.ShapBruteForce.bf_interaction_values`` /
        ``bf_interaction_covariance`` (independent subset enumeration).
        """
        if normalize not in ("shap", "sii"):
            raise ValueError("normalize must be 'shap' or 'sii'.")

        # Precision: SHAP weights are near-cancelling, so float64 by default.
        if use_double:
            self.ktc.set_precision(use_double=True)
        comp = self.ktc.comp_dtype

        x_oh = self.target_to_onehot(x).to(comp)
        pi = pi.to(device=self.device, dtype=comp)
        self.ktc.validate_pis(pi)

        T_list = self._resolve_interactions(interactions, order, indices)

        # Unnormalized SII rows mK = m^T K_{*,X}.
        mK = self._shap_z_interaction(x_oh, pi, eta, T_list)         # (n, t)

        # Per-tuple normalization: 'shap' divides an order-k value by k! (== 1/2 at k=2,
        # matching the notes; a no-op at k=1). 'sii' keeps the raw index.
        if normalize == "shap":
            scale = torch.tensor([1.0 / factorial(len(T)) for T in T_list],
                                 device=self.device, dtype=comp)
        else:
            scale = torch.ones(len(T_list), device=self.device, dtype=comp)

        mK = scale[:, None] * mK                                     # (n, t)
        Qy = self.ktc.Qy.to(comp)                                    # (t,)
        phi = mK @ Qy                                                # (n,)

        if not calc_covar:
            return T_list, phi

        # Covariance: same assembly as shap_values, R and mK from the interaction GKT.
        R = self._shap_R_interaction(x_oh, pi, eta, T_list)          # (n, n) SII
        R = (scale[:, None] * R) * scale[None, :]                    # apply normalization
        S = self.ktc.S.to(comp)                                      # (t, t) root
        zS = mK @ S                                                  # (n, t)
        cov = R - zS @ zS.T                                          # (n, n)
        return T_list, phi, cov

    def shap_pairwise_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        fill_diagonal: bool = False,
        use_double: bool = True,
    ) -> torch.Tensor:
        """Pairwise interaction matrix ``Phi[i, j] = phi_ij`` (notes, Section 3.4).

        Convenience wrapper over :meth:`shap_interaction_values` that lays the pairwise
        values out as a symmetric matrix. Off-diagonal entries are the notes' / Lundberg
        pairwise SHAP interaction (``normalize='shap'`` at order 2, i.e. SII / 2), with
        ``Phi[i, j] == Phi[j, i]``.

        Args:
            x: target sequence (string or one-hot ``(L, A)``).
            pi: background distribution ``(L, A)``, row-stochastic.
            eta: background-channel scale. Default 1.0.
            indices: positions to include (default all ``L``); the matrix is indexed in
                this order.
            fill_diagonal: if True, set ``Phi[i, i] = phi_i - sum_{j != i} phi_ij`` from
                the first-order values -- the Lundberg main-effect-on-diagonal convention,
                which makes each row sum to ``phi_i`` (a convenient self-consistency
                check). If False (default -- strictly the notes, which define only
                ``i != j``) the diagonal is left zero.
            use_double: run in float64.

        Returns:
            ``Phi`` of shape ``(m, m)`` with ``m = len(indices)``.
        """
        L = self.seq_length
        idx = list(range(L)) if indices is None else [int(i) for i in indices]
        pos = {p: r for r, p in enumerate(idx)}
        m = len(idx)

        pairs = list(itertools.combinations(idx, 2))
        if pairs:
            T_list, phi_pairs = self.shap_interaction_values(
                x, pi, interactions=pairs, eta=eta, normalize="shap",
                calc_covar=False, use_double=use_double,
            )
            Phi = phi_pairs.new_zeros(m, m)
            for (i, j), val in zip(T_list, phi_pairs):
                ri, rj = pos[i], pos[j]
                Phi[ri, rj] = val
                Phi[rj, ri] = val
        else:
            comp = self.ktc.comp_dtype
            Phi = torch.zeros(m, m, device=self.device, dtype=comp)

        if fill_diagonal:
            phi_i = self.shap_values(x, pi, eta=eta, indices=idx,
                                     calc_covar=False, use_double=use_double)  # (m,)
            offdiag = Phi.sum(dim=1)                        # diagonal is still zero here
            Phi[range(m), range(m)] = phi_i - offdiag
        return Phi

    # ------------------------------------------------------------------ #
    # Interaction index selection (weight vector for the R-functional)
    # ------------------------------------------------------------------ #

    def _index_weight_fn(self, index: str) -> Callable[[int], torch.Tensor]:
        """Return a callable ``k -> g_k`` giving the R-weight vector for a named index.

        Every interaction quantity in this module is the SAME position-factorized GKT
        moment (built by :meth:`_shap_z_interaction` / :meth:`_shap_R_interaction`)
        collapsed by a different weight vector ``g_k(s)`` -- the value the linear
        functional ``R`` contracts the size-``s`` (== ``|S|``) coefficient against.
        Choosing an interaction index is therefore *only* choosing this weight:

        * ``'sii'``      -- Shapley Interaction Index (Grabisch-Roubens),
          ``g_k(s) = s! (L - k - s)! / (L - k + 1)!`` (see :func:`g_weights`). Averages
          the discrete derivative ``Delta_T f_x(S)`` over *all* contexts ``S``, which
          smears a genuine high-order effect down onto its lower-order subsets.
        * ``'dividend'`` -- Harsanyi dividend == full-order (``ell == L``) Faith-Shap,
          ``g_k = [1, 0, 0, ...]`` (keep only ``S = empty``). The *pure*, non-smearing
          interaction ``a(T) = Delta_T f_x(empty)``.

        Args:
            index: ``'sii'`` or ``'dividend'``.

        Returns:
            A function mapping the interaction order ``k`` to a length-``(L - k + 1)``
            weight vector (``self._g_dtype`` on ``self.device``).
        """
        if index == "sii":
            return lambda k: self._g_for_order(k)
        if index == "dividend":
            def _delta_weight(k: int) -> torch.Tensor:
                v = torch.zeros(self.seq_length - k + 1,
                                device=self.device, dtype=self._g_dtype)
                v[0] = 1.0
                return v
            return _delta_weight
        raise ValueError(f"unknown interaction index '{index}' (use 'sii' or 'dividend').")

    # ------------------------------------------------------------------ #
    # Public: Harsanyi dividends (full-order Faith-Shap) -- the pure interaction
    # ------------------------------------------------------------------ #

    def dividend_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        interactions: Optional[Sequence[Sequence[int]]] = None,
        order: Optional[int] = 2,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        calc_covar: bool = False,
        use_double: bool = True,
    ):
        """Harsanyi dividends ``a(T) = Delta_T f_x(empty)`` -- the *pure* interaction.

        For a position set ``T = {i_1, ..., i_k}`` this returns

            a(T) = sum_{W subseteq T} (-1)^{|T| - |W|} f_x(W),

        the discrete ``k``-th derivative of the conditional expectation evaluated at the
        empty context (no averaging over other-position subsets ``S``). This is exactly
        *full-order* (``ell == L``) Faith-Shap: unlike SII, a genuine order-``m`` effect
        contributes to its own order-``m`` set and to *nothing smaller* -- there is no
        smearing onto subsets. It is the object to use for "which position-sets are
        *irreducibly* epistatic?".

        Relation to the SII path
        -------------------------
        Same machinery as :meth:`shap_interaction_values`, with two differences:
        (1) the R-weight is ``[1, 0, 0, ...]`` (keep only ``S = empty``) instead of the
        SII weight (:meth:`_index_weight_fn`); (2) there is no ``k!`` symmetrization --
        dividends are the raw multilinear coefficients. Concretely the mean row collapses
        to a plain product,

            a(T) = sum_t Qy_t * prod_{p in T} (c1[p,t] - c0[p,t]) * prod_{p not in T} c0[p,t],

        with ``c0`` / ``c1`` the pi- and fixed-channels of :meth:`_mean_channels`.

        Interpretation caveats
        ----------------------
        * Under a *product* background ``pi`` at ``eta == 1``, ``a(T)`` as a function of the
          target ``x`` is the functional-ANOVA component ``f_T(x_T)`` -- which is what makes
          the global variance a Sobol index (see :meth:`global_dividend_summary`).
        * Dividends are NOT efficient: there is no per-order sum equal to ``f(x) - phi_0``.
          Use them for ranking / structure discovery, not for an additive "the parts add
          up" explanation.
        * The order-1 dividend ``a({i}) = f_x(i) - f_x(empty)`` is the ANOVA *main effect*
          and differs from the first-order Shapley value :meth:`shap_values` -- do not mix
          them. (This is the "even the singletons change" point of full-order Faith-Shap.)

        Args:
            x, pi, interactions, order, eta, indices, calc_covar, use_double: as in
                :meth:`shap_interaction_values`. ``order`` enumerates all size-``order``
                subsets when ``interactions`` is None (default 2 == pairwise).

        Returns:
            ``(interactions, a)`` if ``calc_covar`` is False, else
            ``(interactions, a, cov)``; shapes match :meth:`shap_interaction_values`.

        Verify with ``shap_brute_force.ShapBruteForce.bf_dividend_values`` /
        ``bf_dividend_covariance`` (independent subset enumeration; the ``S = empty`` case
        of the SII brute force).
        """
        if use_double:
            self.ktc.set_precision(use_double=True)
        comp = self.ktc.comp_dtype

        x_oh = self.target_to_onehot(x).to(comp)
        pi = pi.to(device=self.device, dtype=comp)
        self.ktc.validate_pis(pi)

        T_list = self._resolve_interactions(interactions, order, indices)
        wfn = self._index_weight_fn("dividend")

        # Dividend rows m^T K_{*,X} (no k! normalization -- these are the raw coefficients).
        mK = self._shap_z_interaction(x_oh, pi, eta, T_list, weight_fn=wfn)   # (n, t)
        Qy = self.ktc.Qy.to(comp)                                            # (t,)
        a = mK @ Qy                                                          # (n,)

        if not calc_covar:
            return T_list, a

        R = self._shap_R_interaction(x_oh, pi, eta, T_list, weight_fn=wfn)   # (n, n)
        S = self.ktc.S.to(comp)                                             # (t, t)
        zS = mK @ S                                                         # (n, t)
        cov = R - zS @ zS.T                                                # (n, n)
        return T_list, a, cov

    def dividend_values_batch(
        self,
        xs: Sequence[Union[str, torch.Tensor]],
        pi: torch.Tensor,
        interactions: Optional[Sequence[Sequence[int]]] = None,
        order: Optional[int] = 2,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        use_double: bool = True,
    ) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
        """Dividends ``a(T)(x)`` for many targets against one background ``pi``.

        Loops :meth:`dividend_values` over ``xs`` (the "many x, one pi" axis used by the
        global summaries). Mirrors :meth:`shap_values_batch`.

        Returns:
            ``(interactions, A)`` with ``A`` of shape ``(len(xs), len(interactions))`` and
            ``interactions`` the resolved tuple list (row order of every column of ``A``).
        """
        T_list: Optional[List[Tuple[int, ...]]] = None
        rows = []
        for x in xs:
            T_list, a = self.dividend_values(
                x, pi, interactions=interactions, order=order, eta=eta,
                indices=indices, calc_covar=False, use_double=use_double,
            )
            rows.append(a)
        return T_list, torch.stack(rows, dim=0)

    # ------------------------------------------------------------------ #
    # Public: global (dataset-level) summaries
    # ------------------------------------------------------------------ #

    def global_shap_importance(
        self,
        xs: Sequence[Union[str, torch.Tensor]],
        pi: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
        eta: float = 1.0,
        use_double: bool = True,
    ) -> Tuple[List[int], torch.Tensor]:
        """Global first-order importance: mean ``|phi_i|`` over a target sample.

        The canonical SHAP global summary. Averaging the *absolute* value (not the signed
        value) is deliberate: a position can matter globally yet have signed ``phi_i`` that
        averages to ~0 because it pushes the phenotype up on some targets and down on
        others -- the sign cancellation would hide real importance.

        This is importance *under the target distribution you pass* (typically the training
        set, or a sample of it), and it is still relative to the background ``pi``; it is
        not "irrespective of target sequence" in any absolute sense. A biased ``xs`` biases
        the summary the same way.

        Args:
            xs: target sequences to average over (strings or one-hot tensors).
            pi: background distribution ``(L, A)``.
            indices: positions to score (default all ``L``).
            eta: background-channel scale.
            use_double: run in float64.

        Returns:
            ``(indices, mean_abs)`` with ``mean_abs`` of shape ``(len(indices),)``.
        """
        phis = self.shap_values_batch(xs, pi, eta=eta, indices=indices,
                                      use_double=use_double)          # (n_x, m)
        idx = list(range(self.seq_length)) if indices is None else [int(i) for i in indices]
        return idx, phis.abs().mean(dim=0)

    def global_dividend_summary(
        self,
        xs: Sequence[Union[str, torch.Tensor]],
        pi: torch.Tensor,
        interactions: Optional[Sequence[Sequence[int]]] = None,
        order: Optional[int] = 2,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        use_double: bool = True,
        normalize_sobol: bool = False,
    ) -> Tuple[List[Tuple[int, ...]], Dict[str, torch.Tensor]]:
        """Global interaction summary from dividends ``a(T)`` over a target sample.

        Computes ``a(T)(x)`` for every target ``x`` in ``xs`` (via
        :meth:`dividend_values_batch`) and reduces over the target axis into three
        dataset-level quantities per tuple ``T``:

        * ``mean_abs``    -- ``mean_x |a(T)(x)|``: global importance of the interaction
          ``T`` (analog of mean ``|phi_i|`` for single positions).
        * ``mean_signed`` -- ``mean_x a(T)(x)``: a diagnostic. Under a product background
          ``pi`` with targets drawn from ``pi``, ANOVA components of order ``>= 1`` are
          mean-zero, so this should be ~0 for non-empty ``T`` -- a useful sanity check that
          also flags when ``xs`` is far from ``pi``.
        * ``sobol_unnorm`` -- ``Var_x a(T)(x)``: the (unnormalized) Sobol sensitivity index
          for ``T``. This equality -- variance of the dividend == Sobol index -- holds
          because ``a(T)(x)`` is the functional-ANOVA component ``f_T`` (product ``pi``,
          ``eta == 1``), and ANOVA components are orthogonal, so the total variance splits
          cleanly across tuples. It is the global, non-smearing analog of the local
          dividend.

        Precondition for the Sobol reading
        -----------------------------------
        The variance is a Sobol index *only when the targets ``xs`` are sampled from the
        same product distribution ``pi`` used as the background*. If you instead pass the
        training set (generally NOT distributed as ``pi``), ``sobol_unnorm`` is the variance
        of the ANOVA component over the empirical target distribution -- still a reasonable
        global importance, but not the formal Sobol index. The reduction is reported either
        way; interpret it according to how ``xs`` was drawn.

        Args:
            xs, pi, interactions, order, eta, indices, use_double: as in
                :meth:`dividend_values_batch`.
            normalize_sobol: if True, also return ``sobol_normalized`` = ``sobol_unnorm``
                divided by the *sample* variance of ``f(x)`` over ``xs`` (reported as
                ``total_var_sample``). This normalized value is a sample estimate; it is a
                genuine Sobol *ratio* only under the same "``xs`` ~ ``pi``" precondition.

        Returns:
            ``(interactions, summary)`` where ``summary`` is a dict of ``(n,)`` tensors
            keyed by the names above (plus ``total_var_sample`` / ``sobol_normalized`` when
            ``normalize_sobol`` is True), and ``interactions`` is the resolved tuple list.
        """
        T_list, A = self.dividend_values_batch(
            xs, pi, interactions=interactions, order=order, eta=eta,
            indices=indices, use_double=use_double,
        )                                                            # (n_x, n)
        summary: Dict[str, torch.Tensor] = {
            "mean_abs": A.abs().mean(dim=0),
            "mean_signed": A.mean(dim=0),
            "sobol_unnorm": A.var(dim=0, unbiased=False),
        }
        if normalize_sobol:
            comp = self.ktc.comp_dtype
            # f(x) is E_MAP at the delta distribution on x (fixing every position).
            fvals = torch.stack([self.ktc.E_MAP(self.target_to_onehot(x).to(comp))
                                 for x in xs])                       # (n_x,)
            total_var = fvals.var(unbiased=False)
            summary["total_var_sample"] = total_var
            summary["sobol_normalized"] = summary["sobol_unnorm"] / total_var
        return T_list, summary

    # ------------------------------------------------------------------ #
    # Public: truncated Faith-Shap (Tsai, Yeh & Ravikumar 2023, Thm 19 / Eq. 16)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _faith_shap_coeff(s: int, t: int, ell: int) -> float:
        """Eq. (16) redistribution coefficient of dividend ``a(T)`` into Faith-Shap of ``S``.

        For ``S subset T`` with ``|S| = s <= ell`` and ``|T| = t``, Faith-Shap is

            E^{F-Shap}_S(v, ell) = a(S)
              + [s == 0 excluded] sum_{T superset S, |T| = t > ell} coeff(s, t, ell) * a(T),

        where, from Tsai-Yeh-Ravikumar (2023) Theorem 19,

            coeff(s, t, ell) = (-1)^{ell - s} * (s / (ell + s)) * C(ell, s)
                               * C(t - 1, ell) / C(t + ell - 1, ell + s).

        This is only invoked for the *tail* (``t > ell``); for ``t <= ell`` the contribution
        of ``a(T)`` to ``E_S`` is ``1`` if ``T == S`` and ``0`` otherwise (i.e. the identity
        ``a(S)`` term), which the caller handles directly. Verified against the paper's
        Table 2 worked example and against both anchors (``ell == 1`` -> Shapley,
        ``ell == L`` -> dividends).
        """
        return ((-1.0) ** (ell - s)) * (s / (ell + s)) * comb(ell, s) \
            * comb(t - 1, ell) / comb(t + ell - 1, ell + s)

    def faith_shap_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        order_max: int,
        interactions: Optional[Sequence[Sequence[int]]] = None,
        order: Optional[int] = 2,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        calc_covar: bool = False,
        use_double: bool = True,
        order_cap: Optional[int] = None,
    ):
        """Truncated Faith-Shap interaction index at maximum order ``order_max`` (== ``ell``).

        Faith-Shap (Tsai, Yeh & Ravikumar, JMLR 2023) is the unique interaction index whose
        order-``<= ell`` values are the coefficients of the *best* degree-``ell`` multilinear
        surrogate of the game ``v(S) = f_x(S)`` under the Shapley kernel, subject to
        efficiency. Two anchors fix its behavior:

        * ``ell == 1``  -> the first-order Shapley values (:meth:`shap_values`);
        * ``ell == L``  -> the Harsanyi dividends (:meth:`dividend_values`).

        Closed form (Theorem 19, Eq. 16)
        --------------------------------
        For a set ``S`` with ``|S| = s <= ell``,

            E^{F-Shap}_S(v, ell) = a(v, S)
                + (-1)^{ell-s} (s/(ell+s)) C(ell, s)
                  * sum_{T superset S, |T| > ell}  [ C(|T|-1, ell) / C(|T|+ell-1, ell+s) ] a(v, T),

        i.e. the dividend of ``S`` plus a fixed linear combination of the *higher* dividends
        of supersets ``T`` with ``|T| > ell``. It is therefore *exact* (equals the dividends,
        no smearing) precisely when the true maximum interaction order is ``<= ell``; when the
        true order exceeds ``ell`` it redistributes that higher-order epistasis down into the
        ``<= ell`` terms (a projection-induced smearing, distinct from SII's context-averaging).

        Implementation
        --------------
        Rather than fold Eq. (16) into a single R-weight (its redistribution couples different
        tuple sizes, so it is not a per-order ``g_k(s)``), Faith-Shap is assembled as an
        explicit linear map on the already-validated dividends: this method computes ``a(S)``
        for the requested tuples and ``a(T)`` for the supersets ``T`` (``ell < |T| <=
        order_cap``) via :meth:`dividend_values`, then applies the Eq. (16) coefficients
        (:meth:`_faith_shap_coeff`). This reuses the verified dividend GKT path, keeps the new
        logic to the coefficient bookkeeping, and makes the covariance compose exactly as
        ``M @ cov_a @ M.T`` (a fixed linear map ``M`` applied to the joint dividend covariance).

        Tail truncation (``order_cap``)
        -------------------------------
        Eq. (16)'s tail runs over *all* supersets up to size ``L``. For an EXACT Faith-Shap,
        ``order_cap`` must be ``L`` (the default when ``order_cap is None``). Setting
        ``order_cap < L`` additionally truncates the tail sum -- a genuine approximation that
        drops the contribution of dividends above ``order_cap``. This is only advisable when
        those high-order dividends are known/expected to be negligible (e.g. sparse, low-order
        epistasis); the value is then Faith-Shap of the ``order_cap``-order truncation of the
        model. The default keeps it exact.

        Args:
            x: target sequence (string over the alphabet, or one-hot ``(L, A)``).
            pi: background distribution ``(L, A)``, row-stochastic. (Uniform ``pi`` gives an
                average-background explanation; a one-hot ``pi`` at a wild-type sequence gives
                a wild-type-referenced explanation -- see :func:`make_background`.)
            order_max: the Faith-Shap maximum interaction order ``ell`` (>= 1). Reported tuples
                must have size ``<= ell``.
            interactions: explicit tuples ``S`` to score (each ``|S| <= order_max``). If None,
                all size-``order`` subsets of ``indices`` are enumerated.
            order: size to enumerate when ``interactions`` is None (default 2). Must be
                ``<= order_max``.
            eta: background-channel scale. Default 1.0.
            indices: universe of positions when enumerating (default all).
            calc_covar: if True, also return the posterior covariance of the reported tuples.
            use_double: run in float64 (recommended; the weights are near-cancelling).
            order_cap: highest dividend order to include in the tail sum (default ``L``, exact).

        Returns:
            ``(interactions, phi)`` if ``calc_covar`` is False, else ``(interactions, phi, cov)``,
            with ``phi`` of shape ``(n,)`` and ``cov`` of shape ``(n, n)`` over the reported
            tuples (row/col order == returned ``interactions``).

        Verify with ``shap_brute_force.ShapBruteForce.bf_faith_shap_values`` (weighted
        least-squares brute force) and the anchors ``order_max == 1`` -> :meth:`shap_values`,
        ``order_max == self.seq_length`` -> :meth:`dividend_values`.
        """
        L = self.seq_length
        ell = int(order_max)
        if ell < 1 or ell > L:
            raise ValueError(f"order_max (ell) must be in [1, {L}]; got {order_max}.")
        cap = L if order_cap is None else int(order_cap)
        if cap < ell or cap > L:
            raise ValueError(f"order_cap must be in [ell, L] = [{ell}, {L}]; got {order_cap}.")

        # Reported tuples S (each must have |S| <= ell).
        S_list = self._resolve_interactions(interactions, order, indices)
        if any(len(S) > ell for S in S_list):
            bad = [S for S in S_list if len(S) > ell]
            raise ValueError(f"Faith-Shap reports tuples of size <= order_max={ell}; "
                             f"got larger tuples {bad}.")

        idx_universe = list(range(L)) if indices is None else [int(i) for i in indices]

        # Enumerate every dividend tuple we need: the reported S's, plus all supersets T
        # (ell < |T| <= cap) of any reported S drawn from the same position universe.
        needed: "set[Tuple[int, ...]]" = set(S_list)
        for S in S_list:
            rest = [p for p in idx_universe if p not in set(S)]
            for extra_k in range(1, len(rest) + 1):
                for extra in itertools.combinations(rest, extra_k):
                    T = tuple(sorted(set(S) | set(extra)))
                    if ell < len(T) <= cap:
                        needed.add(T)
        div_tuples = sorted(needed, key=lambda T: (len(T), T))
        div_index = {T: i for i, T in enumerate(div_tuples)}

        # Dividends a(T) (+ joint covariance) via the validated GKT path.
        if calc_covar:
            _, a_vec, a_cov = self.dividend_values(
                x, pi, interactions=div_tuples, eta=eta,
                calc_covar=True, use_double=use_double,
            )
        else:
            _, a_vec = self.dividend_values(
                x, pi, interactions=div_tuples, eta=eta,
                calc_covar=False, use_double=use_double,
            )
        comp = a_vec.dtype

        # Linear map M: E^{F-Shap}_S = a(S) + sum_{T superset S, ell<|T|<=cap} coeff(|S|,|T|,ell) a(T).
        n = len(S_list)
        M = a_vec.new_zeros(n, len(div_tuples))
        for r, S in enumerate(S_list):
            Sset = set(S); s = len(S)
            M[r, div_index[S]] += 1.0                     # identity a(S) term
            for T in div_tuples:
                if len(T) > ell and Sset.issubset(T):
                    M[r, div_index[T]] += self._faith_shap_coeff(s, len(T), ell)

        phi = M @ a_vec                                   # (n,)
        if not calc_covar:
            return S_list, phi
        cov = M @ a_cov @ M.T                             # (n, n)
        return S_list, phi, cov

    def faith_shap_values_batch(
        self,
        xs: Sequence[Union[str, torch.Tensor]],
        pi: torch.Tensor,
        order_max: int,
        interactions: Optional[Sequence[Sequence[int]]] = None,
        order: Optional[int] = 2,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        use_double: bool = True,
        order_cap: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
        """Faith-Shap ``E_S(x)`` for many targets against one background ``pi``.

        Loops :meth:`faith_shap_values` over ``xs`` (means only), mirroring
        :meth:`dividend_values_batch`.

        Returns:
            ``(interactions, Phi)`` with ``Phi`` of shape ``(len(xs), len(interactions))``.
        """
        S_list: Optional[List[Tuple[int, ...]]] = None
        rows = []
        for x in xs:
            S_list, phi = self.faith_shap_values(
                x, pi, order_max, interactions=interactions, order=order, eta=eta,
                indices=indices, calc_covar=False, use_double=use_double, order_cap=order_cap,
            )
            rows.append(phi)
        return S_list, torch.stack(rows, dim=0)

    def global_faith_shap_summary(
        self,
        xs: Sequence[Union[str, torch.Tensor]],
        pi: torch.Tensor,
        order_max: int,
        interactions: Optional[Sequence[Sequence[int]]] = None,
        order: Optional[int] = 2,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        use_double: bool = True,
        order_cap: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, ...]], Dict[str, torch.Tensor]]:
        """Global (dataset-level) Faith-Shap summary over a target sample.

        Reduces ``E_S(x)`` over the targets ``xs`` into per-tuple quantities, mirroring
        :meth:`global_dividend_summary`:

        * ``mean_abs``    -- ``mean_x |E_S(x)|``: global importance of interaction ``S``.
        * ``mean_signed`` -- ``mean_x E_S(x)``: signed diagnostic.
        * ``var``         -- ``Var_x E_S(x)``: dispersion of the Faith-Shap value across
          targets. (Unlike the dividend's ``sobol_unnorm``, this is NOT a clean Sobol index
          when ``ell < L`` -- truncated Faith-Shap mixes ANOVA orders -- so it is reported as
          a plain variance, not labelled Sobol.)

        Args:
            xs, pi, order_max, interactions, order, eta, indices, use_double: as in
                :meth:`faith_shap_values_batch`.
            order_cap: highest dividend order to include in the Faith-Shap tail sum,
                forwarded to :meth:`faith_shap_values` for every target (default ``L``,
                i.e. exact). On long sequences with ``ell < L`` the exact tail runs over
                ~``2^(L - ell)`` supersets and is intractable; pass a low ``order_cap``
                (e.g. ``order_max + 1``) for a tractable, truncated approximation. Only
                affects the ``ell < L`` case -- when ``order_max >= L`` Faith-Shap equals
                the dividends and the tail is empty regardless.

        Returns:
            ``(interactions, summary)`` with ``summary`` a dict of ``(n,)`` tensors.
        """
        S_list, Phi = self.faith_shap_values_batch(
            xs, pi, order_max, interactions=interactions, order=order, eta=eta,
            indices=indices, use_double=use_double, order_cap=order_cap,
        )
        summary: Dict[str, torch.Tensor] = {
            "mean_abs": Phi.abs().mean(dim=0),
            "mean_signed": Phi.mean(dim=0),
            "var": Phi.var(dim=0, unbiased=False),
        }
        return S_list, summary

    # ------------------------------------------------------------------ #
    # Public: extensions (specified, not yet implemented)
    # ------------------------------------------------------------------ #

    # def shap_region(self, pi_star, pi, eta: float = 1.0, indices=None, calc_covar: bool = False):
    #     """Region SHAP (notes, Section 3.4 'Extension to regions'). NOT YET IMPLEMENTED.

    #     Explain a region described by a product distribution ``pi_star`` against a
    #     background ``pi``. Generalization: the fixed-channel ``a^p_{x_p, .}`` is replaced
    #     by the ``pi_star``-averaged channel ``sum_c pi_star^p_c a^p_{c, .}`` (another
    #     ``compute_zeta_pc`` call with ``pi_star``). Everything else -- the polynomial
    #     layer, the ``w``-bookkeeping, ``apply_R`` -- is unchanged, so this is a small
    #     edit to ``_mean_channels`` / ``_cov_scalars`` (swap the 'fixed' definition) once
    #     the point-target path is verified.
    #     """
    #     raise NotImplementedError("Region SHAP is specified but not implemented in v1.")

    # =====================================================================


    # ------------------------------------------------------------------ #
    # Vectorized (batched-over-targets) global builders
    # ------------------------------------------------------------------ #
    #
    # Motivation: the per-target loops in *_batch / global_* are launch-bound --
    # each target rebuilds the SAME shared pi-channel c0 and fires many tiny GPU
    # kernels. But c0 is target-INDEPENDENT and only the fixed-channel c1 depends
    # on x, so a whole batch of targets can be pushed through the polynomial layer
    # at once (the primitives broadcast over a leading B axis). This turns
    # ~B tiny passes into one big pass. Verified equivalent to the per-target
    # methods (probe_batch_indices.py) to ~1e-12.

    def _auto_batch_size(self, extra_deg: int = 1) -> int:
        """Pick a target-batch size B from free memory and the (B, L, t) footprint.

        Peak memory of a batched build is dominated by coefficient tensors of shape
        ``(B, t, <=L+1)`` in the working dtype. We size B so a conservative fraction
        of the free GPU memory holds a few such tensors at once, then clamp to a sane
        range. On CPU (no reliable free-memory query) we fall back to a fixed modest
        chunk. ``extra_deg`` pads the per-target degree budget for the SII/first-order
        coefficient arrays (dividends need only degree 1).

        This is a closed-form estimate, not a guess: the footprint is exactly
        ``B * t * (L + extra_deg) * itemsize`` per live coefficient tensor, and we
        budget for ~4 of them.
        """
        L, t = self.seq_length, self.ktc._train_x.reshape(-1, self.seq_length, self.n_alleles).shape[0]
        itemsize = torch.tensor([], dtype=self.ktc.comp_dtype).element_size()
        per_target = 4 * t * (L + extra_deg) * itemsize     # ~4 live coeff tensors
        dev = self.device
        is_cuda = str(dev).startswith("cuda") or (hasattr(dev, "type") and dev.type == "cuda")
        if is_cuda:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(dev)
            except Exception:  # noqa: BLE001
                free_bytes = 2 * (1024 ** 3)                # 2 GB fallback
            budget = 0.25 * free_bytes                       # conservative headroom
            B = int(max(1, budget // max(1, per_target)))
        else:
            B = 1024                                         # fixed modest CPU chunk
        return int(min(max(B, 64), 20000))                   # clamp to [64, 20000]

    def _batched_channels(self, xs, pi, eta: float):
        """Batched (c0, c1) for a list of targets. c0 is (L, t) [shared]; c1 is (B, L, t).

        Mirrors :meth:`_mean_channels` but builds the fixed-channel for the whole
        batch in one einsum. Verified bit-identical to stacking per-target
        :meth:`_mean_channels` (probe_batch_indices.py).
        """
        L, A = self.seq_length, self.n_alleles
        comp = self.ktc.comp_dtype
        Xtr = self.ktc._train_x.reshape(-1, L, A).to(comp)                       # (t, L, A)
        pi_b = pi.to(device=self.device, dtype=comp).unsqueeze(0)                # (1, L, A)
        zeta = self.ktc.compute_zeta_pc(pi_b, eta)[0]                            # (L, A)
        c0 = torch.einsum("la,tla->lt", zeta, Xtr)                               # (L, t) shared
        site = self.ktc.site_kernels.to(comp)                                    # (L, A, A)
        Xoh = torch.stack([self.target_to_onehot(x).to(comp) for x in xs])       # (B, L, A)
        Ax = torch.einsum("blc,lca->bla", Xoh, site)                             # (B, L, A)
        c1 = torch.einsum("bla,tla->blt", Ax, Xtr)                               # (B, L, t)
        return c0, c1

    def _batched_index_rows(self, c0, c1, interactions, weight_fn):
        """Batched mK rows for one chunk: returns (B, n_tuples, t).

        The batched analogue of :meth:`_shap_z_interaction`: every position in a
        tuple T takes the degree-0 factor (c1-c0) (shape (B, t, 1)), every other
        position the degree-1 factor [c0, c1] (broadcast to (B, t, 2)); the product
        over positions and the R-contraction run with a leading B axis untouched.
        """
        L = self.seq_length
        B, t = c1.shape[0], c1.shape[2]
        c0b = c0.unsqueeze(0).expand(B, L, t)                                    # (B, L, t)
        rows = []
        for T in interactions:
            Tset = set(T)
            g_k = weight_fn(len(Tset)).to(c1.dtype)
            factors = []
            for p in range(L):
                if p in Tset:
                    factors.append((c1[:, p, :] - c0b[:, p, :]).unsqueeze(-1))   # (B, t, 1)
                else:
                    factors.append(torch.stack([c0b[:, p, :], c1[:, p, :]], -1)) # (B, t, 2)
            coeffs = poly_prod_over_positions(factors)                           # (B, t, L-|T|+1)
            rows.append(apply_R(coeffs, g_k))                                    # (B, t)
        return torch.stack(rows, dim=1)                                          # (B, n, t)

    def dividend_values_batch_vectorized(
        self, xs, pi, interactions=None, order=2, eta: float = 1.0,
        indices=None, use_double: bool = True, batch_size=None,
    ):
        """Vectorized-over-targets dividends. Returns ``(interactions, A)`` with
        ``A`` of shape ``(len(xs), n_tuples)`` -- identical to
        :meth:`dividend_values_batch` but computed in target-chunks that share the
        channel build. Verified equal to the per-target path (~1e-12).
        """
        if use_double:
            self.ktc.set_precision(use_double=True)
        comp = self.ktc.comp_dtype
        pi = pi.to(device=self.device, dtype=comp)
        self.ktc.validate_pis(pi)
        T_list = self._resolve_interactions(interactions, order, indices)
        wfn = self._index_weight_fn("dividend")
        Qy = self.ktc.Qy.to(comp)
        B = batch_size or self._auto_batch_size(extra_deg=1)
        out = []
        for s in range(0, len(xs), B):
            chunk = xs[s:s + B]
            c0, c1 = self._batched_channels(chunk, pi, eta)
            mK = self._batched_index_rows(c0, c1, T_list, wfn)                   # (b, n, t)
            out.append(mK @ Qy)                                                  # (b, n)
        return T_list, torch.cat(out, dim=0)

    def interaction_values_batch_vectorized(
        self, xs, pi, interactions=None, order=2, eta: float = 1.0,
        indices=None, normalize: str = "sii", use_double: bool = True, batch_size=None,
    ):
        """Vectorized-over-targets SII (or 'shap'-normalized) interaction values.
        Returns ``(interactions, Phi)`` with ``Phi`` of shape ``(len(xs), n_tuples)``,
        matching a per-target :meth:`shap_interaction_values` loop (~1e-12).
        """
        if normalize not in ("shap", "sii"):
            raise ValueError("normalize must be 'shap' or 'sii'.")
        if use_double:
            self.ktc.set_precision(use_double=True)
        comp = self.ktc.comp_dtype
        pi = pi.to(device=self.device, dtype=comp)
        self.ktc.validate_pis(pi)
        T_list = self._resolve_interactions(interactions, order, indices)
        wfn = self._g_for_order  # SII weight
        if normalize == "shap":
            scale = torch.tensor([1.0 / factorial(len(T)) for T in T_list],
                                 device=self.device, dtype=comp)
        else:
            scale = torch.ones(len(T_list), device=self.device, dtype=comp)
        Qy = self.ktc.Qy.to(comp)
        B = batch_size or self._auto_batch_size(extra_deg=1)
        out = []
        for s in range(0, len(xs), B):
            chunk = xs[s:s + B]
            c0, c1 = self._batched_channels(chunk, pi, eta)
            mK = self._batched_index_rows(c0, c1, T_list, wfn)                   # (b, n, t)
            phi = mK @ Qy                                                        # (b, n)
            out.append(scale.unsqueeze(0) * phi)
        return T_list, torch.cat(out, dim=0)

    def shap_values_batch_vectorized(
        self, xs, pi, eta: float = 1.0, indices=None,
        use_double: bool = True, batch_size=None,
    ):
        """Vectorized-over-targets first-order SHAP values. Returns a tensor of shape
        ``(len(xs), len(indices))``, matching :meth:`shap_values_batch` (~1e-12).

        Uses the batched channels + the first-order prefix/suffix trick, generalized
        to a leading B axis.
        """
        if use_double:
            self.ktc.set_precision(use_double=True)
        comp = self.ktc.comp_dtype
        pi = pi.to(device=self.device, dtype=comp)
        self.ktc.validate_pis(pi)
        L = self.seq_length
        idx = list(range(L)) if indices is None else [int(i) for i in indices]
        g = self.g.to(comp)
        Qy = self.ktc.Qy.to(comp)
        B = batch_size or self._auto_batch_size(extra_deg=1)
        out = []
        for s in range(0, len(xs), B):
            chunk = xs[s:s + B]
            c0, c1 = self._batched_channels(chunk, pi, eta)                      # (L,t),(B,L,t)
            b, t = c1.shape[0], c1.shape[2]
            c0b = c0.unsqueeze(0).expand(b, L, t)                                # (B,L,t)
            factors = torch.stack([c0b, c1], dim=-1)                             # (B,L,t,2)
            ones = c1.new_ones(b, t, 1)
            prefix = [ones] * L
            for p in range(1, L):
                prefix[p] = poly_prod_over_positions([prefix[p - 1], factors[:, p - 1]])
            suffix = [ones] * L
            for p in range(L - 2, -1, -1):
                suffix[p] = poly_prod_over_positions([suffix[p + 1], factors[:, p + 1]])
            cols = []
            for i in idx:
                prod_not_i = poly_prod_over_positions([prefix[i], suffix[i]])    # (B,t,<=L)
                R_val = apply_R(prod_not_i, g)                                   # (B,t)
                mK_i = (c1[:, i, :] - c0b[:, i, :]) * R_val                      # (B,t)
                cols.append(mK_i @ Qy)                                           # (B,)
            out.append(torch.stack(cols, dim=1))                                 # (B, len(idx))
        return torch.cat(out, dim=0)

    def faith_shap_values_batch_vectorized(
        self, xs, pi, order_max, interactions=None, order=2, eta: float = 1.0,
        indices=None, use_double: bool = True, order_cap=None, batch_size=None,
    ):
        """Vectorized-over-targets truncated Faith-Shap E_S(x).

        Returns ``(S_list, Phi)`` with ``Phi`` of shape ``(len(xs), len(S_list))`` --
        identical to a per-target :meth:`faith_shap_values` loop, but the dividends it
        rests on are computed with :meth:`dividend_values_batch_vectorized` (shared
        channel build across the target chunk) and the target-INDEPENDENT Eq. 16
        redistribution map ``M`` is applied once as a matmul.

        Why this is safe / correct
        --------------------------
        Faith-Shap is ``E_S = a(S) + sum_{T superset S, ell<|T|<=cap} coeff(|S|,|T|,ell) a(T)``
        -- a fixed linear map ``M`` on the dividend vector ``a``. ``M`` and the set of
        dividend tuples it needs depend only on (reported tuples, ell, cap), NOT on the
        target, so they are built once. The ``needed``-set enumeration and ``M``
        construction below are the SAME code as :meth:`faith_shap_values` (reproduced,
        not refactored, to keep that verified method untouched); the batch dividends are
        the verified vectorized path. probe_faith_batch.py checks both the per-target
        equivalence and the ``order_max == L`` -> dividends anchor.

        Only the ell < L case is meaningfully different from dividends; at ell >= L the
        tail is empty, ``M`` is the identity, and this returns the (vectorized) dividends.
        """
        L = self.seq_length
        ell = int(order_max)
        if ell < 1 or ell > L:
            raise ValueError(f"order_max (ell) must be in [1, {L}]; got {order_max}.")
        cap = L if order_cap is None else int(order_cap)
        if cap < ell or cap > L:
            raise ValueError(f"order_cap must be in [ell, L] = [{ell}, {L}]; got {order_cap}.")

        # Reported tuples S (each |S| <= ell) -- identical resolution to faith_shap_values.
        S_list = self._resolve_interactions(interactions, order, indices)
        if any(len(S) > ell for S in S_list):
            bad = [S for S in S_list if len(S) > ell]
            raise ValueError(f"Faith-Shap reports tuples of size <= order_max={ell}; "
                             f"got larger tuples {bad}.")

        idx_universe = list(range(L)) if indices is None else [int(i) for i in indices]

        # ---- needed dividend tuples: reported S's + supersets T (ell<|T|<=cap) ----
        # (verbatim from faith_shap_values so div_tuples order matches the M columns)
        import itertools as _it
        needed = set(S_list)
        for S in S_list:
            rest = [p for p in idx_universe if p not in set(S)]
            for extra_k in range(1, len(rest) + 1):
                for extra in _it.combinations(rest, extra_k):
                    T = tuple(sorted(set(S) | set(extra)))
                    if ell < len(T) <= cap:
                        needed.add(T)
        div_tuples = sorted(needed, key=lambda T: (len(T), T))
        div_index = {T: i for i, T in enumerate(div_tuples)}

        # ---- target-independent Eq. 16 map M: (n_S, n_div) ----
        comp = self.ktc.comp_dtype
        n = len(S_list)
        M = torch.zeros(n, len(div_tuples), device=self.device, dtype=comp)
        for r, S in enumerate(S_list):
            Sset = set(S); s = len(S)
            M[r, div_index[S]] += 1.0                      # identity a(S) term
            for T in div_tuples:
                if len(T) > ell and Sset.issubset(T):
                    M[r, div_index[T]] += self._faith_shap_coeff(s, len(T), ell)

        # ---- batched dividends a(T)(x) for ALL needed tuples, then Phi = A @ M^T ----
        _, A = self.dividend_values_batch_vectorized(
            xs, pi, interactions=div_tuples, eta=eta,
            use_double=use_double, batch_size=batch_size,
        )                                                  # (n_x, n_div)
        Phi = A @ M.T                                      # (n_x, n_S)
        return S_list, Phi

def make_background(
    mode: str,
    seq_length: int,
    alphabet: str,
    wildtype: Optional[Union[str, torch.Tensor]] = None,
    empirical_counts: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a background distribution ``pi`` (row-stochastic ``(L, A)``) for SHAP.

    The background is the reference every attribution is measured *against*: a dividend or
    Faith-Shap value explains the target relative to ``E_pi[f]``. The choice is not cosmetic
    -- it defines the baseline -- so it is made explicit here rather than defaulted silently.

    Modes:
      * ``'uniform'``   -- ``pi[p, a] = 1/A`` at every position. The "average background":
        each residue is equally likely. Matches the synthetic validation suite, and gives the
        functional-ANOVA / Sobol reading for dividends (product background). Use when you want
        "contribution relative to a random sequence".
      * ``'wildtype'``  -- ``pi[p, :]`` is the one-hot of the wild-type residue at ``p``. The
        "wild-type background": attributions are deviations from the WT sequence, i.e. the
        effect of *mutating away from* wild type. Requires ``wildtype``. NOTE: a one-hot (delta)
        background is a valid row-stochastic ``pi``; conditional expectations then collapse to
        substituting WT at the averaged-over positions, which is exactly the WT-referenced game.
      * ``'empirical'`` -- ``pi[p, a]`` proportional to the per-position residue frequencies in
        ``empirical_counts`` (shape ``(L, A)``). The "dataset background": residues as common as
        they are in the data. Use when the natural reference is the observed sequence ensemble
        rather than a uniform or single-sequence baseline.

    Args:
        mode: ``'uniform'``, ``'wildtype'``, or ``'empirical'``.
        seq_length, alphabet: define ``L`` and the ``A`` symbols (index = position in the string).
        wildtype: required for ``'wildtype'`` -- an ``L``-char string or ``(L, A)`` one-hot.
        empirical_counts: required for ``'empirical'`` -- non-negative ``(L, A)`` counts/freqs;
            each row is renormalized to sum to 1.
        device, dtype: placement / precision of the returned tensor.

    Returns:
        ``pi`` of shape ``(L, A)``, each row summing to 1.
    """
    A = len(alphabet)
    L = int(seq_length)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    m = mode.strip().lower()

    if m == "uniform":
        return torch.full((L, A), 1.0 / A, device=dev, dtype=dtype)

    if m == "wildtype":
        if wildtype is None:
            raise ValueError("mode='wildtype' requires `wildtype` (string or (L, A) one-hot).")
        if isinstance(wildtype, str):
            if len(wildtype) != L:
                raise ValueError(f"wildtype string length {len(wildtype)} != seq_length {L}.")
            sym_to_idx = {c: i for i, c in enumerate(alphabet)}
            pi = torch.zeros((L, A), device=dev, dtype=dtype)
            for p, ch in enumerate(wildtype):
                if ch not in sym_to_idx:
                    raise ValueError(f"wildtype char '{ch}' at position {p} not in alphabet.")
                pi[p, sym_to_idx[ch]] = 1.0
            return pi
        wt = torch.as_tensor(wildtype, device=dev, dtype=dtype)
        if wt.shape != (L, A):
            raise ValueError(f"wildtype one-hot must be ({L}, {A}); got {tuple(wt.shape)}.")
        return wt

    if m == "empirical":
        if empirical_counts is None:
            raise ValueError("mode='empirical' requires `empirical_counts` of shape (L, A).")
        c = torch.as_tensor(empirical_counts, device=dev, dtype=dtype)
        if c.shape != (L, A):
            raise ValueError(f"empirical_counts must be ({L}, {A}); got {tuple(c.shape)}.")
        if (c < 0).any():
            raise ValueError("empirical_counts must be non-negative.")
        row = c.sum(dim=1, keepdim=True)
        if (row == 0).any():
            raise ValueError("empirical_counts has an all-zero row; cannot normalize.")
        return c / row

    raise ValueError(f"unknown background mode '{mode}' (use 'uniform', 'wildtype', 'empirical').")


__all__ = [
    "ShapKernelTrickComputer",
    "make_background",
]