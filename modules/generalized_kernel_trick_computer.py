"""
generalized_kernel_trick_computer.py
====================================

Compute **SHAP values** (with their full posterior uncertainty) for a
sequence-function map learned by a factorized-kernel Gaussian process, using the
*generalized kernel trick* (GKT).

This module is the SHAP counterpart to ``kernel_trick_computer_refactored.py`` and
is used the same way on genotype->phenotype data: build a computer from a fitted
EpiK GP model, then ask it for the SHAP attribution of a target sequence against a
background distribution ``pi``.

    from kernel_trick_computer_refactored import KernelTrickComputer
    from generalized_kernel_trick_computer import ShapKernelTrickComputer

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
* Section A -- standalone polynomial primitives (pure functions, model-independent):
  ``g_weights``, ``poly_prod_over_positions`` (+ bivariate), ``apply_R`` /
  ``apply_RxR``. This "polynomial layer" is the entire conceptual addition over the
  ordinary kernel trick, and can be unit-tested in isolation.
* Section B -- ``ShapKernelTrickComputer``, which *composes* a
  ``KernelTrickComputer`` (all kernel / training / posterior state is read from it)
  and adds the SHAP-specific factor builders and the public API (first-order values,
  and pairwise / higher-order interaction values).

The independent brute-force reference used to verify these paths now lives in its own
module, ``shap_brute_force.ShapBruteForce`` (moved out of this file). It recomputes
every SHAP quantity -- first-order and interaction -- by direct subset enumeration,
without the polynomial layer, so agreement is genuine evidence of correctness. Build
it from the same ``ktc`` (or straight from a ``ShapKernelTrickComputer``)::

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
"""

from __future__ import annotations

import itertools
from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch


# =============================================================================
# SECTION A -- Polynomial primitives (pure, model-independent functions)
# -----------------------------------------------------------------------------
# The "polynomial layer" that replaces the scalar ``torch.prod(..., dim=positions)``
# of the ordinary kernel trick. Each per-position factor is a coefficient vector in
# the bookkeeping variable(s), so the reduction over positions is a convolution.
# =============================================================================


def g_weights(L: int, k: int = 1, device=None, dtype=torch.float64) -> torch.Tensor:
    """Build the (interaction) SHAP weight vector ``g_k(s) = s! (L - k - s)! / (L - k + 1)!``.

    This is the vector the linear functional ``R`` contracts against: after the
    per-position polynomial product yields a coefficient vector indexed by the power
    ``s`` of the bookkeeping variable (i.e. by ``|S|``), ``R`` maps ``w^s -> g_k(s)``
    and sums, so ``R(coeffs) = coeffs . g_k``.

    ``k`` is the interaction order (the size of the interaction set ``T``). It is the
    Shapley Interaction Index weight (Grabisch-Roubens), which at ``k == 1`` reduces to
    the ordinary first-order Shapley weight ``s! (L - s - 1)! / L!`` used by
    :meth:`ShapKernelTrickComputer.shap_values`. Higher ``k`` fixes ``k`` positions in
    ``T``, so the largest possible ``|S|`` is ``L - k`` and the vector has length
    ``L - k + 1``.

    Args:
        L: sequence length (== number of ML features).
        k: interaction order (default 1 == first-order Shapley).
        device, dtype: placement of the returned tensor. float64 by default for
            numerical safety (the factorials cancel to small magnitudes).

    Returns:
        Tensor of shape ``(L - k + 1,)`` with ``g_k[s] = s! (L - k - s)! / (L - k + 1)!``.

    Note:
        Uses lgamma for stability:
        ``log g_k(s) = lgamma(s+1) + lgamma(L-k-s+1) - lgamma(L-k+2)``.
        Useful test identity: for any fixed size-``k`` set ``T``,
        ``sum_s C(L-k, s) g_k(s) == 1``.
    """
    if k < 1 or k > L:
        raise ValueError(f"interaction order k must satisfy 1 <= k <= L (got k={k}, L={L}).")
    s = torch.arange(L - k + 1, device=device, dtype=dtype)
    log_g = (
        torch.lgamma(s + 1)
        + torch.lgamma(L - k - s + 1)
        - torch.lgamma(torch.tensor(L - k + 2.0, dtype=dtype))
    )
    return torch.exp(log_g)


def _poly_mul_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convolve two coefficient tensors along their last (degree) axis.

    Computes the polynomial product ``out(w) = a(w) * b(w)`` where the last axis of
    each input holds coefficients ``[c0, c1, ...]`` of ``w^0, w^1, ...``. All leading
    axes are treated as batch and broadcast together:

        out[..., k] = sum_{i + j = k} a[..., i] * b[..., j].

    Degrees are tiny (each per-position factor is degree <= 1), so the loop over
    ``b``'s degree axis is cheap; clarity beats cleverness here.

    Args:
        a: ``(..., da)`` coefficient tensor.
        b: ``(..., db)`` coefficient tensor.

    Returns:
        ``(..., da + db - 1)`` coefficient tensor of the product.
    """
    la, lb = a.shape[-1], b.shape[-1]
    batch = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    out = a.new_zeros(*batch, la + lb - 1)
    for j in range(lb):
        out[..., j:j + la] = out[..., j:j + la] + a * b[..., j:j + 1]
    return out


def poly_prod_over_positions(factors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Univariate polynomial product across all ``L`` positions (the DP step).

    The direct analog of ``torch.prod(out, dim=<position>)`` in the ordinary
    ``compute_z`` -- but each per-position "value" is a coefficient vector in ``w``,
    so the reduction is a convolution rather than a scalar product.

    Args:
        factors: length-``L`` sequence; ``factors[p]`` has shape ``(..., deg_p + 1)``
            and holds the coefficients of the position-``p`` factor. For the SHAP
            mean, the target position contributes degree 0 (length 1) and every other
            position degree 1 (length 2), so the result has degree <= ``L-1``.

    Returns:
        ``(..., total_deg + 1)`` coefficient tensor of the full product.
    """
    acc = factors[0]
    for f in factors[1:]:
        acc = _poly_mul_1d(acc, f)
    return acc


def apply_R(coeffs: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Apply the linear functional ``R``: contract coefficients with ``g``.

    ``R`` replaces ``w^s`` by ``g(s)`` and sums, i.e. a dot product of the
    coefficient vector with ``g`` (truncated to the coefficient length).

    Args:
        coeffs: ``(..., D)`` coefficient tensor, ``D <= len(g)``.
        g: ``(L,)`` SHAP weight vector from :func:`g_weights`.

    Returns:
        ``(...,)`` tensor -- the scalar value of ``R`` applied along the last axis.
    """
    D = coeffs.shape[-1]
    return torch.einsum("...d,d->...", coeffs, g[:D].to(coeffs.dtype))


def _poly_mul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convolve two *bivariate* coefficient tensors along the last two axes.

    Used by the covariance path, where each position carries a factor in two
    bookkeeping variables (``w`` for SHAP index ``i``, ``w'`` for index ``j``).
    Coefficient layout is ``[..., power_of_w, power_of_w']``:

        out[..., k, l] = sum_{i1+j1=k, i2+j2=l} a[..., i1, i2] * b[..., j1, j2].

    Per-position degrees are <= 1 in each variable, so the double loop is cheap.

    Args:
        a: ``(..., da_w, da_wp)``.
        b: ``(..., db_w, db_wp)``.

    Returns:
        ``(..., da_w + db_w - 1, da_wp + db_wp - 1)``.
    """
    daw, dawp = a.shape[-2], a.shape[-1]
    dbw, dbwp = b.shape[-2], b.shape[-1]
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    out = a.new_zeros(*batch, daw + dbw - 1, dawp + dbwp - 1)
    for j1 in range(dbw):
        for j2 in range(dbwp):
            out[..., j1:j1 + daw, j2:j2 + dawp] = (
                out[..., j1:j1 + daw, j2:j2 + dawp] + a * b[..., j1:j1 + 1, j2:j2 + 1]
            )
    return out


def poly_prod_over_positions_bivariate(factors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Bivariate polynomial product across positions (the covariance DP step)."""
    acc = factors[0]
    for f in factors[1:]:
        acc = _poly_mul_2d(acc, f)
    return acc


def apply_RxR(coeffs2: torch.Tensor, g: torch.Tensor, g2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply ``R x R``: contract a bivariate coefficient tensor with ``g_a (x) g_b``.

        (R x R)( sum_{a,b} c_{a,b} w^a w'^b ) = sum_{a,b} c_{a,b} g_a(a) g_b(b).

    The two axes may carry different weight vectors -- needed when the two sides are
    interactions of different orders (the ``w`` axis uses ``g_{k}`` for the row tuple,
    the ``w'`` axis uses ``g_{k'}`` for the column tuple). For the common same-order
    case, pass a single ``g`` and it is used on both axes.

    Args:
        coeffs2: ``(..., Da, Db)`` bivariate coefficients.
        g: weight vector for the ``w`` (row / first) axis.
        g2: weight vector for the ``w'`` (column / second) axis; defaults to ``g``.

    Returns:
        ``(...,)`` tensor.
    """
    if g2 is None:
        g2 = g
    Da, Db = coeffs2.shape[-2], coeffs2.shape[-1]
    return torch.einsum("...ab,a,b->...", coeffs2, g[:Da].to(coeffs2.dtype), g2[:Db].to(coeffs2.dtype))


# =============================================================================
# SECTION B -- The SHAP computer
# =============================================================================


class ShapKernelTrickComputer:
    """Compute SHAP values (and posterior covariance) via the generalized kernel trick.

    A composition wrapper around a ``KernelTrickComputer`` (``self.ktc``): all kernel,
    training-data, and posterior quantities are read from ``self.ktc``, and this class
    adds only the polynomial layer (Section A) plus the SHAP-specific factor builders
    and assembly. Composing rather than subclassing keeps the tested kernel-trick
    implementation untouched and makes the reuse boundary explicit (every reused
    quantity is an attribute access on ``self.ktc``).

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
    ) -> torch.Tensor:
        """Build ``mK = m^T K_{*,X}`` for each interaction tuple ``T`` (raw SII weights).

        Generalizes :meth:`_shap_z_mean` from a single index ``i`` to a set ``T``: every
        position ``p in T`` takes the degree-0 difference factor ``[c1[p] - c0[p]]`` and
        every other position the shared degree-1 factor ``[c0[p], c1[p]]``. Their
        polynomial product over ``p`` is a ``(t, L - |T| + 1)`` coefficient tensor, which
        ``apply_R`` contracts with the order-``|T|`` weight ``g_{|T|}``.

        This is the position-factorized discrete derivative ``Delta_T f_x(S)`` summed
        against ``g_{|T|}(|S|)`` over all ``S`` disjoint from ``T`` -- exactly the
        interaction analog of the first-order row.

        Returns:
            ``mK`` of shape ``(len(interactions), t)``; these are the *unnormalized* SII
            rows (the public method rescales them per the chosen normalization).
        """
        L = self.seq_length
        c0, c1 = self._mean_channels(x_onehot, pi, eta)          # (L, t), (L, t)
        rows = []
        for T in interactions:
            Tset = set(T)
            g_k = self._g_for_order(len(Tset)).to(c0.dtype)
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
    ) -> torch.Tensor:
        """Build ``R[a, b] = m^{T_a} K m^{T_b}`` via the bivariate GKT (raw SII weights).

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
        weight vectors ``g_{|T_a|}`` (rows) and ``g_{|T_b|}`` (columns) -- they differ
        when the tuples have different sizes.

        Returns:
            ``R`` of shape ``(len(interactions), len(interactions))`` (unnormalized SII).
        """
        L = self.seq_length
        sc = self._cov_scalars(x_onehot, pi, eta)
        K_ff, K_fa, K_aa = sc["K_ff"], sc["K_fa"], sc["K_aa"]     # each (L,)

        # per-order weight vectors in the working dtype, cached across the (a, b) loop
        g_by_order: Dict[int, torch.Tensor] = {}

        def gord(k: int) -> torch.Tensor:
            g = g_by_order.get(k)
            if g is None:
                g = self._g_for_order(k).to(K_ff.dtype)
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


__all__ = [
    "ShapKernelTrickComputer",
    "g_weights",
    "poly_prod_over_positions",
    "poly_prod_over_positions_bivariate",
    "apply_R",
    "apply_RxR",
]