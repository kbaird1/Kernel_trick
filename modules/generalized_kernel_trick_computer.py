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
  and adds the SHAP-specific factor builders, the public API, and an independent
  brute-force reference used for verification.

The mean and covariance paths are implemented and verified against the brute-force
reference (see ``bf_*`` methods and ``test_gkt_brute.py``).


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
* ``shap_interaction_values``: higher-order interaction SHAP (pairwise phi_ij and
  beyond -- 3-way, 4-way, ...). The GKT machinery generalizes directly to any order;
  what is NOT yet decided is *which* interaction index to adopt (Shapley Interaction
  Index, Shapley-Taylor, or Faith-Shap), since they differ in their axioms and weight
  vectors. See the method docstring for the trade-offs. Decision pending.
* ``shap_region``: region SHAP against a product distribution -- specified, not yet
  implemented.
* An ML/econ-notion convenience switch (see conventions above).
"""

from __future__ import annotations

import itertools
from math import exp, lgamma
from typing import Optional, Sequence, Tuple, Union

import torch


# =============================================================================
# SECTION A -- Polynomial primitives (pure, model-independent functions)
# -----------------------------------------------------------------------------
# The "polynomial layer" that replaces the scalar ``torch.prod(..., dim=positions)``
# of the ordinary kernel trick. Each per-position factor is a coefficient vector in
# the bookkeeping variable(s), so the reduction over positions is a convolution.
# =============================================================================


def g_weights(L: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """Build the SHAP weight vector ``g(s) = s! (L - s - 1)! / L!``.

    This is the vector the linear functional ``R`` contracts against: after the
    per-position polynomial product yields a coefficient vector indexed by the power
    ``s`` of the bookkeeping variable (i.e. by ``|S|``), ``R`` maps ``w^s -> g(s)``
    and sums, so ``R(coeffs) = coeffs . g``.

    Args:
        L: sequence length (== number of ML features). The vector has length ``L``,
            covering ``s = 0, 1, ..., L-1`` (the largest possible ``|S|`` with ``i``
            fixed).
        device, dtype: placement of the returned tensor. float64 by default for
            numerical safety (the factorials cancel to O(1/L) magnitudes).

    Returns:
        Tensor of shape ``(L,)`` with ``g[s] = s! (L - s - 1)! / L!``.

    Note:
        Uses lgamma for stability: ``log g(s) = lgamma(s+1) + lgamma(L-s) - lgamma(L+1)``.
        Useful test identity: for any fixed ``i``, ``sum_s C(L-1, s) g(s) == 1``.
    """
    s = torch.arange(L, device=device, dtype=dtype)
    log_g = torch.lgamma(s + 1) + torch.lgamma(L - s) - torch.lgamma(torch.tensor(L + 1.0, dtype=dtype))
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


def apply_RxR(coeffs2: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Apply ``R x R``: contract a bivariate coefficient tensor with ``g (x) g``.

        (R x R)( sum_{a,b} c_{a,b} w^a w'^b ) = sum_{a,b} c_{a,b} g(a) g(b).

    Args:
        coeffs2: ``(..., Da, Db)`` bivariate coefficients.
        g: ``(L,)`` SHAP weight vector.

    Returns:
        ``(...,)`` tensor.
    """
    Da, Db = coeffs2.shape[-2], coeffs2.shape[-1]
    return torch.einsum("...ab,a,b->...", coeffs2, g[:Da].to(coeffs2.dtype), g[:Db].to(coeffs2.dtype))


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
        self.g = g_weights(self.seq_length, device=self.device, dtype=g_dtype)

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

    # ================================================================== #
    # Brute-force reference (independent verification path)
    # ------------------------------------------------------------------ #
    # These recompute SHAP quantities by direct subset enumeration, WITHOUT the
    # polynomial layer, so the GKT path can be checked against an O((1+A)^L)
    # reference on small fixtures.
    #
    # Independence: every method below reads only the raw cached arrays off
    # self.ktc (site_kernels, _train_x, Qy) and enumerates -- none of them call
    # _shap_z_mean, _mean_channels, poly_prod_over_positions, apply_R, or any other
    # GKT machinery. Inputs are shared; computation paths are not, so agreement is
    # genuine evidence of correctness rather than a shared-bug coincidence.
    #
    # Targets are one-hot (L, A) here; use target_to_onehot upstream for a string.
    # ================================================================== #

    @staticmethod
    def _bf_g_weight(s: int, L: int) -> float:
        """SHAP weight ``g(s) = s! (L-s-1)! / L!`` (scalar, via lgamma).

        Deliberately a standalone scalar recomputation -- it does NOT read ``self.g``
        (built by the GKT side), so the two paths share no weight vector.
        """
        return exp(lgamma(s + 1) + lgamma(L - s) - lgamma(L + 1))

    def _bf_onehot_to_idx(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """Convert a ``(L, A)`` one-hot target to a length-``L`` LongTensor of indices."""
        assert x_onehot.shape == (self.seq_length, self.n_alleles), \
            f"expected one-hot (L, A)=({self.seq_length},{self.n_alleles})"
        return x_onehot.argmax(dim=-1).long()

    def bf_posterior_mean(self, z_idx: Sequence[int]) -> torch.Tensor:
        """Posterior mean ``f(z) = sum_t (prod_p a^p_{z_p, y_{t,p}}) Qy_t`` for one sequence.

        Rebuilt from the raw caches (site_kernels, _train_x, Qy) -- the same function
        the GKT mean path explains, but evaluated pointwise for a single sequence ``z``.
        """
        L, A = self.seq_length, self.n_alleles
        site = self.ktc.site_kernels                                # (L, A, A)
        qy = self.ktc.Qy                                            # (t,)
        Xtr = self.ktc._train_x.reshape(-1, L, A)
        train_idx = Xtr.argmax(dim=-1)                             # (t, L)

        z = torch.as_tensor(z_idx, device=self.device).long()
        k = torch.ones(train_idx.shape[0], dtype=qy.dtype, device=qy.device)
        for p in range(L):
            k = k * site[p, z[p], train_idx[:, p]].to(qy.dtype)    # a^p_{z_p, y_{t,p}}
        return (k * qy).sum()

    def bf_conditional_expectation(
        self, S: Sequence[int], x_idx: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """Conditional expectation ``f_x(S) = E_pi[ f | z_p = x_p for p in S ]``.

        Evaluated by enumerating all characters at the free (non-``S``) positions and
        weighting by the background ``pi``.
        """
        L, A = self.seq_length, self.n_alleles
        S = set(int(p) for p in S)
        free = [p for p in range(L) if p not in S]
        base = [int(c) for c in x_idx]
        total = self.bf_posterior_mean(base).new_zeros(())        # dtype-matched accumulator
        for combo in itertools.product(range(A), repeat=len(free)):
            z = list(base)
            w = 1.0
            for p, c in zip(free, combo):
                z[p] = c
                w *= float(pi[p, c])
            total = total + w * self.bf_posterior_mean(z)
        return total

    def bf_base_value(self, x_onehot: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """Base value ``phi_0 = f_x(empty set) = E_pi[f]``, by enumeration."""
        x_idx = self._bf_onehot_to_idx(x_onehot)
        pi = pi.to(device=self.device)
        return self.bf_conditional_expectation((), x_idx, pi)

    def bf_shap_values(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """First-order SHAP means by direct subset enumeration (no kernel trick).

            phi_i = sum_{S : i not in S} g(|S|) ( f_x(S u i) - f_x(S) ).

        Args:
            x_onehot: target as one-hot ``(L, A)``.
            pi: background distribution ``(L, A)``, row-stochastic.
            indices: SHAP positions; default all ``0..L-1``.

        Returns:
            ``phi`` of shape ``(len(indices),)``.
        """
        L, A = self.seq_length, self.n_alleles
        x_idx = self._bf_onehot_to_idx(x_onehot)
        pi = pi.to(device=self.device)
        if indices is None:
            indices = list(range(L))

        out = []
        for i in indices:
            rest = [p for p in range(L) if p != i]
            acc = self.bf_posterior_mean([int(c) for c in x_idx]).new_zeros(())
            for r in range(len(rest) + 1):
                gw = self._bf_g_weight(r, L)
                for S in itertools.combinations(rest, r):
                    fs = self.bf_conditional_expectation(S, x_idx, pi)
                    fsi = self.bf_conditional_expectation(tuple(S) + (i,), x_idx, pi)
                    acc = acc + gw * (fsi - fs)
            out.append(acc)
        return torch.stack(out)

    def bf_local_accuracy_residual(
        self, x_onehot: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """``f(x) - phi_0 - sum_i phi_i`` (should be ~0), computed entirely by brute force."""
        x_idx = self._bf_onehot_to_idx(x_onehot)
        pi = pi.to(device=self.device)
        fx = self.bf_posterior_mean([int(c) for c in x_idx])
        phi0 = self.bf_conditional_expectation((), x_idx, pi)
        phi = self.bf_shap_values(x_onehot, pi)
        return fx - phi0 - phi.sum()

    # ------------------------------------------------------------------ #
    # Brute-force covariance reference (Theorem-1 style, explicit rows)
    # ------------------------------------------------------------------ #
    # These build the SHAP linear-functional rows m^i EXPLICITLY over the full
    # sequence space (all A^L sequences) and assemble the posterior covariance the
    # same way the kernel trick does:
    #   cov = m^i K (m^j)^T - (m^i K_{*,X}) S S^T (m^j K_{*,X})^T,
    # where S == ktc.S == covar_cache is the cached root with S S^T = Q + (Qy)(Qy)^T.
    # No polynomial layer is used, so this independently checks _shap_R_cov + assembly.
    #
    # Cost is O(A^L) memory for the rows and O(A^{2L}) time for the full K m^j -- keep
    # L VERY small.

    def _bf_all_seqs_idx(self) -> torch.Tensor:
        """All ``A^L`` sequences as an ``(A^L, L)`` LongTensor of character indices."""
        L, A = self.seq_length, self.n_alleles
        return torch.tensor(list(itertools.product(range(A), repeat=L)),
                            dtype=torch.long, device=self.device)         # (A^L, L)

    def _bf_shap_row(
        self, i: int, x_idx: torch.Tensor, pi: torch.Tensor, all_idx: torch.Tensor
    ) -> torch.Tensor:
        """Explicit SHAP row ``m^i`` over all sequences ``z`` (see notes, section 3.3).

            m^i_z = sum_{S : i not in S} g(|S|) ( mS_z(with i fixed) - mS_z(without i) )
            mS_z  = prod_{p in S} delta_{x_p = z_p} * prod_{p not in S} pi^p_{z_p}.

        Returns:
            A vector of length ``A^L`` aligned with ``all_idx``.
        """
        L, A = self.seq_length, self.n_alleles
        dtype = pi.dtype
        N = all_idx.shape[0]
        x_idx = x_idx.to(self.device)
        pi = pi.to(self.device)

        # delta[p, n] = 1 if z_n[p] == x_p else 0 ;  piz[p, n] = pi[p, z_n[p]]
        z = all_idx                                                       # (N, L)
        delta = (z == x_idx.view(1, L)).to(dtype).T                       # (L, N)
        piz = pi[torch.arange(L).view(L, 1), z.T]                         # (L, N)

        rest = [p for p in range(L) if p != i]
        row = torch.zeros(N, dtype=dtype, device=self.device)
        for r in range(len(rest) + 1):
            gw = self._bf_g_weight(r, L)
            for S in itertools.combinations(rest, r):
                Sset = set(S)
                # mS_z with i FIXED (S u i) and without i (S) share the product over
                # p in S of delta and over p not in (S u i) of pi; they differ only at
                # position i (delta_i vs pi_i).
                base = torch.ones(N, dtype=dtype, device=self.device)
                for p in range(L):
                    if p in Sset:
                        base = base * delta[p]
                    elif p != i:
                        base = base * piz[p]
                # (delta_i - pi_i) * base == mS_z(S u i) - mS_z(S)
                row = row + gw * (delta[i] - piz[i]) * base
        return row                                                        # (N,)

    def _bf_full_kernel(self, idxA: torch.Tensor, idxB: torch.Tensor) -> torch.Tensor:
        """Product kernel between two sets of sequences (given as index tensors).

            K[a, b] = prod_p a^p_{ idxA[a,p], idxB[b,p] }, read from site_kernels.
        """
        L = self.seq_length
        site = self.ktc.site_kernels.to(self.ktc.comp_dtype)              # (L, A, A)
        Ka, Kb = idxA.shape[0], idxB.shape[0]
        K = torch.ones(Ka, Kb, dtype=site.dtype, device=self.device)
        for p in range(L):
            # site[p][idxA[:,p]][:, idxB[:,p]]
            K = K * site[p][idxA[:, p]][:, idxB[:, p]]
        return K                                                          # (Ka, Kb)

    def bf_shap_covariance(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Posterior covariance of the SHAP means by explicit Theorem-1 assembly.

            cov[i,j] = m^i K (m^j)^T - (m^i K_{*,X}) S S^T (m^j K_{*,X})^T.

        Built from explicit full-space rows ``m^i`` (no kernel trick), so it is
        independent of ``_shap_R_cov`` / ``shap_values(calc_covar=True)``. O(A^L)
        memory, O(A^{2L}) time -- for tiny ``L`` only.
        """
        L, A = self.seq_length, self.n_alleles
        comp = self.ktc.comp_dtype
        x_idx = self._bf_onehot_to_idx(x_onehot)
        pi = pi.to(device=self.device, dtype=comp)
        if indices is None:
            indices = list(range(L))

        all_idx = self._bf_all_seqs_idx()                                 # (A^L, L)
        train_idx = self.ktc._train_x.reshape(-1, L, A).argmax(dim=-1).to(self.device)  # (t, L)

        # explicit rows m^i over all sequences
        rows = torch.stack([self._bf_shap_row(i, x_idx, pi, all_idx) for i in indices])  # (n, A^L)

        # prior-kernel term:  m^i K (m^j)^T
        K_all = self._bf_full_kernel(all_idx, all_idx)                    # (A^L, A^L)
        R = rows @ K_all @ rows.T                                         # (n, n)

        # data term:  (m^i K_{*,X}) S S^T (m^j K_{*,X})^T
        K_sx = self._bf_full_kernel(all_idx, train_idx)                   # (A^L, t)
        mK = rows @ K_sx                                                  # (n, t)
        S = self.ktc.S.to(comp)                                          # (t, t) root
        mKS = mK @ S                                                      # (n, t)
        data = mKS @ mKS.T                                               # (n, n)

        return R - data

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
    # Public: extensions (specified, not yet implemented)
    # ------------------------------------------------------------------ #

    def shap_interaction_values(self, x, pi, eta: float = 1.0, calc_covar: bool = False):
        """Higher-order interaction SHAP (pairwise phi_ij and beyond). NOT YET IMPLEMENTED.

        Only first-order phi_i (``shap_values``) is implemented and verified so far.
        This method is intended to compute interaction attributions for a position set
        T = {i_1, ..., i_k}: pairwise (k=2) as in the background papers, and in principle
        3-way, 4-way, ... as well. The GKT extends to any order; the reason this is left
        open is a *modeling* choice (which interaction index), not a machinery gap.

        Shared machinery (identical for every index below)
        ---------------------------------------------------
        Every interaction index is built on the discrete k-th derivative of the set
        function f_x(S) = E_pi[ f | X_S = x_S ]:

            Delta_T f_x(S) = sum_{W subseteq T} (-1)^{|T|-|W|} f_x(S u W).

        For k=2 this is the familiar
            f_x(S u {i,j}) - f_x(S u i) - f_x(S u j) + f_x(S).
        Because S and T are disjoint, Delta_T factorizes over positions:

            Delta_T f_x(S)_z = prod_{p in T}       (delta_{x_p=z_p} - pi^p_{z_p})
                             * prod_{p in S}        delta_{x_p=z_p}
                             * prod_{p not in S u T} pi^p_{z_p}.

        This is exactly the position-factorizable form the GKT already exploits. In
        ``_shap_z_mean``, first order gives the single position i the degree-0 difference
        factor ``c1[i] - c0[i]`` and every other position the degree-1 factor
        ``[c0[p], c1[p]]``. For an order-k interaction, EVERY position in T gets the
        ``c1[p] - c0[p]`` difference factor; nothing else changes. So the interaction

            sum_{S : S cap T = empty} g_k(|S|) * Delta_T f_x(S)

        is one GKT run, and across the candidate indices ONLY the weight vector g_k(s)
        (plus, for some, a boundary term) differs.

        Open decision: which interaction index
        ---------------------------------------
        First-order SHAP is unique (a theorem); interactions are NOT, so different
        axioms yield different indices. The three standard choices, with trade-offs:

        * Shapley Interaction Index (SII; Grabisch-Roubens). Natural per-T recursion,
          symmetric across orders, weight
              g_SII_k(s) = s! (L - k - s)! / (L - k + 1)!.
          Downside: does NOT reconstruct the prediction (no additive / efficiency
          decomposition). NOTE: Lundberg tree-SHAP pairwise values are SII at k=2 (with
          a 1/2 symmetrization) -- i.e. the four-term formula above. Choose SII to stay
          numerically comparable with those papers.

        * Shapley-Taylor (STI; Sundararajan et al.). Fix a max order k: orders < k use
          the discrete-derivative value directly, order == k absorbs the remainder via a
          Shapley average. Gives an EXACT efficiency decomposition (all terms up to k sum
          to f(x) - base value) -- a bounded k-th-order Taylor explanation. Downside: the
          top order is defined differently from lower orders, and the values depend on the
          chosen truncation k. Best fit for the "phenotype = single positions + pairwise +
          3-way epistasis, and the terms add up" story.

        * Faith-Shap (Tsai et al.). Interactions are the coefficients of the best
          order-<=k polynomial approximation of f_x under Shapley-weighted least squares.
          Strongest theoretical footing and also yields an efficiency decomposition.
          Downside: newer / less widely recognized, and the weights couple across orders
          (a least-squares solve over GKT-computable moments, not a single g_k(s) swap).

        What each needs on top of the shared machinery
        -----------------------------------------------
        * SII, and STI's sub-top orders: supply the appropriate g_k(s); done.
        * STI top order: one extra boundary rule for order == k.
        * Faith-Shap: assemble the least-squares coefficients from the (GKT-computable)
          Shapley-weighted moments -- more glue code, same primitives underneath.

        Practical scoping
        -----------------
        Any single interaction T is cheap (polynomial in L). But there are C(L, k)
        order-k sets, so "all interactions up to order 5" blows up combinatorially. Plan
        to compute SELECTED tuples (or all pairs / all triples for small L) rather than
        the full interaction tensor.

        Covariance and verification
        ----------------------------
        Covariance generalizes ``_shap_R_cov`` by branching on ``p in T`` / ``p in T'``
        instead of ``p == i`` / ``p == j`` (a 2^k-term difference structure); it can be
        deferred (mean first). Verification mirrors the existing brute-force references:
        enumerate Delta_T f_x(S) directly, independent of the polynomial layer.

        The eventual signature will need an interaction set T (or an order plus
        enumeration) and a choice of index; kept as-is until the index is chosen.
        """
        raise NotImplementedError(
            "Interaction SHAP: the machinery generalizes to any order, but the "
            "interaction index (SII / Shapley-Taylor / Faith-Shap) is not yet chosen. "
            "See the docstring for the trade-offs."
        )

    def shap_region(self, pi_star, pi, eta: float = 1.0, indices=None, calc_covar: bool = False):
        """Region SHAP (notes, Section 3.4 'Extension to regions'). NOT YET IMPLEMENTED.

        Explain a region described by a product distribution ``pi_star`` against a
        background ``pi``. Generalization: the fixed-channel ``a^p_{x_p, .}`` is replaced
        by the ``pi_star``-averaged channel ``sum_c pi_star^p_c a^p_{c, .}`` (another
        ``compute_zeta_pc`` call with ``pi_star``). Everything else -- the polynomial
        layer, the ``w``-bookkeeping, ``apply_R`` -- is unchanged, so this is a small
        edit to ``_mean_channels`` / ``_cov_scalars`` (swap the 'fixed' definition) once
        the point-target path is verified.
        """
        raise NotImplementedError("Region SHAP is specified but not implemented in v1.")


__all__ = [
    "ShapKernelTrickComputer",
    "g_weights",
    "poly_prod_over_positions",
    "poly_prod_over_positions_bivariate",
    "apply_R",
    "apply_RxR",
]