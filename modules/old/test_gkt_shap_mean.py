"""
generalized_kernel_trick_computer_mean.py
=========================================
Posterior-**mean** SHAP values for a factorized-kernel Gaussian process, computed
with the generalized kernel trick (GKT). This is the mean-only slice of the full
GKT computer (the covariance path lives elsewhere); it is meant to be used as a
tool: build it once from a fitted model's ``KernelTrickComputer``, then ask it for
SHAP values of any target sequence against any background distribution.

Math (Theorem 2 of the notes, specialized to first-order SHAP means)
--------------------------------------------------------------------
For a target sequence ``x``, background ``pi`` and SHAP index ``i``:

    phi_i = m^i f ,   m^i_z = R( q^i_{z_i} * prod_{p != i} q^i_{z_p},  g )
        q^i_{z_i}(w) = 1{x_i = z_i} - pi^i_{z_i}                    (degree 0)
        q^i_{z_p}(w) = 1{x_p = z_p} * w + pi^p_{z_p}   (p != i)     (degree 1)
        g(s) = s! (L - s - 1)! / L!

Contracting each per-position factor against a training column ``y_t`` gives, per
position, a coefficient vector in the bookkeeping variable ``w`` (which counts
|S|):

    pi-channel   c0[p, t] = sum_c pi^p_c a^p_{c, y_{t,p}}      (eta-scaled zeta . y_t)
    fixed-chan   c1[p, t] = a^p_{x_p, y_{t,p}}                 (pins target char x_p)

so ``q^i_{z_p}`` -> coefficients ``[c0[p], c1[p]]`` for p != i, and ``q^i_{z_i}`` ->
the scalar ``c1[i] - c0[i]``. The product over positions is a convolution; the
linear functional ``R`` collapses ``w^s -> g(s)``. The result is the kernel-trick
row ``mK[i, t] = (m^i K_{*,X})_t``, and the posterior-mean SHAP value is

    phi_i = mK[i, :] @ Qy         (Qy == the model's mean_cache, reused from the KTC).

Efficiency
----------
The product ``prod_{p != i}`` shares almost all of its factors across the different
indices ``i``. Instead of re-convolving all L-1 factors for every i (O(L^2)
convolutions total), we precompute prefix and suffix products once
(``prefix[i] = prod_{p<i}``, ``suffix[i] = prod_{p>i}``) and combine them per index
with a single convolution (O(L) convolutions total, division-free).

Precision
---------
The SHAP weights ``g`` are near-cancelling, so the polynomial layer and assembly
run in float64 by default.

Usage
-----
    from kernel_trick_computer_refactored import KernelTrickComputer
    ktc  = KernelTrickComputer(model, alphabet, seq_length)
    shap = ShapKernelTrickComputer(ktc=ktc)

    phi  = shap.shap_values("ABACA", pi)   # (L,) posterior-mean SHAP values
    base = shap.base_value(pi)             # phi_0 = E_pi[f]
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch


class ShapKernelTrickComputer:
    """Posterior-mean first-order SHAP values via the generalized kernel trick.

    Composes with a fully-built ``KernelTrickComputer`` (``self.ktc``): all kernel,
    training-design and posterior state is read from it. This class adds the
    SHAP-specific polynomial layer (static methods below) and the mean assembly.

    Reused from ``self.ktc``: ``site_kernels``, ``Qy`` (mean_cache), ``_train_x``,
    ``comp_dtype``, ``set_precision``, ``validate_pis``, ``compute_zeta_pc``, ``E_MAP``.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model=None,
        alphabet: Optional[str] = None,
        seq_length: Optional[int] = None,
        ktc=None,
        g_dtype: torch.dtype = torch.float64,
    ):
        """Build from an existing ``KernelTrickComputer`` (preferred) or construct one.

        Args:
            model, alphabet, seq_length: forwarded to ``KernelTrickComputer`` when
                ``ktc`` is None.
            ktc: a pre-built ``KernelTrickComputer`` to compose with.
            g_dtype: dtype for the cached SHAP weight vector ``g`` (float64 by default).
        """
        if ktc is None:
            if model is None or alphabet is None or seq_length is None:
                raise ValueError("Provide either `ktc`, or all of (model, alphabet, seq_length).")
            from kernel_trick_computer_refactored import KernelTrickComputer
            ktc = KernelTrickComputer(model, alphabet, seq_length)

        self.ktc = ktc
        self.seq_length = ktc.seq_length
        self.n_alleles = ktc.n_alleles
        self.alphabet = ktc.alphabet
        self.device = ktc.device

        # The SHAP weight vector depends only on L, so cache it once.
        self._g_dtype = g_dtype
        self.g = self._g_weights(self.seq_length, device=self.device, dtype=g_dtype)

    # ------------------------------------------------------------------ #
    # Polynomial layer -- pure GKT algebra, no instance state.
    # Kept as static methods because they are intrinsic to the GKT (not
    # general-purpose utilities) yet stateless and unit-testable in isolation.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _g_weights(L: int, device=None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """SHAP weight vector ``g(s) = s! (L - s - 1)! / L!`` for s = 0..L-1.

        Built via lgamma for stability. Sanity identity (for any fixed i):
            sum_{s=0}^{L-1} C(L-1, s) * g(s) == 1.
        """
        s = torch.arange(L, device=device, dtype=dtype)
        log_g = (
            torch.lgamma(s + 1)
            + torch.lgamma(L - s)
            - torch.lgamma(torch.tensor(L + 1.0, dtype=dtype, device=device))
        )
        return torch.exp(log_g)

    @staticmethod
    def _poly_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convolve two coefficient tensors along the last (degree) axis.

        ``out[..., k] = sum_{i+j=k} a[..., i] * b[..., j]``. Leading axes broadcast
        (used to batch over training points). Returns (..., da + db - 1).
        """
        la, lb = a.shape[-1], b.shape[-1]
        batch = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        out = a.new_zeros(*batch, la + lb - 1)
        for j in range(lb):
            out[..., j:j + la] = out[..., j:j + la] + a * b[..., j:j + 1]
        return out

    @staticmethod
    def _apply_R(coeffs: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Linear functional R: map ``w^s -> g(s)`` and sum (dot of coeffs with g)."""
        D = coeffs.shape[-1]
        return torch.einsum("...d,d->...", coeffs, g[:D].to(coeffs.dtype))

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def target_to_onehot(self, x: Union[str, torch.Tensor]) -> torch.Tensor:
        """Target sequence -> (L, A) one-hot on device in the KTC comp dtype.

        A SHAP target is always fully specified, so this is a plain one-hot over the
        base alphabet (no '-' wildcard channel).
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
    # Per-position channels
    # ------------------------------------------------------------------ #

    def _mean_channels(
        self, x_onehot: torch.Tensor, pi: torch.Tensor, eta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the two per-position channels, each (L, t):

            pi-channel  c0[p,t] = sum_c pi^p_c a^p_{c, y_{t,p}}   (eta-scaled zeta . y_t)
            fixed-chan  c1[p,t] = a^p_{x_p, y_{t,p}}              (pins target char x_p)
        """
        L, A = self.seq_length, self.n_alleles
        Xtr = self.ktc._train_x.reshape(-1, L, A).to(x_onehot.dtype)          # (t, L, A)

        pi_b = pi.to(device=self.device, dtype=x_onehot.dtype).unsqueeze(0)   # (1, L, A)
        zeta = self.ktc.compute_zeta_pc(pi_b, eta)[0]                         # (L, A), eta-scaled

        site = self.ktc.site_kernels.to(x_onehot.dtype)                      # (L, A, A)
        Ax = torch.einsum("pc,pca->pa", x_onehot, site)                      # a^p_{x_p, .}  (L, A)

        c0 = torch.einsum("la,tla->lt", zeta, Xtr)                           # (L, t)
        c1 = torch.einsum("la,tla->lt", Ax, Xtr)                             # (L, t)
        return c0, c1

    # ------------------------------------------------------------------ #
    # GKT mean rows:  mK[i, t] = (m^i K_{*,X})_t
    # ------------------------------------------------------------------ #

    def _mean_rows(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        eta: float,
        indices: Sequence[int],
    ) -> torch.Tensor:
        """Build ``mK`` of shape (len(indices), t) via prefix/suffix polynomial products.

        Degree-1 factor per position: ``q_p(w) = c0[p] + c1[p] w`` -> coeffs [c0, c1].
        For index i the product over ``p != i`` is ``prefix[i] * suffix[i]`` (one
        convolution), scaled by the degree-0 difference ``c1[i] - c0[i]`` at i, then
        collapsed by R.
        """
        c0, c1 = self._mean_channels(x_onehot, pi, eta)          # (L, t), (L, t)
        L, t = c0.shape
        g = self.g.to(c0.dtype)

        factors = torch.stack([c0, c1], dim=-1)                  # (L, t, 2)

        # prefix[i] = prod_{p<i} q_p ;  suffix[i] = prod_{p>i} q_p   (batched over t)
        ones = c0.new_ones(t, 1)
        prefix = [ones] * L
        for p in range(1, L):
            prefix[p] = self._poly_mul(prefix[p - 1], factors[p - 1])
        suffix = [ones] * L
        for p in range(L - 2, -1, -1):
            suffix[p] = self._poly_mul(suffix[p + 1], factors[p + 1])

        rows = []
        for i in indices:
            prod_not_i = self._poly_mul(prefix[i], suffix[i])    # prod_{p != i} q_p , (t, L)
            R_val = self._apply_R(prod_not_i, g)                 # (t,)
            rows.append((c1[i] - c0[i]) * R_val)                 # (t,)
        return torch.stack(rows, dim=0)                          # (len(indices), t)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def shap_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        use_double: bool = True,
    ) -> torch.Tensor:
        """Posterior-mean first-order SHAP values ``phi_i(f, x)``.

        Assembly: ``phi = (m^i K_{*,X}) @ Qy`` (same shape as the KTC's mean path).

        Args:
            x: target sequence (string over the alphabet, or one-hot (L, A)).
            pi: background distribution, row-stochastic (L, A).
            eta: background-channel scale (default 1.0 == standard SHAP).
            indices: SHAP positions; default all 0..L-1.
            use_double: run in float64 (recommended; g near-cancels).

        Returns:
            phi of shape (len(indices),).
        """
        if use_double:
            self.ktc.set_precision(use_double=True)
        comp = self.ktc.comp_dtype

        x_oh = self.target_to_onehot(x).to(comp)
        pi = pi.to(device=self.device, dtype=comp)
        self.ktc.validate_pis(pi)

        if indices is None:
            indices = list(range(self.seq_length))

        mK = self._mean_rows(x_oh, pi, eta, indices)      # (len(indices), t)
        Qy = self.ktc.Qy.to(comp)                         # (t,)
        return mK @ Qy                                    # (len(indices),)

    def shap_values_batch(
        self,
        xs: Sequence[Union[str, torch.Tensor]],
        pi: torch.Tensor,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        use_double: bool = True,
    ) -> torch.Tensor:
        """Mean SHAP values for many targets against one background. Returns (len(xs), len(indices)).

        Loops for now; the fixed channel is the only x-dependent piece, so a vectorized
        version would batch ``_mean_channels`` over targets (c0/zeta are shared).
        """
        rows = [self.shap_values(x, pi, eta=eta, indices=indices, use_double=use_double)
                for x in xs]
        return torch.stack(rows, dim=0)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def base_value(self, pi: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        """phi_0 = f_x(empty set) = E_pi[f] = KTC.E_MAP(pi)."""
        pi = pi.to(device=self.device, dtype=self.ktc.comp_dtype)
        return self.ktc.E_MAP(pi)

    def local_accuracy_residual(
        self, x: Union[str, torch.Tensor], pi: torch.Tensor, eta: float = 1.0
    ) -> torch.Tensor:
        """f(x) - phi_0 - sum_i phi_i  (should be ~0 at eta == 1).

        f(x) is E_MAP at the delta distribution on x (fixing every position leaves
        nothing to average); phi_0 is E_MAP(pi).
        """
        x_oh = self.target_to_onehot(x)
        f_x = self.ktc.E_MAP(x_oh)
        phi0 = self.base_value(pi, eta=eta)
        phi = self.shap_values(x_oh, pi, eta=eta)
        return f_x - phi0 - phi.sum()


__all__ = ["ShapKernelTrickComputer"]