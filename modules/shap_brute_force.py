"""
shap_brute_force.py
===================

Independent brute-force reference for the SHAP generalized-kernel-trick (GKT)
computer in ``generalized_kernel_trick_computer.py``.

Everything here recomputes SHAP quantities by *direct subset enumeration* -- no
polynomial layer, no ``apply_R`` / ``apply_RxR``, no ``_shap_z_mean`` /
``_interaction_z_mean``. The methods read only the raw cached arrays off the composed
``KernelTrickComputer`` (``site_kernels``, ``_train_x``, ``Qy``, ``S``) and enumerate,
so agreement with the GKT path is genuine evidence of correctness rather than a
shared-bug coincidence.

Because it enumerates, the cost is ``O((1 + A)^L)`` for the mean/conditional path and
``O(A^{2L})`` for the explicit-row covariance path -- keep ``L`` VERY small (this is a
test/verification tool, not a production path).

Usage::

    from kernel_trick_computer_refactored import KernelTrickComputer
    from generalized_kernel_trick_computer import ShapKernelTrickComputer
    from shap_brute_force import ShapBruteForce

    ktc  = KernelTrickComputer(model, alphabet, seq_length)
    shap = ShapKernelTrickComputer(ktc=ktc)
    bf   = ShapBruteForce(shap)          # or ShapBruteForce(ktc)

    x_oh = shap.target_to_onehot("ABACA")
    # first order
    torch.allclose(shap.shap_values(x_oh, pi),            bf.bf_shap_values(x_oh, pi))
    torch.allclose(shap.shap_values(x_oh, pi, calc_covar=True)[1],
                   bf.bf_shap_covariance(x_oh, pi))
    # interactions
    Ts = [(0, 1), (0, 2, 4)]
    torch.allclose(shap.shap_interaction_values(x_oh, pi, Ts),
                   bf.bf_shap_interaction_values(x_oh, pi, Ts))
    torch.allclose(shap.shap_interaction_values(x_oh, pi, Ts, calc_covar=True)[1],
                   bf.bf_shap_interaction_covariance(x_oh, pi, Ts))

The ``convention`` argument ('shap' / 'sii') mirrors
``ShapKernelTrickComputer.shap_interaction_values`` exactly, and is reimplemented here
(rather than imported) so the reference stays independent of the code under test.

Migration note
--------------
These methods previously lived on ``ShapKernelTrickComputer`` as ``bf_*`` / ``_bf_*``.
Call sites (e.g. ``test_gkt_brute.py``) should now build a ``ShapBruteForce`` and call
the identically named methods on it: ``shap.bf_shap_values(...)`` becomes
``ShapBruteForce(shap).bf_shap_values(...)``.
"""

from __future__ import annotations

import itertools
from math import exp, lgamma
from typing import Optional, Sequence, Tuple

import torch


class ShapBruteForce:
    """Enumeration-based reference for first-order and interaction SHAP.

    Args:
        shap_or_ktc: either a ``ShapKernelTrickComputer`` (preferred) or a bare
            ``KernelTrickComputer``. All raw state is read from the underlying
            ``KernelTrickComputer`` (``self.ktc``); ``seq_length`` / ``n_alleles`` /
            ``device`` are mirrored from whichever object is passed.
    """

    def __init__(self, shap_or_ktc):
        # Duck-type: a ShapKernelTrickComputer exposes `.ktc`; a KernelTrickComputer
        # does not. Either way we only need the raw caches and the shape metadata.
        ktc = getattr(shap_or_ktc, "ktc", shap_or_ktc)
        self.ktc = ktc
        self.seq_length = ktc.seq_length
        self.n_alleles = ktc.n_alleles
        self.device = ktc.device

    # ------------------------------------------------------------------ #
    # Weights and normalization (reimplemented locally to stay independent)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bf_g_weight(s: int, L: int) -> float:
        """First-order SHAP weight ``g(s) = s! (L-s-1)! / L!`` (scalar, via lgamma)."""
        return exp(lgamma(s + 1) + lgamma(L - s) - lgamma(L + 1))

    @staticmethod
    def _bf_g_order(s: int, L: int, k: int) -> float:
        """Order-``k`` SII weight ``g_k(s) = s! (L-k-s)! / (L-k+1)!`` (scalar).

        Reduces to :meth:`_bf_g_weight` at ``k == 1``.
        """
        return exp(lgamma(s + 1) + lgamma(L - k - s + 1) - lgamma(L - k + 2))

    @staticmethod
    def _bf_norm_factor(k: int, convention: str) -> float:
        """Convention scale, matching ``ShapKernelTrickComputer._norm_factor``."""
        if convention == "sii":
            return 1.0
        if convention == "shap":
            return 0.5 if k == 2 else 1.0
        raise ValueError(f"unknown convention '{convention}' (use 'shap' or 'sii').")

    @staticmethod
    def _normalize_interactions(interactions) -> list:
        """Coerce to a list of position tuples (single flat tuple/list is wrapped)."""
        seq = list(interactions)
        if len(seq) > 0 and all(isinstance(v, int) for v in seq):
            return [tuple(int(v) for v in seq)]
        return [tuple(int(p) for p in T) for T in seq]

    # ------------------------------------------------------------------ #
    # Encoding + core enumerated quantities
    # ------------------------------------------------------------------ #

    def _bf_onehot_to_idx(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """Convert a ``(L, A)`` one-hot target to a length-``L`` LongTensor of indices."""
        assert x_onehot.shape == (self.seq_length, self.n_alleles), \
            f"expected one-hot (L, A)=({self.seq_length},{self.n_alleles})"
        return x_onehot.argmax(dim=-1).long()

    def bf_posterior_mean(self, z_idx: Sequence[int]) -> torch.Tensor:
        """Posterior mean ``f(z) = sum_t (prod_p a^p_{z_p, y_{t,p}}) Qy_t`` for one sequence."""
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

    # ------------------------------------------------------------------ #
    # First-order SHAP means / residual
    # ------------------------------------------------------------------ #

    def bf_shap_values(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """First-order SHAP means by direct subset enumeration (no kernel trick).

            phi_i = sum_{S : i not in S} g(|S|) ( f_x(S u i) - f_x(S) ).
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
    # Interaction SHAP means (pairwise / higher order)
    # ------------------------------------------------------------------ #

    def bf_shap_interaction_values(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        interactions,
        convention: str = "shap",
    ) -> torch.Tensor:
        """Interaction SHAP means by direct enumeration of the discrete derivative.

            phi_T = c(k) * sum_{S : S cap T = empty} g_k(|S|) * Delta_T f_x(S),
            Delta_T f_x(S) = sum_{W subseteq T} (-1)^{|T|-|W|} f_x(S u W).

        The reference for ``ShapKernelTrickComputer.shap_interaction_values`` (mean).
        Args mirror that method: ``interactions`` is a list of position tuples (or a
        single flat tuple), ``convention`` is 'shap' or 'sii'.
        """
        L = self.seq_length
        x_idx = self._bf_onehot_to_idx(x_onehot)
        pi = pi.to(device=self.device)
        interactions = self._normalize_interactions(interactions)

        out = []
        for T in interactions:
            Tset = set(T)
            k = len(Tset)
            members = sorted(Tset)
            rest = [p for p in range(L) if p not in Tset]
            acc = self.bf_posterior_mean([int(c) for c in x_idx]).new_zeros(())
            for r in range(len(rest) + 1):
                gw = self._bf_g_order(r, L, k)
                for S in itertools.combinations(rest, r):
                    # discrete k-th derivative Delta_T f_x(S)
                    delta = acc.new_zeros(())
                    for wsz in range(k + 1):
                        for W in itertools.combinations(members, wsz):
                            sign = (-1.0) ** (k - len(W))
                            delta = delta + sign * self.bf_conditional_expectation(
                                tuple(S) + tuple(W), x_idx, pi
                            )
                    acc = acc + gw * delta
            out.append(self._bf_norm_factor(k, convention) * acc)
        return torch.stack(out)

    # ------------------------------------------------------------------ #
    # Explicit full-space rows -> covariance (first order and interaction)
    # ------------------------------------------------------------------ #

    def _bf_all_seqs_idx(self) -> torch.Tensor:
        """All ``A^L`` sequences as an ``(A^L, L)`` LongTensor of character indices."""
        L, A = self.seq_length, self.n_alleles
        return torch.tensor(list(itertools.product(range(A), repeat=L)),
                            dtype=torch.long, device=self.device)         # (A^L, L)

    def _bf_full_kernel(self, idxA: torch.Tensor, idxB: torch.Tensor) -> torch.Tensor:
        """Product kernel between two sets of sequences: ``K[a,b] = prod_p a^p_{A[a,p],B[b,p]}``."""
        L = self.seq_length
        site = self.ktc.site_kernels.to(self.ktc.comp_dtype)              # (L, A, A)
        Ka, Kb = idxA.shape[0], idxB.shape[0]
        K = torch.ones(Ka, Kb, dtype=site.dtype, device=self.device)
        for p in range(L):
            K = K * site[p][idxA[:, p]][:, idxB[:, p]]
        return K                                                          # (Ka, Kb)

    def _bf_shap_row(
        self, i: int, x_idx: torch.Tensor, pi: torch.Tensor, all_idx: torch.Tensor
    ) -> torch.Tensor:
        """Explicit first-order SHAP row ``m^i`` over all sequences (notes, section 3.3).

            m^i_z = sum_{S : i not in S} g(|S|) (delta_i - pi_i) prod_{p in S} delta_p
                                                              prod_{p not in S u i} pi_p.
        """
        L, A = self.seq_length, self.n_alleles
        dtype = pi.dtype
        N = all_idx.shape[0]
        x_idx = x_idx.to(self.device)
        pi = pi.to(self.device)

        z = all_idx                                                       # (N, L)
        delta = (z == x_idx.view(1, L)).to(dtype).T                       # (L, N)
        piz = pi[torch.arange(L).view(L, 1), z.T]                         # (L, N)

        rest = [p for p in range(L) if p != i]
        row = torch.zeros(N, dtype=dtype, device=self.device)
        for r in range(len(rest) + 1):
            gw = self._bf_g_weight(r, L)
            for S in itertools.combinations(rest, r):
                Sset = set(S)
                base = torch.ones(N, dtype=dtype, device=self.device)
                for p in range(L):
                    if p in Sset:
                        base = base * delta[p]
                    elif p != i:
                        base = base * piz[p]
                row = row + gw * (delta[i] - piz[i]) * base
        return row                                                        # (N,)

    def _bf_interaction_row(
        self, T: Sequence[int], x_idx: torch.Tensor, pi: torch.Tensor, all_idx: torch.Tensor
    ) -> torch.Tensor:
        """Explicit interaction row ``m^T`` over all sequences (raw SII, unnormalized).

            m^T_z = sum_{S : S cap T = empty} g_k(|S|)
                        prod_{p in T} (delta_p - pi_p)
                        prod_{p in S} delta_p  prod_{p not in S u T} pi_p.

        The order-1 case (``T = (i,)``) coincides with :meth:`_bf_shap_row`.
        """
        L, A = self.seq_length, self.n_alleles
        dtype = pi.dtype
        N = all_idx.shape[0]
        x_idx = x_idx.to(self.device)
        pi = pi.to(self.device)
        Tset = set(int(p) for p in T)
        k = len(Tset)

        z = all_idx                                                       # (N, L)
        delta = (z == x_idx.view(1, L)).to(dtype).T                       # (L, N)
        piz = pi[torch.arange(L).view(L, 1), z.T]                         # (L, N)

        # prod_{p in T} (delta_p - pi_p) is shared across all S.
        diff = torch.ones(N, dtype=dtype, device=self.device)
        for p in Tset:
            diff = diff * (delta[p] - piz[p])

        rest = [p for p in range(L) if p not in Tset]
        row = torch.zeros(N, dtype=dtype, device=self.device)
        for r in range(len(rest) + 1):
            gw = self._bf_g_order(r, L, k)
            for S in itertools.combinations(rest, r):
                Sset = set(S)
                base = torch.ones(N, dtype=dtype, device=self.device)
                for p in range(L):
                    if p in Sset:
                        base = base * delta[p]
                    elif p not in Tset:
                        base = base * piz[p]
                row = row + gw * diff * base
        return row                                                        # (N,)

    def _bf_covariance_from_rows(self, rows: torch.Tensor, all_idx: torch.Tensor) -> torch.Tensor:
        """Posterior covariance from explicit rows: ``R - (m K_{*,X}) S S^T (m K_{*,X})^T``.

            cov = rows K rows^T - (rows K_{*,X}) S S^T (rows K_{*,X})^T,
        with ``S == ktc.S`` the cached root (``S S^T = Q + (Qy)(Qy)^T``).
        """
        L = self.seq_length
        comp = self.ktc.comp_dtype
        train_idx = self.ktc._train_x.reshape(-1, L, self.n_alleles).argmax(dim=-1).to(self.device)  # (t, L)

        K_all = self._bf_full_kernel(all_idx, all_idx)                    # (A^L, A^L)
        R = rows @ K_all @ rows.T                                         # (n, n)

        K_sx = self._bf_full_kernel(all_idx, train_idx)                   # (A^L, t)
        mK = rows @ K_sx                                                  # (n, t)
        S = self.ktc.S.to(comp)                                          # (t, t)
        mKS = mK @ S                                                      # (n, t)
        return R - mKS @ mKS.T                                           # (n, n)

    def bf_shap_covariance(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """First-order posterior covariance by explicit Theorem-1 assembly.

        Built from explicit full-space rows ``m^i`` (no kernel trick), so it is
        independent of ``_shap_R_cov`` / ``shap_values(calc_covar=True)``. O(A^L) memory,
        O(A^{2L}) time -- for tiny ``L`` only.
        """
        L = self.seq_length
        comp = self.ktc.comp_dtype
        x_idx = self._bf_onehot_to_idx(x_onehot)
        pi = pi.to(device=self.device, dtype=comp)
        if indices is None:
            indices = list(range(L))

        all_idx = self._bf_all_seqs_idx()                                 # (A^L, L)
        rows = torch.stack([self._bf_shap_row(i, x_idx, pi, all_idx) for i in indices])  # (n, A^L)
        return self._bf_covariance_from_rows(rows, all_idx)

    def bf_shap_interaction_covariance(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        interactions,
        convention: str = "shap",
    ) -> torch.Tensor:
        """Interaction posterior covariance by explicit Theorem-1 assembly.

        Reference for ``shap_interaction_values(..., calc_covar=True)``. Builds explicit
        rows ``m^T`` (scaled by the ``convention`` factor) over all ``A^L`` sequences,
        then assembles ``R - (m K_{*,X}) S S^T (m K_{*,X})^T``. O(A^L) memory,
        O(A^{2L}) time -- tiny ``L`` only.
        """
        L = self.seq_length
        comp = self.ktc.comp_dtype
        x_idx = self._bf_onehot_to_idx(x_onehot)
        pi = pi.to(device=self.device, dtype=comp)
        interactions = self._normalize_interactions(interactions)

        all_idx = self._bf_all_seqs_idx()                                 # (A^L, L)
        rows = torch.stack([self._bf_interaction_row(T, x_idx, pi, all_idx)
                            for T in interactions])                       # (n, A^L) raw SII
        facs = torch.tensor([self._bf_norm_factor(len(T), convention) for T in interactions],
                            device=self.device, dtype=rows.dtype)         # (n,)
        rows = rows * facs[:, None]
        return self._bf_covariance_from_rows(rows, all_idx)


__all__ = ["ShapBruteForce"]