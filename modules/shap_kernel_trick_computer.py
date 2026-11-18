"""
shap_kernel_trick_computer.py
=============================

SHAP values -- with their full GP posterior uncertainty -- for a sequence-function map
learned by a factorized-kernel Gaussian process, via the *generalized kernel trick* (GKT).

This module is the SHAP consumer of the model-independent polynomial layer in
``generalized_kernel_trick.py`` (``g_weights``, ``poly_prod_over_positions`` (+ bivariate),
``apply_R`` / ``apply_RxR``). It composes a ``KernelTrickComputer``: every kernel,
training-data, and posterior quantity is read from ``self.ktc``, so the tested kernel-trick
implementation stays untouched and the reuse boundary is explicit.

    ktc  = KernelTrickComputer(model, alphabet, seq_length)
    shap = ShapKernelTrickComputer(ktc=ktc)

    phi      = shap.shap_values("ABACA", pi)                   # (L,) first-order means
    phi, cov = shap.shap_values("ABACA", pi, calc_covar=True)   # + posterior covariance
    base     = shap.base_value(pi)                             # phi_0 = E_pi[f]


How it works
------------
A first-order SHAP value sums over all ``2^{L-1}`` subsets of the other positions,

    phi_i(f, x) = sum_{S : i not in S} g(|S|) ( f_x(S u i) - f_x(S) ),
    g(s)        = s! (L - s - 1)! / L!,

where ``f_x(S) = E[f(X) | X_S = x_S]`` fixes the target's characters on ``S`` and averages
under ``pi`` elsewhere. Each ``f_x(S)`` is position-factorizable -- the ordinary kernel
trick can evaluate one -- but there are exponentially many.

The GKT collapses the sum with a bookkeeping variable ``w`` that *counts* ``|S|``. Each
position contributes a small polynomial in ``w`` instead of a scalar,

    q^i_{z_p}(w) = { delta_{x_i = z_i} - pi^i_{z_i}       if p == i   (degree 0)
                   { delta_{x_p = z_p} * w + pi^p_{z_p}   if p != i   (degree 1),

and the linear functional ``R`` maps ``w^s -> g(s)``. In the product over positions the
coefficient of ``w^s`` collects exactly the size-``s`` subsets, so a ``2^{L-1}``-term sum
becomes a degree-``(L-1)`` polynomial evaluated by repeated convolution across positions.
The posterior covariance is the same construction in two bookkeeping variables, collapsed
by ``R x R``.

Assembly is identical to the kernel trick's ``_gauge_fixed_posterior``::

    mean = mK @ Qy                        mK = m^T K_{*,X}   (built by _index_mK)
    cov  = R - (mK @ S) (mK @ S)^T        R  = m^T K m^T'    (built by _shap_R_interaction)


Choosing an interaction index
-----------------------------
Every interaction quantity here is the SAME position-factorized GKT moment collapsed by a
different weight vector (see :meth:`ShapKernelTrickComputer._index_weight_fn`). Pick by
the question:

* ``shap_values`` -- first-order Shapley. "How much does each single position contribute
  to *this* prediction, fairly divided?" The unique attribution with local accuracy (the
  ``phi_i`` sum to ``f(x) - phi_0``).
* ``dividend_values`` -- Harsanyi dividends == full-order Faith-Shap. "Which position
  *sets* are irreducibly epistatic?" A genuine 3-way effect contributes to its triple and
  to nothing smaller. Equals the functional-ANOVA component ``f_T`` under a product ``pi``
  at ``eta == 1``. NOT efficient. Note ``a({i})`` is the ANOVA *main effect*, which differs
  from the Shapley value ``phi_i`` -- do not mix them.
* ``shap_interaction_values`` -- SII (Grabisch-Roubens / Lundberg). Comparable to the
  SHAP-interaction literature, but it averages the discrete derivative over all contexts
  ``S``, which *smears* a true order-``m`` effect onto all its lower-order subsets: one
  triplet lights up all three of its pairs. Don't read its pairwise matrix as pure
  pairwise epistasis.
* ``faith_shap_values`` -- truncated Faith-Shap at max order ``ell``. An *efficient*
  explanation using only interactions up to order ``ell``; exact (== dividends) when the
  model's true maximum order is ``<= ell``, otherwise it redistributes higher-order
  epistasis into the ``<= ell`` terms (a different smearing than SII).


Covariance cost switch
----------------------
Each covariance-returning method takes ``covariance``:

* ``"full"`` (DEFAULT) -- the ``(n, n)`` posterior covariance matrix. This is the
  brute-force-verified path (``test_gkt_brute.py``).
* ``"diag"`` (opt-in) -- only the per-term VARIANCES, as a ``(n,)`` vector. The full ``R``
  is an ``n x n`` grid of bivariate products, but the diagonal needs just the ``n``
  same-tuple products (:meth:`ShapKernelTrickComputer._shap_R_diag`), and the diagonal of
  ``(mK S)(mK S)^T`` is the row-wise squared norm of ``mK S``. O(n) instead of O(n^2) --
  the win whenever a consumer only reads variances / standard deviations.

``covariance`` is consulted only when ``calc_covar`` is true. ONE EXCEPTION to the O(n)
claim: ``faith_shap_values`` with a non-empty Eq. (16) tail, where ``Var(E_S)`` genuinely
depends on off-diagonal dividend covariance -- see that method.


Verification
------------
``shap_brute_force.ShapBruteForce`` recomputes every quantity by direct subset enumeration
without the polynomial layer, so agreement is real evidence. ``test_gkt_brute.py`` drives
it; ``probe_R_diag.py`` gates the ``"diag"`` path against the ``"full"`` one.


Still to do
-----------
* KeOps / LazyTensor variant (this module is dense/torch only), mirroring
  ``compute_LTDR_lazy`` in the kernel-trick module.
* ``shap_region``: region SHAP against a product distribution ``pi_star`` -- specified,
  not implemented. The only change is swapping the fixed channel for a
  ``pi_star``-averaged one in :meth:`_batched_channels` / :meth:`_cov_scalars`.
* An ML/econ-notion convenience switch. (ML notion is the default: one feature per
  position, background-averaged under ``pi``. The econ notion -- background concentrated
  on the wild type, features = non-WT positions -- is already reachable by passing a
  WT-concentrated ``pi`` and restricting ``indices``.)

``eta`` scales the background-averaged ("pi") channel only; the fixed channel is unscaled.
Standard SHAP semantics and local accuracy assume ``eta == 1``. ``eta != 1`` is an
experimental knob inherited from the kernel-trick code and is not expected to telescope.
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


class ShapKernelTrickComputer:
    """SHAP values and posterior covariance via the generalized kernel trick.

    Composes a ``KernelTrickComputer`` (``self.ktc``) and adds only the SHAP-specific
    factor builders plus the public API.

    Attributes:
        ktc: the composed ``KernelTrickComputer`` (source of all reused state).
        seq_length, n_alleles, alphabet, device: mirrored from ``ktc``.
        g: cached first-order SHAP weight vector ``g(s)`` (depends only on ``L``).
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
            model, alphabet, seq_length: forwarded to ``KernelTrickComputer`` if ``ktc``
                is None.
            ktc: a pre-built computer to compose with (preferred).
            g_dtype: dtype of the SHAP weight vectors (float64 by default; they are
                near-cancelling).

        Raises:
            ValueError: if neither ``ktc`` nor all of (model, alphabet, seq_length) given.
        """
        if ktc is None:
            if model is None or alphabet is None or seq_length is None:
                raise ValueError("Provide either `ktc`, or all of (model, alphabet, seq_length).")
            # Local import so this module stays importable when the refactored module is
            # resolved via the test-suite path shim (see conftest.py).
            from kernel_trick_computer_refactored import KernelTrickComputer
            ktc = KernelTrickComputer(model, alphabet, seq_length)

        self.ktc = ktc
        self.seq_length = ktc.seq_length
        self.n_alleles = ktc.n_alleles
        self.alphabet = ktc.alphabet
        self.device = ktc.device

        self._g_dtype = g_dtype
        self.g = g_weights(self.seq_length, k=1, device=self.device, dtype=g_dtype)
        self._g_cache: Dict[int, torch.Tensor] = {1: self.g}

    # ================================================================== #
    # Weights, validation, and small shared helpers
    # ================================================================== #

    def _g_for_order(self, k: int) -> torch.Tensor:
        """Return (and cache) the order-``k`` SII weight vector ``g_k``, length ``L-k+1``.

        ``g_1`` is the first-order Shapley weight ``self.g``. See :func:`g_weights`.
        """
        g = self._g_cache.get(k)
        if g is None:
            g = g_weights(self.seq_length, k=k, device=self.device, dtype=self._g_dtype)
            self._g_cache[k] = g
        return g

    def _index_weight_fn(self, index: str) -> Callable[[int], torch.Tensor]:
        """Return ``k -> g_k``, the R-weight vector naming an interaction index.

        Choosing an index is *only* choosing this weight -- the GKT moment is identical.

        * ``'sii'``      -- Shapley Interaction Index, ``g_k(s) = s!(L-k-s)!/(L-k+1)!``:
          averages the discrete derivative over all contexts ``S`` (smears high orders
          onto subsets).
        * ``'dividend'`` -- Harsanyi dividend == full-order Faith-Shap,
          ``g_k = [1, 0, 0, ...]``: keeps only ``S = empty``, so no smearing.
        """
        if index == "sii":
            return self._g_for_order
        if index == "dividend":
            def _delta_weight(k: int) -> torch.Tensor:
                v = torch.zeros(self.seq_length - k + 1,
                                device=self.device, dtype=self._g_dtype)
                v[0] = 1.0
                return v
            return _delta_weight
        raise ValueError(f"unknown interaction index '{index}' (use 'sii' or 'dividend').")

    @staticmethod
    def _validate_covariance(covariance: str) -> None:
        """Validate the ``covariance`` cost switch (shared so the methods can't drift)."""
        if covariance not in ("full", "diag"):
            raise ValueError("covariance must be 'full' or 'diag'.")

    def _prepare(self, x, pi: torch.Tensor, use_double: bool):
        """Set precision, encode the target, cast + validate ``pi``. -> (x_oh, pi, dtype).

        Every single-target public method opens with this. ``use_double`` is the default
        because the SHAP weights ``g`` are near-cancelling.
        """
        pi, comp = self._prepare_pi(pi, use_double)
        return self.target_to_onehot(x).to(comp), pi, comp

    def _prepare_pi(self, pi: torch.Tensor, use_double: bool):
        """Target-free half of :meth:`_prepare`, for the batched paths. -> (pi, dtype)."""
        if use_double:
            self.ktc.set_precision(use_double=True)   # recasts S, Qy, kernels; clears caches
        comp = self.ktc.comp_dtype
        pi = pi.to(device=self.device, dtype=comp)
        self.ktc.validate_pis(pi)
        return pi, comp

    def target_to_onehot(self, x: Union[str, torch.Tensor]) -> torch.Tensor:
        """Encode a target sequence as a one-hot ``(L, A)`` tensor.

        A plain one-hot over the base alphabet (no ``'-'`` wildcard channel), because a
        SHAP target is always fully specified -- unlike the extended-alphabet 'subseq'
        encoding the kernel trick uses for its M-rows.

        Args:
            x: an ``L``-character string over ``self.alphabet``, or an ``(L, A)`` one-hot.
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

    def _resolve_interactions(
        self,
        interactions: Optional[Sequence[Sequence[int]]],
        order: Optional[int],
        indices: Optional[Sequence[int]],
    ) -> List[Tuple[int, ...]]:
        """Normalize ``(interactions, order, indices)`` to a list of position tuples.

        Explicit ``interactions`` are used directly (each sorted and de-duplicated);
        otherwise every size-``order`` subset of ``indices`` is enumerated in
        ``itertools.combinations`` order. The returned list defines the row/column order
        of every downstream result, so callers should read it back rather than assume.
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

    # ================================================================== #
    # Per-position channels (the SHAP "encoding")
    # ================================================================== #

    def _batched_channels(self, xs, pi: torch.Tensor, eta: float):
        """The two per-position channels for a batch of targets. -> c0 (L,t), c1 (B,L,t).

        For position ``p`` and training sequence ``y_t``, the two contractions appearing
        in ``(m^T K)_y`` are

            pi-channel     c0[p, t] = sum_c pi^p_c a^p_{c, y_{t,p}}   (== compute_zeta_pc . y)
            fixed-channel  c1[b, p, t] = a^p_{x^b_p, y_{t,p}}         (pins target b's char)

        ``c0`` is TARGET-INDEPENDENT -- which is what makes the batched paths worthwhile:
        one shared build per chunk instead of one per target. ``eta`` scales only ``c0``
        (it is already inside ``compute_zeta_pc``).
        """
        L, A = self.seq_length, self.n_alleles
        comp = self.ktc.comp_dtype
        Xtr = self.ktc._train_x.reshape(-1, L, A).to(comp)                  # (t, L, A)
        pi = pi.to(device=self.device, dtype=comp)                         # (L, A)
        zeta = self.ktc.compute_zeta_pc(pi.unsqueeze(0), eta)[0]           # (L, A)
        site = self.ktc.site_kernels.to(comp)                              # (L, A, A)
        Xoh = torch.stack([self.target_to_onehot(x).to(comp) for x in xs])  # (B, L, A)
        c0 = torch.einsum("la,tla->lt", zeta, Xtr)                          # (L, t) shared
        Ax = torch.einsum("blc,lca->bla", Xoh, site)                        # (B, L, A)
        c1 = torch.einsum("bla,tla->blt", Ax, Xtr)                          # (B, L, t)
        return c0, c1

    def _mean_channels(self, x_onehot: torch.Tensor, pi: torch.Tensor, eta: float):
        """Single-target channels. -> c0 (L,t), c1 (L,t). Batch-of-one wrapper."""
        c0, c1 = self._batched_channels([x_onehot], pi, eta)
        return c0, c1[0]

    # ================================================================== #
    # mK = m^T K_{*,X}   (the mean rows; shared by single and batched paths)
    # ================================================================== #
    #
    # Both builders below take channels shaped (..., L, t) and return (..., n, t), so a
    # leading target-batch axis passes straight through -- the polynomial primitives
    # broadcast over it. That is why there is one implementation of each rather than a
    # per-target version plus a batched twin.

    @staticmethod
    def _align_channels(c0: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
        """Broadcast the shared pi-channel ``c0`` to ``c1``'s (possibly batched) shape."""
        return c0.expand(c1.shape) if c0.dim() < c1.dim() else c0

    def _first_order_mK(self, c0: torch.Tensor, c1: torch.Tensor,
                        indices: Sequence[int]) -> torch.Tensor:
        """First-order rows ``mK[i] = m^i K_{*,X}``. Channels (...,L,t) -> (...,n_idx,t).

        Only position ``i``'s factor changes across ``i``, so prefix/suffix products of
        the shared degree-1 factors ``[c0[p], c1[p]]`` are built once
        (``prefix[i] = prod_{p<i}``, ``suffix[i] = prod_{p>i}``) and combined per ``i``
        with a single convolution, then scaled by ``c1[i] - c0[i]``. O(L) amortized per
        index and division-free -- the reason this exists alongside the general
        :meth:`_index_mK`, which would redo the whole L-position product per tuple.
        """
        L = self.seq_length
        c0 = self._align_channels(c0, c1)
        g = self.g.to(c1.dtype)
        factors = [torch.stack([c0[..., p, :], c1[..., p, :]], dim=-1) for p in range(L)]

        ones = c1.new_ones(*c1.shape[:-2], c1.shape[-1], 1)
        prefix = [ones] * L
        for p in range(1, L):
            prefix[p] = poly_prod_over_positions([prefix[p - 1], factors[p - 1]])
        suffix = [ones] * L
        for p in range(L - 2, -1, -1):
            suffix[p] = poly_prod_over_positions([suffix[p + 1], factors[p + 1]])

        rows = []
        for i in indices:
            prod_not_i = poly_prod_over_positions([prefix[i], suffix[i]])   # (..., t, <=L)
            rows.append((c1[..., i, :] - c0[..., i, :]) * apply_R(prod_not_i, g))
        return torch.stack(rows, dim=-2)

    def _index_mK(self, c0: torch.Tensor, c1: torch.Tensor,
                  interactions: Sequence[Tuple[int, ...]],
                  weight_fn: Optional[Callable[[int], torch.Tensor]] = None) -> torch.Tensor:
        """Interaction rows ``mK[a] = m^{T_a} K_{*,X}``. Channels (...,L,t) -> (...,n,t).

        Every position in ``T`` takes the degree-0 difference factor ``c1[p] - c0[p]``;
        every other position the shared degree-1 factor ``[c0[p], c1[p]]``. The product
        over positions is contracted by ``apply_R`` with the order-``|T|`` weight. This is
        the position-factorized ``Delta_T f_x(S)`` summed against ``g_{|T|}(|S|)`` over all
        ``S`` disjoint from ``T``.

        Args:
            weight_fn: ``k -> g_k``; None selects the SII weight, so unqualified callers
                get SII. Pass the DIVIDEND weight explicitly for dividend quantities.

        Returns:
            *Unnormalized* rows for the chosen index (public methods apply their own
            ``k!`` normalization).
        """
        L = self.seq_length
        if weight_fn is None:
            weight_fn = self._g_for_order
        c0 = self._align_channels(c0, c1)
        rows = []
        for T in interactions:
            Tset = set(T)
            g_k = weight_fn(len(Tset)).to(c1.dtype)
            factors = [
                (c1[..., p, :] - c0[..., p, :]).unsqueeze(-1) if p in Tset       # deg 0
                else torch.stack([c0[..., p, :], c1[..., p, :]], dim=-1)         # deg 1
                for p in range(L)
            ]
            rows.append(apply_R(poly_prod_over_positions(factors), g_k))
        return torch.stack(rows, dim=-2)

    # ================================================================== #
    # R = m^T K m^T'   (the prior-kernel quadratic form; bivariate GKT)
    # ================================================================== #

    def _cov_scalars(self, x_onehot: torch.Tensor, pi: torch.Tensor, eta: float) -> dict:
        """The three per-position kernel contractions the covariance needs, each ``(L,)``.

        With ``a^p`` symmetric::

            K_ff[p] = a^p_{x_p, x_p}                                   (fixed . fixed)
            K_fa[p] = eta   * sum_c' a^p_{x_p, c'} pi^p_{c'}           (fixed . avg)
            K_aa[p] = eta^2 * sum_{c,c'} pi^p_c a^p_{c,c'} pi^p_{c'}   (avg . avg)

        ``eta`` multiplies each 'avg' appearance; ``K_ff`` carries none.
        """
        site = self.ktc.site_kernels.to(x_onehot.dtype)                  # (L, A, A)
        pi = pi.to(device=self.device, dtype=x_onehot.dtype)             # (L, A)
        Kx = torch.einsum("pc,pca->pa", x_onehot, site)                  # a^p_{x_p, .}
        return {
            "K_ff": torch.einsum("pa,pa->p", Kx, x_onehot),
            "K_fa": eta * torch.einsum("pa,pa->p", Kx, pi),
            "K_aa": (eta * eta) * torch.einsum("pc,pca,pa->p", pi, site, pi),
        }

    def _cov_blocks(self, x_onehot: torch.Tensor, pi: torch.Tensor, eta: float) -> dict:
        """Per-position bivariate coefficient blocks, stacked over ``p``.

        ``B[a, b]`` multiplies ``w^a (w')^b``. Which block a position takes depends only on
        its membership in the two tuples ``(T_a, T_b)`` whose product we are forming::

            'free'   p in neither  deg (1,1)  (L,2,2)  [[aa, fa], [fa, ff]]
            'a_only' p in T_a only deg (0,1)  (L,1,2)  [[fa-aa, ff-fa]]
            'b_only' p in T_b only deg (1,0)  (L,2,1)  [[fa-aa], [ff-fa]]
            'both'   p in both     deg (0,0)  (L,1,1)  [[ff - 2 fa + aa]]

        Each block's shape is its ACTUAL degree, not zero-padded to (2,2): padding would be
        numerically harmless but would inflate the product's shape past ``len(g)`` and break
        ``apply_RxR``'s ``g[:D]`` slice.

        Built here once and indexed by every R builder below, so the block algebra has a
        single definition.
        """
        sc = self._cov_scalars(x_onehot, pi, eta)
        ff, fa, aa = sc["K_ff"], sc["K_fa"], sc["K_aa"]
        edge = torch.stack([fa - aa, ff - fa], dim=-1)                   # (L, 2)
        return {
            "free": torch.stack([torch.stack([aa, fa], dim=-1),
                                 torch.stack([fa, ff], dim=-1)], dim=-2),   # (L, 2, 2)
            "a_only": edge.unsqueeze(-2),                                   # (L, 1, 2)
            "b_only": edge.unsqueeze(-1),                                   # (L, 2, 1)
            "both": (ff - 2.0 * fa + aa).reshape(-1, 1, 1),                 # (L, 1, 1)
        }

    def _shap_R_interaction(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        eta: float,
        interactions: Sequence[Tuple[int, ...]],
        weight_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Full ``R[a, b] = m^{T_a} K m^{T_b}``, shape ``(n, n)`` (unnormalized).

        Per position, pick the :meth:`_cov_blocks` entry by membership in ``T_a`` (the ``w``
        axis) and ``T_b`` (the ``w'`` axis), take the 2-D convolution over positions, then
        contract with ``apply_RxR`` using each tuple's own weight vector (they differ when
        the tuples have different sizes).

        ``R`` has no dependence on training points -- it is the prior-kernel quadratic
        form. Training data enters only via the ``(mK S)(mK S)^T`` term in the callers.

        This is the clarity-first ``O(n^2)`` loop, kept naive so it stays checkable against
        brute force. For the DIAGONAL, :meth:`_shap_R_diag` gets the same numbers in
        ``O(n)`` -- pass it the SAME ``weight_fn``.

        Args:
            weight_fn: ``k -> g_k``; None selects the SII weight.
        """
        L = self.seq_length
        if weight_fn is None:
            weight_fn = self._g_for_order
        blk = self._cov_blocks(x_onehot, pi, eta)
        dtype = blk["both"].dtype

        g_by_order: Dict[int, torch.Tensor] = {}

        def gord(k: int) -> torch.Tensor:
            g = g_by_order.get(k)
            if g is None:
                g = weight_fn(k).to(dtype)
                g_by_order[k] = g
            return g

        sets = [set(T) for T in interactions]
        R = blk["both"].new_zeros(len(sets), len(sets))
        for a, Ta in enumerate(sets):
            ga = gord(len(Ta))
            for b, Tb in enumerate(sets):
                factors = []
                for p in range(L):
                    in_a, in_b = p in Ta, p in Tb
                    kind = ("both" if in_a and in_b else
                            "a_only" if in_a else
                            "b_only" if in_b else "free")
                    factors.append(blk[kind][p])
                R[a, b] = apply_RxR(poly_prod_over_positions_bivariate(factors),
                                    ga, gord(len(Tb)))
        return R

    def _shap_R_cov(self, x_onehot: torch.Tensor, pi: torch.Tensor, eta: float,
                    indices: Sequence[int]) -> torch.Tensor:
        """First-order ``R[i, j] = m^i K m^j``, shape ``(m, m)``.

        Exactly the singleton case of :meth:`_shap_R_interaction` under the SII weight
        (whose ``k == 1`` vector IS ``self.g``): the membership tests ``p == i`` / ``p == j``
        and ``p in T_a`` / ``p in T_b`` coincide when every tuple is a singleton. Kept as a
        named entry point because it is the first-order reference several probes compare
        against, but it no longer carries its own copy of the block algebra.
        """
        return self._shap_R_interaction(x_onehot, pi, eta,
                                        [(int(i),) for i in indices])

    def _shap_R_diag(
        self,
        x_onehot: torch.Tensor,
        pi: torch.Tensor,
        eta: float,
        interactions: Sequence[Tuple[int, ...]],
        weight_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Diagonal only: ``R[a, a]`` per tuple, shape ``(n,)`` -- ``O(n)`` products.

        The pair ``(a, a)`` puts every position of ``T_a`` in BOTH tuples, so it takes the
        ``'both'`` block and every other position the ``'free'`` block. That is ``n``
        bivariate products instead of the ``n^2`` grid, which is the whole saving behind
        ``covariance="diag"``.

        Serves the first-order path too, via singleton tuples under the default SII weight:
        then this equals ``torch.diagonal(_shap_R_cov(...))``.

        IMPORTANT: pass the SAME ``weight_fn`` the corresponding mean path used. Omitting it
        on a dividend query silently returns SII variances.

        Verified == ``torch.diagonal`` of both full builders (``probe_R_diag.py``, exact).
        """
        L = self.seq_length
        if weight_fn is None:
            weight_fn = self._g_for_order
        blk = self._cov_blocks(x_onehot, pi, eta)
        dtype = blk["both"].dtype

        g_by_order: Dict[int, torch.Tensor] = {}

        def gord(k: int) -> torch.Tensor:
            g = g_by_order.get(k)
            if g is None:
                g = weight_fn(k).to(dtype)
                g_by_order[k] = g
            return g

        out = blk["both"].new_zeros(len(interactions))
        for a, T in enumerate(interactions):
            Tset = set(T)
            g = gord(len(Tset))
            factors = [blk["both"][p] if p in Tset else blk["free"][p] for p in range(L)]
            out[a] = apply_RxR(poly_prod_over_positions_bivariate(factors), g, g)
        return out

    def _assemble_covariance(self, x_oh, pi, eta, tuples, weight_fn, zS,
                             covariance: str, scale: Optional[torch.Tensor] = None):
        """Finish a covariance from the data-correction rows ``zS = mK S``.

        ``cov = R - (mK S)(mK S)^T``. For ``"diag"`` only the diagonal is needed, and
        ``diag(zS zS^T)`` is exactly the row-wise squared norm ``(zS*zS).sum(-1)``, so the
        ``n x n`` grid is never built.

        ``scale`` is an optional per-tuple normalization ALREADY applied to ``mK`` (hence to
        ``zS``). It therefore has to be applied to ``R`` two-sidedly -- ``scale**2`` on the
        diagonal, ``scale outer scale`` on the matrix -- which is the one place the ``k!``
        convention touches the covariance.

        Returns the ``(n,)`` variance vector or the ``(n, n)`` matrix.
        """
        if covariance == "diag":
            R_diag = self._shap_R_diag(x_oh, pi, eta, tuples, weight_fn=weight_fn)
            if scale is not None:
                R_diag = (scale * scale) * R_diag
            return R_diag - (zS * zS).sum(dim=-1)
        R = self._shap_R_interaction(x_oh, pi, eta, tuples, weight_fn=weight_fn)
        if scale is not None:
            R = (scale[:, None] * R) * scale[None, :]
        return R - zS @ zS.T

    # ================================================================== #
    # Public: single-target attributions
    # ================================================================== #

    def base_value(self, pi: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        """``phi_0 = f_x(empty) = E_pi[f]`` -- the background expectation, as a scalar.

        This is exactly the kernel trick's ``E_MAP(pi)``. ``eta`` is accepted for interface
        symmetry only: ``E_MAP`` uses the unscaled fixed-kernel columns, consistent with the
        SHAP background channel at ``eta == 1``.
        """
        return self.ktc.E_MAP(pi.to(device=self.device, dtype=self.ktc.comp_dtype))

    def local_accuracy_residual(self, x, pi: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        """Diagnostic ``f(x) - phi_0 - sum_i phi_i``, ~0 at ``eta == 1``.

        The cheapest correctness anchor for the mean path. Uses two facts from the kernel
        trick: ``f(x) == E_MAP`` at the delta distribution on ``x`` (fixing every position
        leaves nothing to average), and ``phi_0 == E_MAP(pi)``.
        """
        x_oh = self.target_to_onehot(x)
        return (self.ktc.E_MAP(x_oh) - self.base_value(pi, eta=eta)
                - self.shap_values(x_oh, pi, eta=eta).sum())

    def shap_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        calc_covar: bool = False,
        use_double: bool = True,
        covariance: str = "full",
    ):
        """First-order SHAP values ``phi_i(f, x)``, optionally with posterior covariance.

        Args:
            x: target sequence (string, or one-hot ``(L, A)``).
            pi: background distribution, row-stochastic ``(L, A)``.
            eta: background-channel scale (see module notes). Default 1.0.
            indices: positions to compute; default all ``0..L-1``.
            calc_covar: also return the posterior covariance of the ``phi_i``.
            use_double: run in float64 (recommended -- ``g`` is near-cancelling).
            covariance: ``'full'`` (default) ``(m, m)`` matrix, or ``'diag'`` for the
                ``(m,)`` variance vector via the O(m) path. Consulted only when
                ``calc_covar``. Last in the signature so no positional call shifts meaning.

        Returns:
            ``phi`` ``(m,)``, or ``(phi, cov)`` with ``cov`` ``(m, m)`` -- or ``(phi, var)``
            with ``var`` ``(m,)`` under ``covariance='diag'``.

        Verified against ``bf_shap_values`` / ``bf_shap_covariance``.
        """
        self._validate_covariance(covariance)
        x_oh, pi, comp = self._prepare(x, pi, use_double)
        if indices is None:
            indices = list(range(self.seq_length))

        c0, c1 = self._mean_channels(x_oh, pi, eta)
        mK = self._first_order_mK(c0, c1, indices)              # (m, t)
        mean = mK @ self.ktc.Qy.to(comp)                        # (m,)
        if not calc_covar:
            return mean

        zS = mK @ self.ktc.S.to(comp)                           # (m, t)
        cov = self._assemble_covariance(x_oh, pi, eta, [(int(i),) for i in indices],
                                        None, zS, covariance)
        return mean, cov

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
        covariance: str = "full",
    ):
        """Interaction SHAP for position sets ``T`` -- the Shapley Interaction Index (SII).

            phi_T(f, x) = sum_{S : S cap T = empty} g_k(|S|) Delta_T f_x(S),
            Delta_T f_x(S) = sum_{W subseteq T} (-1)^{|T|-|W|} f_x(S u W),

        which at ``k == 2`` is the familiar four-term
        ``f_x(S u {i,j}) - f_x(S u i) - f_x(S u j) + f_x(S)``.

        The weight is SII (Grabisch-Roubens), ``g_k(s) = s!(L-k-s)!/(L-k+1)!``. It reduces
        to the first-order Shapley weight at ``k == 1`` (so ``order=1`` agrees with
        :meth:`shap_values`) and, at ``k == 2`` with ``normalize='shap'``, reproduces the
        ``phi_ij`` of Lundberg et al. SII was chosen over Shapley-Taylor / Faith-Shap for
        comparability with that literature; both alternatives layer onto the same GKT
        moments (Faith-Shap does, in :meth:`faith_shap_values`).

        Args:
            interactions: explicit position tuples (orders may be mixed). If None, all
                size-``order`` subsets of ``indices``.
            order: order to enumerate when ``interactions`` is None (default 2 == pairwise).
            normalize: ``'shap'`` (default) divides an order-``k`` value by ``k!`` -- the
                ``1/2`` symmetrization at ``k == 2``, a no-op at ``k == 1``; use this for the
                ``phi_ij`` matrix. ``'sii'`` returns the raw index.
            calc_covar, covariance, eta, use_double: as in :meth:`shap_values`. Note the
                normalization enters the covariance squared.

        Returns:
            ``(interactions, phi)``, or ``(interactions, phi, cov)`` -- ``cov`` is ``(n, n)``
            under ``'full'`` and a ``(n,)`` variance vector under ``'diag'``. The returned
            tuple list defines the row/column order.

        Verified against ``bf_interaction_values`` / ``bf_interaction_covariance``.
        """
        if normalize not in ("shap", "sii"):
            raise ValueError("normalize must be 'shap' or 'sii'.")
        self._validate_covariance(covariance)
        x_oh, pi, comp = self._prepare(x, pi, use_double)
        T_list = self._resolve_interactions(interactions, order, indices)

        mK = self._index_mK(*self._mean_channels(x_oh, pi, eta), T_list)   # (n, t) raw SII
        scale = self._normalization(T_list, normalize, comp)
        mK = scale[:, None] * mK
        phi = mK @ self.ktc.Qy.to(comp)                                   # (n,)
        if not calc_covar:
            return T_list, phi

        zS = mK @ self.ktc.S.to(comp)                                     # (n, t)
        cov = self._assemble_covariance(x_oh, pi, eta, T_list, None, zS,
                                        covariance, scale=scale)
        return T_list, phi, cov

    def _normalization(self, T_list, normalize: str, dtype) -> torch.Tensor:
        """Per-tuple ``k!`` scale vector for ``normalize='shap'``; ones for ``'sii'``."""
        vals = ([1.0 / factorial(len(T)) for T in T_list] if normalize == "shap"
                else [1.0] * len(T_list))
        return torch.tensor(vals, device=self.device, dtype=dtype)

    def shap_pairwise_values(
        self,
        x: Union[str, torch.Tensor],
        pi: torch.Tensor,
        eta: float = 1.0,
        indices: Optional[Sequence[int]] = None,
        fill_diagonal: bool = False,
        use_double: bool = True,
    ) -> torch.Tensor:
        """Pairwise interactions laid out as a symmetric matrix ``Phi[i, j] = phi_ij``.

        Convenience wrapper over :meth:`shap_interaction_values` at order 2 with
        ``normalize='shap'`` (i.e. SII / 2), the Lundberg pairwise convention.

        Args:
            indices: positions to include (default all); the matrix is indexed in this order.
            fill_diagonal: if True set ``Phi[i, i] = phi_i - sum_{j != i} phi_ij`` from the
                first-order values -- the Lundberg main-effect-on-diagonal convention, which
                makes each row sum to ``phi_i`` (a handy self-consistency check). Default
                False leaves the diagonal zero, since ``phi_ij`` is defined only for ``i != j``.
        """
        idx = list(range(self.seq_length)) if indices is None else [int(i) for i in indices]
        pos = {p: r for r, p in enumerate(idx)}
        m = len(idx)
        pairs = list(itertools.combinations(idx, 2))

        if not pairs:
            return torch.zeros(m, m, device=self.device, dtype=self.ktc.comp_dtype)

        T_list, phi_pairs = self.shap_interaction_values(
            x, pi, interactions=pairs, eta=eta, normalize="shap", use_double=use_double)
        Phi = phi_pairs.new_zeros(m, m)
        for (i, j), val in zip(T_list, phi_pairs):
            Phi[pos[i], pos[j]] = val
            Phi[pos[j], pos[i]] = val

        if fill_diagonal:
            phi_i = self.shap_values(x, pi, eta=eta, indices=idx, use_double=use_double)
            Phi[range(m), range(m)] = phi_i - Phi.sum(dim=1)   # diagonal still zero here
        return Phi

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
        covariance: str = "full",
    ):
        """Harsanyi dividends ``a(T) = Delta_T f_x(empty)`` -- the *pure* interaction.

            a(T) = sum_{W subseteq T} (-1)^{|T| - |W|} f_x(W)

        The discrete ``|T|``-th derivative at the EMPTY context: no averaging over other
        subsets ``S``. This is full-order (``ell == L``) Faith-Shap, so a genuine order-``m``
        effect contributes to its own set and to nothing smaller -- no smearing. Use it for
        "which position sets are irreducibly epistatic?".

        Same machinery as :meth:`shap_interaction_values` with two changes: the R-weight is
        ``[1, 0, 0, ...]`` instead of SII, and there is no ``k!`` symmetrization (dividends
        are the raw multilinear coefficients). The mean row collapses to a plain product,
        ``a(T) = sum_t Qy_t prod_{p in T}(c1-c0) prod_{p not in T} c0``.

        Caveats: dividends are NOT efficient (no per-order sum to ``f(x) - phi_0``), so use
        them for ranking and structure discovery rather than an additive story; and
        ``a({i})`` is the ANOVA main effect, NOT the Shapley value :meth:`shap_values`.

        Args:
            covariance: ``'diag'`` forwards the DIVIDEND weight into :meth:`_shap_R_diag` --
                omitting that forward would silently return SII variances. Other args as in
                :meth:`shap_interaction_values`.

        Returns:
            ``(interactions, a)``, or ``(interactions, a, cov)``; shapes as in
            :meth:`shap_interaction_values`.

        Verified against ``bf_dividend_values`` / ``bf_dividend_covariance``.
        """
        self._validate_covariance(covariance)
        x_oh, pi, comp = self._prepare(x, pi, use_double)
        T_list = self._resolve_interactions(interactions, order, indices)
        wfn = self._index_weight_fn("dividend")

        mK = self._index_mK(*self._mean_channels(x_oh, pi, eta), T_list, wfn)   # (n, t)
        a = mK @ self.ktc.Qy.to(comp)                                          # (n,)
        if not calc_covar:
            return T_list, a

        zS = mK @ self.ktc.S.to(comp)                                          # (n, t)
        cov = self._assemble_covariance(x_oh, pi, eta, T_list, wfn, zS, covariance)
        return T_list, a, cov

    # ================================================================== #
    # Public: truncated Faith-Shap (Tsai, Yeh & Ravikumar 2023, Thm 19 / Eq. 16)
    # ================================================================== #

    @staticmethod
    def _faith_shap_coeff(s: int, t: int, ell: int) -> float:
        """Eq. (16) weight redistributing dividend ``a(T)`` into Faith-Shap of ``S``.

        For ``S subset T``, ``|S| = s <= ell``, ``|T| = t > ell``::

            coeff(s, t, ell) = (-1)^{ell-s} (s/(ell+s)) C(ell, s)
                               * C(t-1, ell) / C(t+ell-1, ell+s)

        Only the tail (``t > ell``) needs this; for ``t <= ell`` the contribution is 1 if
        ``T == S`` and 0 otherwise, which :meth:`_faith_shap_map` handles as the identity
        term. Checked against the paper's Table 2 and both anchors (``ell == 1`` -> Shapley,
        ``ell == L`` -> dividends).
        """
        return ((-1.0) ** (ell - s)) * (s / (ell + s)) * comb(ell, s) \
            * comb(t - 1, ell) / comb(t + ell - 1, ell + s)

    def _faith_shap_map(self, S_list, ell: int, cap: int, idx_universe, dtype):
        """Build the target-independent Eq. (16) linear map. -> (div_tuples, M, tail_empty).

        Faith-Shap is ``E_S = a(S) + sum_{T superset S, ell < |T| <= cap} coeff(...) a(T)``
        -- a FIXED linear map ``M`` on the dividend vector. ``M`` and the dividend tuples it
        needs depend only on ``(S_list, ell, cap, idx_universe)``, never on the target, which
        is what lets the batched path build them once and apply ``A @ M.T`` as a single
        matmul. Shared by :meth:`faith_shap_values` and :meth:`faith_shap_values_batch` so
        the enumeration has one definition and the column order cannot drift between them.

        ``tail_empty`` reports whether no needed tuple exceeds ``ell`` -- true iff every row
        of ``M`` is a single ``1.0`` (``M`` is a selection matrix), which is what makes the
        diagonal covariance path exact and cheap. It holds whenever ``ell >= L`` or
        ``cap == ell``.

        Returns:
            ``div_tuples`` sorted by ``(size, tuple)``; ``M`` of shape
            ``(len(S_list), len(div_tuples))``; ``tail_empty``.
        """
        needed = set(S_list)
        tail_by_S: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
        for S in S_list:
            rest = [p for p in idx_universe if p not in set(S)]
            # Enumerate only the extra-position COUNTS that can put |T| = |S| + extra_k
            # inside the (ell, cap] tail window. `cap` bounds the RESULT either way, but
            # generating every subset of `rest` and filtering afterwards costs 2^|rest|
            # regardless of cap: 8 steps at L=4, 3.4e10 steps per reported tuple at L=36 --
            # and it paid that cost even when the window is EMPTY (ell >= L), where there
            # are no supersets to find. Since extra is disjoint from S, |T| = |S| + extra_k
            # exactly, so the window is a contiguous range of extra_k and the post-filter
            # becomes unnecessary rather than merely cheap.
            lo = max(1, ell - len(S) + 1)
            hi = min(len(rest), cap - len(S))
            sup: List[Tuple[int, ...]] = []
            for extra_k in range(lo, hi + 1):
                for extra in itertools.combinations(rest, extra_k):
                    T = tuple(sorted(set(S) | set(extra)))
                    sup.append(T)
                    needed.add(T)
            tail_by_S[S] = sup
        div_tuples = sorted(needed, key=lambda T: (len(T), T))
        div_index = {T: i for i, T in enumerate(div_tuples)}

        # Row r gets the identity term a(S) plus one Eq. (16) term per tail superset of S.
        # Those supersets are already in hand from the pass above, so we do NOT rescan all
        # of div_tuples per row (which was O(n_S * n_div): 5.2M subset tests at L=36).
        M = torch.zeros(len(S_list), len(div_tuples), device=self.device, dtype=dtype)
        for r, S in enumerate(S_list):
            s = len(S)
            M[r, div_index[S]] += 1.0                       # identity a(S) term
            for T in tail_by_S[S]:
                M[r, div_index[T]] += self._faith_shap_coeff(s, len(T), ell)

        return div_tuples, M, all(len(T) <= ell for T in div_tuples)

    def _faith_shap_setup(self, order_max, order_cap, interactions, order, indices):
        """Validate the Faith-Shap orders and resolve the reported tuples.

        -> ``(ell, cap, S_list, idx_universe)``. Shared by the single-target and batched
        methods so their argument validation cannot diverge.
        """
        L = self.seq_length
        ell = int(order_max)
        if ell < 1 or ell > L:
            raise ValueError(f"order_max (ell) must be in [1, {L}]; got {order_max}.")
        cap = L if order_cap is None else int(order_cap)
        if cap < ell or cap > L:
            raise ValueError(f"order_cap must be in [ell, L] = [{ell}, {L}]; got {order_cap}.")
        S_list = self._resolve_interactions(interactions, order, indices)
        if any(len(S) > ell for S in S_list):
            bad = [S for S in S_list if len(S) > ell]
            raise ValueError(f"Faith-Shap reports tuples of size <= order_max={ell}; "
                             f"got larger tuples {bad}.")
        idx_universe = list(range(L)) if indices is None else [int(i) for i in indices]
        return ell, cap, S_list, idx_universe

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
        covariance: str = "full",
    ):
        """Truncated Faith-Shap at maximum interaction order ``order_max`` (== ``ell``).

        Faith-Shap (Tsai, Yeh & Ravikumar, JMLR 2023) is the unique interaction index whose
        order-``<= ell`` values are the coefficients of the best degree-``ell`` multilinear
        surrogate of the game ``v(S) = f_x(S)`` under the Shapley kernel, subject to
        efficiency. Two anchors fix it: ``ell == 1`` gives the first-order Shapley values,
        ``ell == L`` gives the Harsanyi dividends. It is exact (no smearing) precisely when
        the true maximum interaction order is ``<= ell``; above that it redistributes the
        higher-order epistasis down into the ``<= ell`` terms.

        By Theorem 19 / Eq. (16) that redistribution is a fixed linear map on the dividends,
        so rather than fold it into an R-weight (it couples different tuple sizes, so it is
        not a per-order ``g_k(s)``) this computes the dividends via the verified GKT path and
        applies ``M`` from :meth:`_faith_shap_map`. The covariance then composes exactly as
        ``M cov_a M^T``.

        Tail truncation (``order_cap``)
            Eq. (16)'s tail runs over all supersets up to size ``L``, which is
            ``~2^(L-ell)`` of them. ``order_cap`` bounds it: the default (``L``) is EXACT,
            and a lower value is a genuine approximation -- the result is Faith-Shap of the
            ``order_cap``-order truncation of the model. Only advisable when the dropped
            high-order dividends are expected to be negligible.

        Why ``covariance='diag'`` is not always cheap here
            ``diag(M cov_a M^T)_r = sum_{j,k} M[r,j] cov_a[j,k] M[r,k]`` reads OFF-DIAGONAL
            dividend covariance whenever a row of ``M`` has more than one nonzero. So with
            an EMPTY tail (``ell >= L``, or ``order_cap == ell``) ``M`` is a selection matrix
            and the O(n) variance path is exact; with a NON-EMPTY tail the full ``cov_a``
            over ``div_tuples`` is genuinely required and ``'diag'`` only skips the final
            ``(n, n)`` product -- correct, but no asymptotic saving, and ``div_tuples`` is
            much larger than the reported set. Callers doing this per target should budget
            for it.

        Args:
            order_max: the maximum order ``ell`` (>= 1). Reported tuples must be ``<= ell``.
            interactions: explicit tuples ``S`` (each ``|S| <= order_max``); if None, all
                size-``order`` subsets of ``indices``.
            order_cap: highest dividend order in the tail sum (default ``L``, exact).
            Other args as in :meth:`shap_interaction_values`.

        Returns:
            ``(interactions, phi)``, or ``(interactions, phi, cov)`` with ``cov`` ``(n, n)``
            under ``'full'`` / ``(n,)`` variances under ``'diag'``.

        Verify with ``bf_faith_shap_values`` (weighted least-squares brute force) and the
        two anchors above.
        """
        self._validate_covariance(covariance)
        ell, cap, S_list, universe = self._faith_shap_setup(
            order_max, order_cap, interactions, order, indices)
        _, comp = self._prepare_pi(pi, use_double)
        div_tuples, M, tail_empty = self._faith_shap_map(S_list, ell, cap, universe, comp)

        # Dividends over every needed tuple, via the validated path. The diagonal-only
        # dividend covariance suffices exactly when M is a selection matrix.
        want_diag_only = calc_covar and covariance == "diag" and tail_empty
        if calc_covar:
            _, a_vec, a_second = self.dividend_values(
                x, pi, interactions=div_tuples, eta=eta, calc_covar=True,
                use_double=use_double, covariance="diag" if want_diag_only else "full")
        else:
            _, a_vec = self.dividend_values(
                x, pi, interactions=div_tuples, eta=eta, use_double=use_double)

        phi = M @ a_vec                                     # (n,)
        if not calc_covar:
            return S_list, phi
        if covariance == "diag":
            # selection M => diag(M diag(a_var) M^T) == (M*M) @ a_var; otherwise take the
            # diagonal image of the full a_cov without forming the (n, n) product.
            var = ((M * M) @ a_second if tail_empty
                   else ((M @ a_second) * M).sum(dim=1))
            return S_list, phi, var
        return S_list, phi, M @ a_second @ M.T

    # ================================================================== #
    # Public: batched over targets  ("many x, one pi")
    # ================================================================== #
    #
    # These share ONE pi-channel build across a whole chunk of targets. The per-target
    # loops they replaced were launch-bound: c0 is target-independent, so rebuilding it per
    # target left the GPU ~25% utilized firing tiny kernels. Numbers match a per-target loop
    # to ~1e-12 (probe_batch_indices.py); only speed differs. All are MEANS-ONLY -- for
    # per-target uncertainty, loop the single-target methods with covariance="diag".

    def _auto_batch_size(self, extra_deg: int = 1) -> int:
        """Choose a target-batch size from free memory and the per-chunk footprint.

        A batched build holds, simultaneously, the ``L`` per-position factor tensors of
        shape ``(B, t, 2)`` AND a few coefficient tensors of shape ``(B, t, <=L+1)``, so
        the footprint is ``B * t * (2L + 4(L + extra_deg)) * itemsize``. Counting only the
        coefficient tensors (as an earlier version did) understates it by ~L/4 -- fine at
        L=4, a ~9x under-estimate and a likely OOM at L=36.

        We budget a quarter of free GPU memory, then clamp to ``[64, 20000]``. On CPU
        there is no reliable free-memory query, so a fixed modest chunk is used.
        """
        L = self.seq_length
        t = self.ktc._train_x.reshape(-1, L, self.n_alleles).shape[0]
        itemsize = torch.tensor([], dtype=self.ktc.comp_dtype).element_size()
        per_target = t * (2 * L + 4 * (L + extra_deg)) * itemsize
        dev = self.device
        is_cuda = str(dev).startswith("cuda") or getattr(dev, "type", None) == "cuda"
        if is_cuda:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(dev)
            except Exception:  # noqa: BLE001
                free_bytes = 2 * (1024 ** 3)                 # 2 GB fallback
            B = int(max(1, (0.25 * free_bytes) // max(1, per_target)))
        else:
            B = 1024
        return int(min(max(B, 64), 20000))

    def _chunked_values(self, xs, pi, eta, use_double, batch_size, row_fn):
        """Run ``row_fn(c0, c1) -> (b, n, t)`` over target chunks, contract with ``Qy``.

        The single chunking / channel-sharing loop behind every batched method; the only
        thing that varies between them is which mK builder ``row_fn`` calls.

        Returns:
            ``(n_x, n)`` values.
        """
        pi, comp = self._prepare_pi(pi, use_double)
        Qy = self.ktc.Qy.to(comp)
        B = batch_size or self._auto_batch_size()
        out = []
        for s in range(0, len(xs), B):
            c0, c1 = self._batched_channels(xs[s:s + B], pi, eta)
            out.append(row_fn(c0, c1) @ Qy)
        return torch.cat(out, dim=0)

    def shap_values_batch(
        self, xs, pi, eta: float = 1.0, indices=None,
        use_double: bool = True, batch_size=None,
    ) -> torch.Tensor:
        """Mean first-order SHAP values for many targets. -> ``(len(xs), len(indices))``."""
        idx = list(range(self.seq_length)) if indices is None else [int(i) for i in indices]
        return self._chunked_values(xs, pi, eta, use_double, batch_size,
                                    lambda c0, c1: self._first_order_mK(c0, c1, idx))

    def dividend_values_batch(
        self, xs, pi, interactions=None, order=2, eta: float = 1.0,
        indices=None, use_double: bool = True, batch_size=None,
    ):
        """Mean dividends ``a(T)(x)`` for many targets. -> ``(interactions, (n_x, n))``."""
        T_list = self._resolve_interactions(interactions, order, indices)
        wfn = self._index_weight_fn("dividend")
        A = self._chunked_values(xs, pi, eta, use_double, batch_size,
                                 lambda c0, c1: self._index_mK(c0, c1, T_list, wfn))
        return T_list, A

    def interaction_values_batch(
        self, xs, pi, interactions=None, order=2, eta: float = 1.0,
        indices=None, normalize: str = "sii", use_double: bool = True, batch_size=None,
    ):
        """Mean SII (or ``k!``-normalized) interaction values for many targets.

        -> ``(interactions, (n_x, n))``. ``normalize`` as in :meth:`shap_interaction_values`.
        """
        if normalize not in ("shap", "sii"):
            raise ValueError("normalize must be 'shap' or 'sii'.")
        T_list = self._resolve_interactions(interactions, order, indices)
        Phi = self._chunked_values(xs, pi, eta, use_double, batch_size,
                                   lambda c0, c1: self._index_mK(c0, c1, T_list))
        return T_list, self._normalization(T_list, normalize, Phi.dtype).unsqueeze(0) * Phi

    def faith_shap_values_batch(
        self, xs, pi, order_max, interactions=None, order=2, eta: float = 1.0,
        indices=None, use_double: bool = True, order_cap=None, batch_size=None,
    ):
        """Mean truncated Faith-Shap ``E_S(x)`` for many targets. -> ``(S_list, (n_x, n))``.

        The Eq. (16) map ``M`` is target-independent, so it is built once and applied to the
        batched dividend matrix as a single ``A @ M.T``. At ``ell >= L`` the tail is empty,
        ``M`` is the identity, and this returns the dividends.
        """
        ell, cap, S_list, universe = self._faith_shap_setup(
            order_max, order_cap, interactions, order, indices)
        div_tuples, M, _ = self._faith_shap_map(S_list, ell, cap, universe,
                                                self.ktc.comp_dtype)
        _, A = self.dividend_values_batch(xs, pi, interactions=div_tuples, eta=eta,
                                          use_double=use_double, batch_size=batch_size)
        return S_list, A @ M.T

    # Compatibility aliases: the batched methods above ARE the vectorized implementations
    # (the older per-target loop wrappers of the same names are gone, since they returned
    # identical shapes for ~8.5x the time). These names are what analysis.py currently
    # calls; drop them once callers use the short names.
    shap_values_batch_vectorized = shap_values_batch
    dividend_values_batch_vectorized = dividend_values_batch
    interaction_values_batch_vectorized = interaction_values_batch
    faith_shap_values_batch_vectorized = faith_shap_values_batch

    # ================================================================== #
    # Public: global (dataset-level) summaries
    # ================================================================== #

    def global_shap_importance(self, xs, pi, indices=None, eta: float = 1.0,
                               use_double: bool = True, batch_size=None):
        """Global first-order importance: ``mean_x |phi_i|``. -> ``(indices, (m,))``.

        The ABSOLUTE value is deliberate: a position can matter globally yet have signed
        ``phi_i`` averaging to ~0 because it pushes the phenotype up on some targets and down
        on others, and that cancellation would hide real importance.

        This is importance under the target distribution you pass, still relative to ``pi`` --
        not "irrespective of target" in any absolute sense. A biased ``xs`` biases it equally.
        """
        phis = self.shap_values_batch(xs, pi, eta=eta, indices=indices,
                                      use_double=use_double, batch_size=batch_size)
        idx = list(range(self.seq_length)) if indices is None else [int(i) for i in indices]
        return idx, phis.abs().mean(dim=0)

    def global_dividend_summary(self, xs, pi, interactions=None, order=2, eta: float = 1.0,
                               indices=None, use_double: bool = True, batch_size=None,
                               normalize_sobol: bool = False):
        """Dataset-level dividend summary. -> ``(interactions, dict of (n,) tensors)``.

        Reductions over the target axis, per tuple ``T``:

        * ``mean_abs``     -- ``mean_x |a(T)(x)|``: global importance of the interaction.
        * ``mean_signed``  -- ``mean_x a(T)(x)``: a diagnostic. Under a product ``pi`` with
          targets drawn from ``pi``, ANOVA components of order >= 1 are mean-zero, so this
          should be ~0 -- it also flags when ``xs`` is far from ``pi``.
        * ``sobol_unnorm`` -- ``Var_x a(T)(x)``: the unnormalized Sobol sensitivity index.
          The equality holds because ``a(T)(x)`` is the functional-ANOVA component ``f_T``
          (product ``pi``, ``eta == 1``) and ANOVA components are orthogonal, so total
          variance splits cleanly across tuples.

        PRECONDITION for the Sobol reading: the targets must be sampled from the same product
        ``pi`` used as the background. Pass the training set instead (generally NOT ``pi``-
        distributed) and ``sobol_unnorm`` is the variance of the ANOVA component over the
        empirical target distribution -- a reasonable global importance, but not the formal
        Sobol index. It is reported either way; interpret it by how ``xs`` was drawn.

        Args:
            normalize_sobol: also return ``sobol_normalized`` = ``sobol_unnorm`` over the
                *sample* variance of ``f(x)`` (reported as ``total_var_sample``). A sample
                estimate, and a genuine Sobol ratio only under the same precondition.
        """
        T_list, A = self.dividend_values_batch(
            xs, pi, interactions=interactions, order=order, eta=eta, indices=indices,
            use_double=use_double, batch_size=batch_size)                    # (n_x, n)
        summary = {
            "mean_abs": A.abs().mean(dim=0),
            "mean_signed": A.mean(dim=0),
            "sobol_unnorm": A.var(dim=0, unbiased=False),
        }
        if normalize_sobol:
            comp = self.ktc.comp_dtype
            # f(x) == E_MAP at the delta distribution on x.
            fvals = torch.stack([self.ktc.E_MAP(self.target_to_onehot(x).to(comp))
                                 for x in xs])
            summary["total_var_sample"] = fvals.var(unbiased=False)
            summary["sobol_normalized"] = summary["sobol_unnorm"] / summary["total_var_sample"]
        return T_list, summary

    def global_faith_shap_summary(self, xs, pi, order_max, interactions=None, order=2,
                                  eta: float = 1.0, indices=None, use_double: bool = True,
                                  order_cap=None, batch_size=None):
        """Dataset-level Faith-Shap summary. -> ``(S_list, dict of (n,) tensors)``.

        Keys ``mean_abs`` and ``mean_signed`` mirror :meth:`global_dividend_summary`. The
        third is plain ``var`` rather than ``sobol_unnorm``: truncated Faith-Shap (``ell < L``)
        mixes ANOVA orders, so its dispersion across targets is NOT a clean Sobol index.

        On long sequences with ``ell < L`` the exact Eq. (16) tail is intractable -- pass a
        low ``order_cap`` (see :meth:`faith_shap_values`). At ``order_max >= L`` the tail is
        empty regardless and this equals the dividend summary.
        """
        S_list, Phi = self.faith_shap_values_batch(
            xs, pi, order_max, interactions=interactions, order=order, eta=eta,
            indices=indices, use_double=use_double, order_cap=order_cap,
            batch_size=batch_size)
        return S_list, {
            "mean_abs": Phi.abs().mean(dim=0),
            "mean_signed": Phi.mean(dim=0),
            "var": Phi.var(dim=0, unbiased=False),
        }


def make_background(
    mode: str,
    seq_length: int,
    alphabet: str,
    wildtype: Optional[Union[str, torch.Tensor]] = None,
    empirical_counts: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a background distribution ``pi`` -- row-stochastic ``(L, A)``.

    The background is the reference every attribution is measured against: an attribution
    explains the target relative to ``E_pi[f]``. The choice is not cosmetic -- it defines the
    baseline -- so it is explicit here rather than defaulted silently.

    Modes:
      * ``'uniform'``   -- ``1/A`` everywhere. The "average background": contribution
        relative to a random sequence. A product background, so dividends carry the
        functional-ANOVA / Sobol reading.
      * ``'wildtype'``  -- one-hot at the WT residue per position. Attributions become
        deviations from WT, i.e. the effect of mutating away from it. A delta background is
        a valid ``pi``; conditional expectations then collapse to substituting WT at the
        averaged-over positions, which is exactly the WT-referenced game.
      * ``'empirical'`` -- per-position residue frequencies from ``empirical_counts``
        ``(L, A)``, each row renormalized. Use when the natural reference is the observed
        sequence ensemble.

    Raises:
        ValueError: on an unknown mode, a missing/ill-shaped ``wildtype`` or
            ``empirical_counts``, negative counts, or an all-zero count row.
    """
    A, L = len(alphabet), int(seq_length)
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