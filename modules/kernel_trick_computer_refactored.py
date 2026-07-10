"""
Kernel Trick Computer for Factorized Gaussian Processes.

This module provides utilities to compute the kernel trick for functions learned 
from Gaussian Processes (GPs) with positionally factorized learned kernels. 
It assumes that the learned kernel and the statistics matrix M can be 
factorized across sequence positions.
"""

import torch
from typing import List, Sequence, Any, Tuple, Union
from pykeops.torch import LazyTensor
from linear_operator.operators import KernelLinearOperator
from gpytorch.distributions import MultivariateNormal


def compute_LTDR_lazy(cL: torch.Tensor, cD: torch.Tensor, cR: torch.Tensor) -> LazyTensor:
    """
    Computes a LazyTensor representation of the product across positions without 
    materializing large intermediate tensors.

    Returns a LazyTensor K of shape (m, n, 1), where:
    K[m,n] = prod_p sum_a cL[m,p,a] * cD[p,a] * cR[n,p,a].

    Args:
        cL: Left tensor of shape (m, L, A)
        cD: Diagonal/weight tensor of shape (L, A)
        cR: Right tensor of shape (n, L, A)

    Returns:
        LazyTensor: The computed kernel trick tensor of shape (m, n, 1).
    """
    cL = cL.to(memory_format=torch.contiguous_format).contiguous()
    cD = cD.to(device=cL.device, dtype=cL.dtype, memory_format=torch.contiguous_format).contiguous()
    cR = cR.to(device=cL.device, dtype=cL.dtype, memory_format=torch.contiguous_format).contiguous()

    m, L, A = cL.shape
    n = cR.shape[0]

    K = None  # (m,n,1) LazyTensor accumulator
    for p in range(L):
        # Per-position contributions
        x = (cL[:, p, :] * cD[p, :]).contiguous()  # (m, A)
        y = cR[:, p, :].contiguous()               # (n, A)

        x_i = LazyTensor(x[:, None, :])            # (m,1,A)
        y_j = LazyTensor(y[None, :, :])            # (1,n,A)

        Kp = (x_i * y_j).sum(-1)                   # (m,n,1) LazyTensor
        K = Kp if K is None else (K * Kp)

    return K # (m,n,1) LazyTensor


def quadratic_form_from_lazy(K: LazyTensor, z: torch.Tensor) -> torch.Tensor:
    """
    Computes the quadratic form z^T K z without materializing K in memory.
    Executes v = K @ z via KeOps, then returns z^T v.

    Args:
        K: LazyTensor of shape (m, n, 1)
        z: Tensor of shape (n,)

    Returns:
        torch.Tensor: Scalar result of the quadratic form.
    """
    z = z.contiguous().view(-1)               # (m,)
    z_j = LazyTensor(z[None, :, None])        # (1,n,1)
    v = (K * z_j).sum(dim=1).squeeze(-1)      # (m,)
    return (v * z).sum()


def compute_LTDR(cL: Union[torch.Tensor, None], cD: torch.Tensor, cR: torch.Tensor) -> torch.Tensor:
    """
    Computes L^T D R without explicitly constructing full matrices L, D, or R.
    Relies on the positional factorization of the kernel.

    Args:
        cL: Left tensor (m x L x A). If None, implicitly treated as an identity.
        cD: Diagonal/weight tensor (L x A).
        cR: Right tensor (n x L x A).

    Returns:
        torch.Tensor: The product matrix of shape (m, n).
    """
    if cL is None:
        out = torch.einsum('la,nla->nl', cD, cR)
    else:
        out = torch.einsum('mla,la,nla->mnl', cL, cD, cR)
    return torch.prod(out, dim=-1)


class KernelTrickComputer:
    """
    Encapsulates kernel-site helpers and gauge-fixed posterior computations.
    Uses the kernel object exposed by EpiK model instances.
    
    All tensors are device-aware and compute types are configurable.
    """

    def __init__(self, model: Any, alphabet: str, seq_length: int):
        assert seq_length > 0
        assert len(alphabet) > 0

        self.model = model
        self.device = getattr(model, "device", torch.device("cpu"))

        self.alphabet = alphabet
        self.seq_length = seq_length
        self.alphabet_set = set(alphabet)
        self.n_alleles = len(self.alphabet_set)
        self.extended_alphabet = self.alphabet + "-"
        self.extended_size = self.n_alleles + 1

        self.kernel = model.kernel
        assert hasattr(self.kernel, "get_site_kernels")
        
        # NOTE: .clone() strips the gpytorch "inference tensor" flag so these
        # caches can be saved for backward when differentiating w.r.t. pi
        # (otherwise: "Inference tensors cannot be saved for backward").
        self.sigma2 = torch.exp(self.kernel.log_var.to(self.device)).detach().clone()
        raw_site_kernels = self.kernel.get_site_kernels()
        if isinstance(raw_site_kernels, list):
            raw_site_kernels = torch.stack(raw_site_kernels, dim=0)
        self.site_kernels = (
            raw_site_kernels.to(self.device).detach()
            * (self.sigma2 ** (1.0 / self.seq_length))
        ).detach().clone()

        self.S = model.gp.prediction_strategy.covar_cache.detach().clone()
        self.Qy = model.gp.prediction_strategy.mean_cache.detach().clone()

        self.linear_subseqs = [
            "-" * i + ch + "-" * (self.seq_length - i - 1)
            for i in range(self.seq_length)
            for ch in self.alphabet
        ]

        self.constant_subseq = ["-" * self.seq_length]

        # Caches for pi-invariant quantities reused across the k x k metrics loop.
        self._encode_cache: dict = {}   # tuple(subseqs) -> encoded tensor
        self._kx_cache: dict = {}       # dtype -> (KX, Qy)

        self.encoded_linear_subseqs = self._encode_subseqs(self.linear_subseqs)
        self.encoded_constant_subseq = self._encode_subseqs(self.constant_subseq)
        
        self.encoded_both_subseqs = self._encode_subseqs(self.constant_subseq + self.linear_subseqs)

        self._train_x = model.X.to(self.device).detach().clone()
        self._feature_dtype = self._train_x.dtype
        self.comp_dtype = self._feature_dtype
        
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)

        self.prior_mean = self._extract_prior_mean(model)

    @staticmethod
    def _extract_prior_mean(model) -> torch.Tensor:
        """Extract the GP's prior mean so predictions include it."""
        gp = model.gp
        val = float(getattr(gp, "constant_mean", 0.0))
        mean_mod = getattr(gp, "mean_module", None)
        if mean_mod is not None and hasattr(mean_mod, "constant"):
            val += float(mean_mod.constant.detach().squeeze())
        device = getattr(model, "device", torch.device("cpu"))
        return torch.tensor(val, device=device)

    def set_precision(self, use_double: bool = True):
        """Toggle internal computation precision and cast caches appropriately."""
        self.comp_dtype = torch.float64 if use_double else self._feature_dtype
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)
        self.prior_mean = self.prior_mean.to(self.comp_dtype)
        self._train_x = self._train_x.to(self.comp_dtype)
        self._kx_cache.clear()  # KX is keyed on dtype; drop entries after a recast

    # ---------- validation ----------
    def validate_sequences(self, sequences: Sequence[str], alphabet: str):
        alphabet_set = set(alphabet)
        for seq in sequences:
            assert len(seq) == self.seq_length, f"Sequence length must be {self.seq_length}"
            assert set(seq).issubset(alphabet_set), "Sequence contains invalid characters"

    def validate_pis(self, pis: torch.Tensor):
        with torch.no_grad():
            pis = pis.to(self.device)
            # Accept either (L, A) or (B, L, A). Normalize to batch dimension.
            if pis.ndim == 2:
                pis = pis.unsqueeze(0)  # B x L x A
            assert pis.ndim == 3, "pis must have shape (B, seq_length, n_alleles) or (seq_length, n_alleles)"
            B, L, A = pis.shape
            assert L == self.seq_length
            assert A == self.n_alleles
            assert torch.all(pis >= 0), f"pis must be non-negative, got:\n{pis}"
            assert torch.allclose(
                torch.sum(pis, dim=-1),
                torch.ones(B, L, device=pis.device, dtype=pis.dtype),
                atol=1e-6,
            )

    def _as_batch_pis(self, pis: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Normalize `pis` to batched shape (B, L, A). Returns (batched_pis, squeeze_batch_dim_flag)."""
        if pis.ndim == 2:
            return pis.unsqueeze(0), True
        if pis.ndim == 3:
            return pis, False
        raise ValueError(f"pis must have shape (L, A) or (B, L, A), got {tuple(pis.shape)}")

    @staticmethod
    def _maybe_squeeze_batch(out, squeeze_batch_dim: bool):
        if not squeeze_batch_dim:
            return out
        if isinstance(out, tuple):
            return tuple(x.squeeze(0) for x in out)
        return out.squeeze(0)

    def _encode_subseqs(self, subseqs: Sequence[str]) -> torch.Tensor:
        """Encode subsequences and reshape to (n_subseq, seq_length, n_alleles+1)."""
        key = tuple(subseqs)
        cached = self._encode_cache.get(key)
        if cached is not None:
            return cached

        import epik_loader
        epik_loader.ensure_epik_loaded()
        from epik.epik.utils import encode_seqs

        encoded = encode_seqs(subseqs, self.extended_alphabet)
        encoded = torch.as_tensor(encoded, device=self.device)
        encoded = encoded.reshape(-1, self.seq_length, self.n_alleles+1).contiguous()
        self._encode_cache[key] = encoded
        return encoded
    
    # ---------- main computational routines ----------

    def compute_zeta_pc(self, pis: torch.Tensor, eta: float) -> torch.Tensor:
        return eta * torch.einsum("pij,bpj->bpi", self.site_kernels, pis)

    def compute_zeta_p(self, pis1: torch.Tensor, pis2: torch.Tensor, eta: float) -> torch.Tensor:
        return (eta * eta) * torch.einsum("bpi,pij,bpj->bp", pis1, self.site_kernels, pis2)

    def _compute_R_invariants(self, subseqs: torch.Tensor):
        """pi-independent pieces of compute_R: the kernel-agreement term Y and the
        position-match sign mask M. Both have shape (1, L, r, r)."""
        S = subseqs            # r x L x (A+1)
        A = self.site_kernels  # L x A x A

        # contribution from K: a^p_(s_p,t_p) when s_p, t_p agree on a character at position p
        Y = torch.einsum("rla,lab,qlb->lrq", S[..., :-1], A, S[..., :-1])[None, ...]  # 1 x L x r x r

        M = (S[:, None, :, -1] == S[None, :, :, -1])   # r x r x L
        M = (2 * M - 1).permute(2, 0, 1).unsqueeze(0)  # 1 x L x r x r
        return Y, M

    def _compute_R_core(self, pis1: torch.Tensor, pis2: torch.Tensor, eta: float,
                        subseqs: torch.Tensor, Y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """pi-dependent remainder of compute_R, given precomputed invariants Y and M."""
        S = subseqs  # r x L x (A+1)
        zetas_pc_1 = self.compute_zeta_pc(pis1, eta)   # b x L x A
        zetas_pc_2 = self.compute_zeta_pc(pis2, eta)   # b x L x A
        zeta_p = self.compute_zeta_p(pis1, pis2, eta)  # b x L

        # contribution from zeta_p (appears in all terms)
        U = zeta_p[:, :, None, None]                                              # b x L x 1 x 1
        # contribution from zeta_pc for s_p
        V = torch.einsum('bla,rla->blr', zetas_pc_1, S[..., :-1])[:, :, None, :]  # b x L x 1 x r
        # contribution from zeta_pc for t_p
        W = torch.einsum('bla,rla->blr', zetas_pc_2, S[..., :-1])[:, :, :, None]  # b x L x r x 1

        R1 = Y + U - W - V                # b x L x r x r
        return torch.prod(R1 * M, dim=1)  # b x r x r

    def compute_R(self, pis1: torch.Tensor, pis2: torch.Tensor, eta: float, subseqs: torch.Tensor) -> torch.Tensor:
        Y, M = self._compute_R_invariants(subseqs)
        return self._compute_R_core(pis1, pis2, eta, subseqs, Y, M)

    def compute_R_various_pis_same_subseqs(self, pis: torch.Tensor, eta: float, subseqs: torch.Tensor) -> torch.Tensor:
        r = subseqs.shape[0]
        k = pis.shape[0]
        R = torch.zeros(k*r, k*r, device=self.device, dtype=self.site_kernels.dtype)

        # Y and M depend only on subseqs + site_kernels (not pis): compute once.
        Y, M = self._compute_R_invariants(subseqs)
        for i in range(k):
            for j in range(k):
                R[i*r:(i+1)*r, j*r:(j+1)*r] = self._compute_R_core(pis[i:i+1], pis[j:j+1], eta, subseqs, Y, M)
        return R
    
    def compute_z(self, pis: torch.Tensor, eta: float, subseqs: torch.Tensor) -> torch.Tensor:
        # dtype-consistent inputs
        dtype = self.comp_dtype
        X = self._train_x.reshape(-1, self.seq_length, self.n_alleles)  # t x L x A
        S = subseqs.to(dtype)  # r x L x (A+1)
        zetas = self.compute_zeta_pc(pis.to(dtype), eta)  # B x L x A
        A = self.site_kernels  # L x A x A
        # make Y align as (L, t, r) to add with W (B, L, t, r)
        Y = torch.einsum("tla,lab,rlb->ltr", X, A, S[...,:-1])  # L x t x r
        W1 = torch.einsum("tla,bla->btl", X, zetas)  # B x t x L
        W2 = 2*S[...,-1]-1  # r x L
        W = torch.einsum('btl,rl->bltr', W1, W2)  # B x L x t x r 
        
        Z1 = Y + W  # B x L x t x r
        return torch.prod(Z1, dim=1)  # B x t x r

    # ---------- Gauge Fixed Posterior Functions ----------

    def _resolve_subseqs(self, subseq_type: str, subseqs: Sequence[str]) -> torch.Tensor:
        """Helper to resolve the encoded subsequence tensor based on user flags."""
        if subseq_type == 'custom':
            return self._encode_subseqs(subseqs)
        elif subseq_type == 'linear':
            return self.encoded_linear_subseqs
        elif subseq_type == 'constant':
            return self.encoded_constant_subseq
        elif subseq_type == 'both':
            return self.encoded_both_subseqs
        else:
            raise ValueError(f"Unknown subseq_type '{subseq_type}'.")

    def _gauge_fixed_posterior(
        self, pis: torch.Tensor, eta: float = 1.0, encoded_subseqs: torch.Tensor = None, calc_covar: bool = False
    ):
        dtype = self.comp_dtype
        pis = pis.to(device=self.device, dtype=dtype).contiguous()
        enc = encoded_subseqs.to(dtype)
        z = self.compute_z(pis, eta, enc)  # B x t x r
        mean = self.Qy.to(dtype) @ z  # (t) @ (B x t x r) -> B x r
        
        if not calc_covar:
            return mean
        
        R = self.compute_R(pis, pis, eta, enc)  # b x r x r
        zS = torch.einsum('btr,tk->brk', z, self.S.to(dtype))  # b x r x t
        cov = R - torch.bmm(zS, zS.transpose(1, 2))  # b x r x r
        return mean, cov

    def _gauge_fixed_posterior_across_pis(
        self, pis: torch.Tensor, eta: float = 1.0, encoded_subseqs: torch.Tensor = None, calc_covar: bool = False
    ):
        dtype = self.comp_dtype
        pis = pis.to(device=self.device, dtype=dtype).contiguous()
        r = encoded_subseqs.shape[0]
        t = self._train_x.shape[0]
        B = pis.shape[0]

        # compute_z is batched over pis; build z for all pis in one call instead of looping.
        enc = encoded_subseqs.to(dtype)
        Z = self.compute_z(pis, eta, enc)        # B x t x r
        z = Z.permute(1, 0, 2).reshape(t, B * r)  # t x (B*r)
        mean = self.Qy.to(dtype) @ z  # (B*r)

        if not calc_covar:
            return mean

        R = self.compute_R_various_pis_same_subseqs(pis, eta, enc)  # (B*r) x (B*r)
        zS = z.T @ self.S.to(dtype)  # (B*r) x t
        cov = R - zS @ zS.T # (B*r) x (B*r)
        return mean, cov

    def gauge_fixed_posterior(
        self,
        pis: torch.Tensor,
        eta: float = 1.0,
        subseqs: Sequence[str] = (),
        subseq_type: str = 'custom',
        calc_covar: bool = False,
        validate: bool = True,
        max_batch: int = None,
        max_subseq_batch: int = None,
        across_pis: bool = False
    ):
        """
        Unified Gauge-Fixed Posterior interface. Replaces standalone functions for 
        linear, constant, constant+linear, and their cross-covariance ('various_pis') equivalents.
        
        Args:
            pis: Tensor of shapes (L, A) or (B, L, A).
            eta: Scaling parameter.
            subseqs: Custom sequences if subseq_type == 'custom'.
            subseq_type: 'custom', 'linear', 'constant', or 'both'.
            calc_covar: Whether to calculate covariance.
            validate: Validate input tensors.
            max_batch: Split batch dimension to avoid OOM.
            max_subseq_batch: Split sequence dimension (not supported with calc_covar).
            across_pis: If True, computes the full block matrix covariance across all pis (formerly `various_pis`).
        """
        raw_subseqs = tuple(subseqs) if subseq_type == 'custom' else ()
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        
        if validate:
            self.validate_pis(pis_batched)
            if subseq_type == 'custom':
                self.validate_sequences(raw_subseqs, self.extended_alphabet)

        # Handle Subseq Batching (only valid for non-cross-covariance independent pi runs)
        if not across_pis and max_subseq_batch is not None and max_subseq_batch > 0 and len(raw_subseqs) > max_subseq_batch:
            if calc_covar:
                raise ValueError("max_subseq_batch is not supported when calc_covar=True")
            chunks = []
            for start in range(0, len(raw_subseqs), max_subseq_batch):
                chunk_out = self.gauge_fixed_posterior(
                    pis, eta=eta, subseqs=raw_subseqs[start : start + max_subseq_batch],
                    subseq_type='custom', calc_covar=calc_covar, validate=False,
                    max_batch=max_batch, max_subseq_batch=None, across_pis=False
                )
                chunks.append(chunk_out)
            return self._maybe_squeeze_batch(torch.cat(chunks, dim=-1), squeeze_batch_dim)

        encoded_subseqs = self._resolve_subseqs(subseq_type, raw_subseqs)
        posterior_fn = self._gauge_fixed_posterior_across_pis if across_pis else self._gauge_fixed_posterior
        B = pis_batched.shape[0]

        # Process without batching
        if max_batch is None or max_batch <= 0 or max_batch >= B:
            out = posterior_fn(pis_batched, eta, encoded_subseqs, calc_covar)
            return self._maybe_squeeze_batch(out, squeeze_batch_dim)

        # Process with max_batch chunking (stitch back together)
        chunks = []
        for start in range(0, B, max_batch):
            chunk_out = posterior_fn(pis_batched[start:start+max_batch], eta, encoded_subseqs, calc_covar)
            chunks.append(chunk_out)

        if isinstance(chunks[0], tuple):
            out = tuple(torch.cat([c[i] for c in chunks], dim=0) for i in range(len(chunks[0])))
        else:
            out = torch.cat(chunks, dim=0)

        return self._maybe_squeeze_batch(out, squeeze_batch_dim)

    # ---------- Expectations and Variance Explained ----------

    def _kx_and_qy(self, dtype):
        """Build KX (kernel-mapped training points) and the matching Qy weights."""
        cached = self._kx_cache.get(dtype)
        if cached is not None:
            return cached

        KX = torch.einsum(
            "lab,tla -> tlb",
            self.site_kernels.to(dtype),
            self._train_x.reshape(-1, self.seq_length, self.n_alleles).to(dtype),
        )
        cached = (KX, self.Qy.to(dtype))
        self._kx_cache[dtype] = cached
        return cached

    def E_MAP(self, pi):
        dtype = pi.dtype
        KX, qy = self._kx_and_qy(dtype)
        return compute_LTDR(None, pi, KX) @ qy
    
    def E_MAP_2(self, pi, keops: bool = False):
        dtype = pi.dtype
        KX, qy = self._kx_and_qy(dtype)
        if keops:
            K_lazy = compute_LTDR_lazy(KX, pi, KX)
            return quadratic_form_from_lazy(K_lazy, qy)
        KXTDKX = compute_LTDR(KX, pi, KX)
        return qy @ KXTDKX @ qy
        
    def E_TRUNC(self, pi, weights, subseqs):
        encoded = self._encode_subseqs(subseqs)
        C = (encoded[...,:-1] + encoded[...,-1:]).to(pi.dtype)  
        return compute_LTDR(None, pi, C) @ weights.to(pi.dtype)
    
    def E_TRUNC_2(self, pi, weights, subseqs):
        encoded = self._encode_subseqs(subseqs)
        C = (encoded[...,:-1] + encoded[...,-1:]).to(pi.dtype)  
        w = weights.to(pi.dtype)
        return w @ compute_LTDR(C, pi, C) @ w
    
    def E_MAP_TRUNC(self, pi, weights, subseqs):
        encoded = self._encode_subseqs(subseqs)
        C = (encoded[...,:-1] + encoded[...,-1:]).to(pi.dtype)
        KX, qy = self._kx_and_qy(pi.dtype)
        return weights.to(pi.dtype) @ compute_LTDR(C, pi, KX) @ qy

    def _compute_expectations(self, pi, weights, subseqs, use_double, keops):
        """Centralized helper for VE and MSE components to avoid repeated math."""
        comp = torch.float64 if use_double else self._feature_dtype
        pi_c, w_c = pi.to(comp), weights.to(comp)
        
        return (
            self.E_MAP_2(pi_c, keops=keops).to(comp),
            self.E_TRUNC(pi_c, w_c, subseqs).to(comp),
            self.E_TRUNC_2(pi_c, w_c, subseqs).to(comp),
            self.E_MAP_TRUNC(pi_c, w_c, subseqs).to(comp),
            self.E_MAP(pi_c).to(comp)
        )

    def variance_explained(self, pi, weights, subseqs, use_double: bool = True, keops: bool = False):
        """Compute variance explained. If use_double=True, performs ops in float64."""
        e_map_2, e_trunc, e_trunc_2, e_map_trunc, e_map = self._compute_expectations(pi, weights, subseqs, use_double, keops)
        den = (e_map_2 - e_map * e_map)
        num = (e_trunc_2 + e_map_2 - 2.0 * e_map_trunc) - ((e_trunc - e_map) ** 2)
        ve = 1.0 - (num / den)
        return ve if use_double else ve.to(self._feature_dtype)

    def MSE(self, pi, weights, subseqs, use_double: bool = True, keops: bool = False):
        """Compute analytic MSE = Var_pi(g - h)."""
        e_map_2, e_trunc, e_trunc_2, e_map_trunc, e_map = self._compute_expectations(pi, weights, subseqs, use_double, keops)
        num = (e_trunc_2 + e_map_2 - 2.0 * e_map_trunc) - ((e_trunc - e_map) ** 2)
        return num if use_double else num.to(self._feature_dtype)

    def variance_explained_pieces(self, pi, weights, subseqs, use_double: bool = True, keops: bool = False):
        """Return the denominator and numerator of variance explained separately."""
        e_map_2, e_trunc, e_trunc_2, e_map_trunc, e_map = self._compute_expectations(pi, weights, subseqs, use_double, keops)
        den = (e_map_2 - e_map * e_map)
        num = (e_trunc_2 + e_map_2 - 2.0 * e_map_trunc) - ((e_trunc - e_map) ** 2)
        return den, num if use_double else (den.to(self._feature_dtype), num.to(self._feature_dtype))
    
    def compute_metrics_matrix(self, pis, metric: str = 've', use_double: bool = True, keops: bool = False, diagonal_only: bool = False):
        """
        Unified matrix generator for Variance Explained and MSE.
        If diagonal_only is True, returns a 1D tensor [k], else returns a 2D matrix [k, k].
        """
        k = pis.shape[0]
        result = torch.zeros(k if diagonal_only else (k, k))
        
        subseqs = self.constant_subseq + self.linear_subseqs
        weights = self.gauge_fixed_posterior(pis, eta=1, subseq_type='both')
        
        compute_fn = self.variance_explained if metric == 've' else self.MSE
        
        for i in range(k):
            if diagonal_only:
                result[i] = compute_fn(pis[i], weights[i], subseqs, use_double=use_double, keops=keops)
            else:
                for j in range(k):
                    result[i, j] = compute_fn(pis[j], weights[i], subseqs, use_double=use_double, keops=keops)
        return result

    def compute_MSE_cross(self, pis_region, pis_model, use_double: bool = True, keops: bool = False):
        """MSE in region pi_region by linear model generated from pi_model."""
        subseqs = self.linear_subseqs + self.constant_subseq
        weights = self.gauge_fixed_posterior(pis_model, eta=1, subseq_type='both')
        return self.MSE(pis_region, weights.squeeze(), subseqs, use_double=use_double, keops=keops)











class KernelTrickComputerSingletonKeOps:
    """KeOps/LazyTensor variant of :class:`KernelTrickComputer`.

    This is the memory-lean analogue of the dense ``KernelTrickComputer``. It
    computes the same gauge-fixed posterior of a position-factorizable statistic
    ``M`` of a GP-learned sequence-function map,

        M f  ~  N( theta_bar, R ),
        theta_bar = M K_{*,X} Q y,
        R         = M K M^T - (M K_{*,X}) (Q + (Qy)(Qy)^T) (M K_{*,X})^T,

    but never materializes the ``(#subseqs x #train)`` or ``(#subseqs x #subseqs)``
    matrices eagerly. Instead it hands the per-position feature blocks to a KeOps
    ``KernelLinearOperator`` whose ``covar_func`` evaluates the product kernel
    lazily, so the sequence space can be very large without exhausting memory.

    Differences from the dense class:
        * ``pis`` is a single distribution of shape ``(seq_length, n_alleles)`` --
          there is no batch dimension here (see :meth:`validate_pis`).
        * The encode / KX caches used by the dense class are omitted.

    Terminology and shapes used throughout:
        * ``L`` = ``seq_length``, ``A`` = ``n_alleles``. The *extended* alphabet
          appends a ``'-'`` wildcard, so its size is ``A + 1`` (``extended_size``).
          A ``'-'`` at a position means "average that position under ``pi``"; a real
          character means "fix that position".
        * ``a^p`` = the symmetric site-kernel matrix at position ``p`` (``site_kernels``,
          shape ``(L, A, A)``).
        * ``pi^p_c`` = background probability of character ``c`` at position ``p``.
        * ``S`` = ``covar_cache`` (a posterior root with ``S S^T = Q + (Qy)(Qy)^T``),
          ``Qy`` = ``mean_cache`` = ``Q y``.

    Args:
        model: a fitted EpiK GP whose kernel exposes ``get_site_kernels()`` and
            whose ``gp.prediction_strategy`` caches are already populated (i.e. a
            ``predict`` call has been made).
        alphabet: the base alphabet (without the ``'-'`` wildcard).
        seq_length: sequence length ``L``.
    """

    def __init__(self, model: Any, alphabet: str, seq_length: int):
        assert seq_length > 0
        assert len(alphabet) > 0

        # ---- basic identity / alphabet bookkeeping ---------------------------
        self.model = model
        self.device = getattr(model, "device", torch.device("cpu"))
        self.alphabet = alphabet
        self.seq_length = seq_length
        self.alphabet_set = set(alphabet)
        self.n_alleles = len(self.alphabet_set)
        # Extended alphabet adds the '-' wildcard used to mark averaged positions.
        self.extended_alphabet = self.alphabet + "-"
        self.extended_size = self.n_alleles + 1

        # ---- pull the kernel factors off the fitted model --------------------
        self.kernel = model.kernel
        assert hasattr(self.kernel, "get_site_kernels")
        # Overall signal variance sigma^2 = exp(log_var).
        self.sigma2 = torch.exp(self.kernel.log_var.to(self.device)).detach()
        # Site kernels a^p. Each is scaled by sigma2**(1/L) so that their product
        # over the L positions carries the full model variance (the product kernel
        # K_{x,y} = prod_p a^p_{x_p,y_p} then has the correct overall scale).
        self.site_kernels = (
            self.kernel.get_site_kernels().to(self.device).detach()
            * (self.sigma2 ** (1.0 / self.seq_length))
        ).detach()

        # ---- posterior assembly terms (cached on the fitted GP) --------------
        # S: root of the posterior data-term, S S^T = Q + (Qy)(Qy)^T.
        # Qy: the mean cache Q y, used to turn (M K_{*,X}) into the posterior mean.
        self.S = model.gp.prediction_strategy.covar_cache.detach()
        self.Qy = model.gp.prediction_strategy.mean_cache.detach()

        # ---- built-in statistic bases (as extended-alphabet subsequences) ----
        # "linear" basis: one row per (position, character) that fixes that single
        # character and averages every other position (all '-').
        self.linear_subseqs = [
            "-" * i + ch + "-" * (self.seq_length - i - 1)
            for i in range(self.seq_length)
            for ch in self.alphabet
        ]
        # "constant" basis: a single all-wildcard row (averages every position) --
        # i.e. the background expectation / mean row.
        self.constant_subseq = ["-" * self.seq_length]

        # Pre-encode the built-in bases once (and their concatenation, 'both').
        self.encoded_linear_subseqs = self._encode_subseqs(self.linear_subseqs)
        self.encoded_constant_subseq = self._encode_subseqs(self.constant_subseq)
        self.encoded_both_subseqs = self._encode_subseqs(self.constant_subseq + self.linear_subseqs)

        # ---- training design matrix + working precision ----------------------
        self._train_x = model.X.to(self.device).detach().clone()
        self._feature_dtype = self._train_x.dtype
        self.comp_dtype = self._feature_dtype  # active compute dtype (see set_precision)

        # Cast all cached tensors to the active compute dtype up front.
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)
        self._train_x = self._train_x.to(self.comp_dtype)

        # GP prior mean, so predictions can include a non-zero offset if present.
        self.prior_mean = KernelTrickComputer._extract_prior_mean(model).to(self.comp_dtype)

    def set_precision(self, use_double: bool = True):
        """Switch the working precision and recast the cached tensors.

        Use float64 (``use_double=True``) when numerical cancellation matters;
        otherwise fall back to the model's native feature dtype.

        Args:
            use_double: if True compute in ``torch.float64``, else in the original
                feature dtype.
        """
        self.comp_dtype = torch.float64 if use_double else self._feature_dtype
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)
        self._train_x = self._train_x.to(self.comp_dtype)
        self.prior_mean = self.prior_mean.to(self.comp_dtype)

    def validate_sequences(self, sequences: Sequence[str], alphabet: str):
        """Assert every subsequence has length ``L`` and uses only allowed characters.

        Args:
            sequences: subsequence strings to check.
            alphabet: the alphabet they must draw from (typically the *extended*
                alphabet, since custom rows may contain the ``'-'`` wildcard).
        """
        alphabet_set = set(alphabet)
        for seq in sequences:
            assert len(seq) == self.seq_length
            assert set(seq).issubset(alphabet_set)

    def validate_pis(self, pis: torch.Tensor):
        """Assert ``pis`` is a single valid, row-stochastic background distribution.

        Unlike the dense class this KeOps variant accepts a single distribution
        only: shape must be exactly ``(seq_length, n_alleles)`` with non-negative
        rows that each sum to 1.

        Args:
            pis: candidate background distribution of shape ``(L, A)``.
        """
        with torch.no_grad():
            pis = pis.to(self.device)
            assert pis.ndim == 2, "pis must have shape (seq_length, n_alleles)"
            L, A = pis.shape
            assert L == self.seq_length
            assert A == self.n_alleles
            assert torch.all(pis >= 0)
            assert torch.allclose(torch.sum(pis, dim=-1), torch.ones(L, device=self.device), atol=1e-6)

    def _encode_subseqs(self, subseqs: Sequence[str]) -> torch.Tensor:
        """One-hot encode subsequences over the extended alphabet, transposed for KeOps.

        Args:
            subseqs: ``r`` subsequence strings over ``extended_alphabet``.

        Returns:
            A contiguous tensor of shape ``(L * extended_size, r)`` -- the extended
            one-hot encoding transposed so the subsequence index is the last axis
            (the layout ``compute_z`` / ``compute_R`` expect).
        """
        import epik_loader
        epik_loader.ensure_epik_loaded()
        from epik.epik.utils import encode_seqs

        encoded = encode_seqs(subseqs, self.extended_alphabet)
        return torch.as_tensor(encoded, device=self.device).T.contiguous()

    def select_site(self, x: torch.Tensor, site: int, num_ch: int):
        """Slice out the ``num_ch`` feature columns belonging to one position.

        Feature vectors are laid out as ``L`` consecutive blocks of ``num_ch``
        channels; this returns the block for a given position.

        Args:
            x: feature tensor whose last axis is ``L * num_ch``.
            site: position index ``p`` in ``[0, L)``.
            num_ch: channels per position (``A`` for training features, ``A+1`` for
                extended subsequence features).

        Returns:
            The ``(..., num_ch)`` slice for position ``site``.
        """
        idx = torch.arange(site * num_ch, (site + 1) * num_ch, device=x.device)
        return x.index_select(-1, idx)

    def _sitewise_prod_covar_func(self, x1, x2, num_ch):
        """Lazy product kernel: product over positions of per-site dot products.

        This is the KeOps realization of the product structure. For inputs laid out
        as ``L`` position-blocks of ``num_ch`` channels each, it returns the lazy
        matrix

            K[i, j] = prod_p  < x1_site_p[i], x2_site_p[j] >,

        i.e. the per-position dot products multiplied across all ``L`` positions.
        No dense ``(len(x1) x len(x2))`` matrix is formed.

        Args:
            x1, x2: feature tensors with last axis ``L * num_ch``.
            num_ch: channels per position.

        Returns:
            A KeOps ``LazyTensor`` representing the kernel matrix.
        """
        # Position 0 initializes the running product...
        x1_site = self.select_site(x1, site=0, num_ch=num_ch)
        x2_site = self.select_site(x2, site=0, num_ch=num_ch)
        K = (LazyTensor(x1_site[:, None, :]) * LazyTensor(x2_site[None, :, :])).sum(-1)

        # ...and each subsequent position multiplies into it.
        for i in range(1, self.seq_length):
            x1_site = self.select_site(x1, site=i, num_ch=num_ch)
            x2_site = self.select_site(x2, site=i, num_ch=num_ch)
            K *= (LazyTensor(x1_site[:, None, :]) * LazyTensor(x2_site[None, :, :])).sum(-1)
        return K

    def get_sitewise_prod_covar_func(self, num_ch):
        """Return a ``covar_func(x1, x2)`` closure over a fixed channel count.

        Args:
            num_ch: channels per position to bind (``A`` or ``A+1``).

        Returns:
            A callable suitable as the ``covar_func`` of a ``KernelLinearOperator``.
        """
        return lambda x1, x2: self._sitewise_prod_covar_func(x1, x2, num_ch)

    def compute_zeta_pc(self, pis: torch.Tensor, eta: float):
        """Per-position, per-character background channel ``zeta^p_i``.

            zeta^p_i = eta * sum_c a^p_{i,c} pi^p_c        (shape (L, A))

        This is the site kernel applied to the background distribution -- the
        contribution of an *averaged* ('-') position. ``eta`` scales the background
        channel (kernel-trick convention).

        Args:
            pis: background distribution ``(L, A)``.
            eta: background-channel scale.

        Returns:
            Tensor of shape ``(L, A)``.
        """
        return eta * torch.einsum("pij,pj->pi", self.site_kernels, pis)

    def compute_zeta_p(self, pis: torch.Tensor, eta: float):
        """Per-position scalar background term ``zeta^p``.

            zeta^p = eta^2 * sum_{c,c'} pi^p_c a^p_{c,c'} pi^p_{c'}    (shape (L,))

        The fully background-averaged (avg . avg) contribution at each position,
        used by the covariance factor. ``eta`` appears squared (two pi channels).

        Args:
            pis: background distribution ``(L, A)``.
            eta: background-channel scale.

        Returns:
            Tensor of shape ``(L,)``.
        """
        return (eta * eta) * torch.einsum("pi,pij,pj->p", pis, self.site_kernels, pis)

    def get_site_specific_values_for_R(self, pis: torch.Tensor, eta: float):
        """Assemble the per-position covariance factor blocks for ``M K M^T``.

        Returns, for each position ``p``, an ``(A+1) x (A+1)`` block indexed by the
        two subsequences' extended-alphabet characters ``(i, j)``. Contracting these
        blocks with two subsequences' encodings and taking the product over positions
        yields ``(M K M^T)_{s,t}``. Each block is the sum of four pieces:

            from_a  : a^p_{i,j}                         (kernel agreement; real chars only)
            from_p  : +/- zeta^p                        (background scalar; sign per wildcard-ness)
            from_sp : -/+ zeta^p_i                      (subsequence s's background channel)
            from_tp : transpose of from_sp              (subsequence t's background channel)

        so block = from_a + from_p + from_sp + from_tp, mirroring the dense factor
        ``Y + U - V - W`` (agreement + background scalar - each subsequence's channel).
        The sign flips on the last (wildcard) row/column implement the fixed-vs-averaged
        distinction: a '-' selects the background channel, a real character selects the
        kernel-agreement channel.

        Args:
            pis: background distribution ``(L, A)``.
            eta: background-channel scale.

        Returns:
            Tensor of shape ``(L, A+1, A+1)``.
        """
        dtype, device = pis.dtype, self.device
        zeta_p = self.compute_zeta_p(pis, eta)    # (L,)   avg.avg scalar
        zeta_pc = self.compute_zeta_pc(pis, eta)  # (L, A) background per character

        # from_p: zeta^p with a +/- sign pattern. Sign is negative iff exactly one of
        # (i, j) is the wildcard column (last index); positive when both or neither are.
        from_p = torch.ones(self.seq_length, self.extended_size, self.extended_size, device=device, dtype=dtype)
        from_p[:, -1, :] *= -1
        from_p[:, :, -1] *= -1
        from_p = zeta_p.view(self.seq_length, 1, 1) * from_p

        # from_sp: subsequence s's background channel zeta^p_i, negated on the real-
        # character columns and zero on the (appended) wildcard row.
        from_sp = zeta_pc[:, :, None] * torch.ones(self.seq_length, self.n_alleles, self.extended_size, device=device, dtype=dtype)
        from_sp[..., :-1] *= -1
        from_sp = torch.cat([from_sp, torch.zeros(self.seq_length, 1, self.extended_size, device=device, dtype=dtype)], dim=-2)
        # from_tp: the same channel for subsequence t -- the transpose of from_sp.
        from_tp = from_sp.transpose(-1, -2)

        # from_a: the site-kernel agreement a^p_{i,j}, only for real characters
        # (the wildcard row/column stay zero here).
        from_a = torch.zeros(self.seq_length, self.extended_size, self.extended_size, device=device, dtype=dtype)
        from_a[:, : self.n_alleles, : self.n_alleles] = self.site_kernels

        return from_p + from_sp + from_tp + from_a

    def get_site_specific_values_for_z(self, pis: torch.Tensor, eta: float):
        """Assemble the per-position mean factor blocks for ``M K_{*,X}``.

        Returns, for each position ``p``, an ``A x (A+1)`` block mapping a subsequence's
        extended one-hot column to the per-position factor seen by each training-feature
        channel ``i``:

            column c < A (a fixed character): a^p_{i,c} - zeta^p_i
            column A     (the '-' wildcard) : + zeta^p_i

        In words: a fixed position contributes the kernel agreement minus the background
        channel, and an averaged ('-') position contributes just the background channel.

        Args:
            pis: background distribution ``(L, A)``.
            eta: background-channel scale.

        Returns:
            Tensor of shape ``(L, A, A+1)``.
        """
        dtype, device = pis.dtype, self.device
        zetas = self.compute_zeta_pc(pis, eta)  # (L, A) background per character

        # Broadcast zeta across the extended-alphabet column axis, then negate every
        # real-character column (leaving the wildcard column positive).
        A_mat = zetas[:, :, None] * torch.ones(self.seq_length, self.n_alleles, self.extended_size, device=device, dtype=dtype)
        A_mat[..., :-1] *= -1

        # Add the site kernel on the real-character columns (wildcard column stays 0).
        kernel_to_add = torch.cat([self.site_kernels, torch.zeros(self.seq_length, self.n_alleles, 1, device=device, dtype=dtype)], dim=-1)
        return A_mat + kernel_to_add

    def compute_z(self, pis: torch.Tensor, eta: float, subseqs: torch.Tensor):
        """Build ``z = M K_{*,X}`` lazily as a ``KernelLinearOperator``.

        Contracts each subsequence's extended encoding with the per-position mean
        blocks to produce a transformed feature matrix, then wraps it in a lazy
        product-kernel operator against the training design. Realizing ``Qy @ z``
        gives the posterior mean.

        Args:
            pis: background distribution ``(L, A)``.
            eta: background-channel scale.
            subseqs: encoded subsequences, shape ``(L * (A+1), r)``.

        Returns:
            A ``KernelLinearOperator`` representing ``M K_{*,X}`` (subseqs x train).
        """
        M_blocks = self.get_site_specific_values_for_z(pis, eta)                 # (L, A, A+1)
        subseqs = subseqs.contiguous().view(self.seq_length, self.extended_size, -1)  # (L, A+1, r)
        # For each position, map the subsequence's extended one-hot to per-channel factors.
        transformed = torch.einsum("pij,pjk->pik", M_blocks, subseqs)            # (L, A, r)
        # Flatten positions into the KeOps feature layout (r rows of L*A channels).
        transformed = transformed.view(self.seq_length * self.n_alleles, -1).T.contiguous()  # (r, L*A)
        return KernelLinearOperator(self._train_x, transformed, covar_func=self.get_sitewise_prod_covar_func(self.n_alleles))

    def compute_R(self, pis: torch.Tensor, eta: float, encoded_subseqs: torch.Tensor):
        """Build ``R = M K M^T`` lazily as a ``KernelLinearOperator``.

        Contracts each subsequence's extended encoding with the per-position
        covariance blocks, then wraps the result in a lazy product-kernel operator
        so the ``(r x r)`` matrix is only formed on demand.

        Args:
            pis: background distribution ``(L, A)``.
            eta: background-channel scale.
            encoded_subseqs: encoded subsequences, shape ``(L * (A+1), r)``.

        Returns:
            A ``KernelLinearOperator`` representing ``M K M^T`` (subseqs x subseqs).
        """
        x = encoded_subseqs.T.contiguous()                                       # (r, L*(A+1))
        x_blocks = x.reshape(x.size(0), self.seq_length, self.extended_size)      # (r, L, A+1)
        R_blocks = self.get_site_specific_values_for_R(pis, eta)                  # (L, A+1, A+1)

        # Contract each subsequence through the per-position (A+1)x(A+1) blocks.
        prod = torch.einsum("slj,ljk->slk", x_blocks, R_blocks).reshape(x.size(0), -1).contiguous()  # (r, L*(A+1))
        return KernelLinearOperator(prod, x, covar_func=self.get_sitewise_prod_covar_func(self.extended_size))

    def gauge_fixed_posterior(self, pis: torch.Tensor, eta: float, subseqs: Sequence[str] = (), subseq_type: str = 'custom', calc_covar: bool = False, validate: bool = True):
        """Unified gauge-fixed posterior interface for the KeOps singleton variant.

        Replaces the individual ``_linear`` / ``_constant`` implementations with a
        single entry point. Computes the posterior mean ``theta_bar = M K_{*,X} Q y``
        and, if requested, the covariance ``R = M K M^T - (M K_{*,X}) S S^T (M K_{*,X})^T``.

        Args:
            pis: background distribution ``(L, A)``, row-stochastic.
            eta: background-channel scale (typically 1.0).
            subseqs: custom subsequence rows (used only when ``subseq_type='custom'``).
            subseq_type: which statistic basis to use --
                ``'custom'`` (use ``subseqs``), ``'linear'``, ``'constant'``, or ``'both'``.
            calc_covar: if True, also return the posterior covariance.
            validate: if True, validate ``pis`` (and custom subseqs) first.

        Returns:
            ``mean`` of shape ``(r,)`` if ``calc_covar`` is False, else ``(mean, cov)``
            with ``cov`` a dense ``(r, r)`` tensor.

        Raises:
            ValueError: if ``subseq_type`` is not one of the recognized options.
        """
        if validate:
            self.validate_pis(pis)

        # Resolve which encoded subsequence basis to use.
        if subseq_type == 'custom':
            raw_subseqs = tuple(subseqs)
            if validate: self.validate_sequences(raw_subseqs, self.extended_alphabet)
            encoded_subseqs = self._encode_subseqs(raw_subseqs)
        elif subseq_type == 'linear':
            encoded_subseqs = self.encoded_linear_subseqs
        elif subseq_type == 'constant':
            encoded_subseqs = self.encoded_constant_subseq
        elif subseq_type == 'both':
            encoded_subseqs = self.encoded_both_subseqs
        else:
            raise ValueError(f"Unknown subseq_type '{subseq_type}'.")

        # ---- posterior mean: theta_bar = Qy @ (M K_{*,X}) --------------------
        dtype = self.comp_dtype
        pis = pis.to(device=self.device, dtype=dtype).contiguous()
        z = self.compute_z(pis, eta, encoded_subseqs.to(dtype))   # lazy (r x train)
        mean = self.Qy.to(dtype) @ z

        if not calc_covar:
            return mean

        # ---- posterior covariance: R - (z S)(z S)^T --------------------------
        # R = M K M^T (prior term); the subtracted term is the data correction,
        # using S with S S^T = Q + (Qy)(Qy)^T.
        R = self.compute_R(pis.to(dtype), eta, encoded_subseqs.to(dtype))  # lazy (r x r)
        zS = z.T @ self.S.to(dtype)
        cov = (R - (zS @ zS.T)).to_dense()
        return mean, cov