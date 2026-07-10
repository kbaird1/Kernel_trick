import torch
from typing import List, Sequence, Any
from pykeops.torch import LazyTensor
from linear_operator.operators import KernelLinearOperator
from gpytorch.distributions import MultivariateNormal


def compute_LTDR_lazy(cL, cD, cR):
    """
    Returns a LazyTensor K of shape (m, n, 1),
    where K[m,n] = prod_p sum_a cL[m,p,a] * cD[p,a] * cR[n,p,a].
    Implemented as a product of per-position KeOps dot-products with valid shapes.
    """
    # Ensure alignment and contiguity
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

        x_i = LazyTensor(x[:, None, :])           # (m,1,A)
        y_j = LazyTensor(y[None, :, :])           # (1,n,A)

        Kp = (x_i * y_j).sum(-1)                  # (m,n,1) LazyTensor
        K = Kp if K is None else (K * Kp)         # product across positions

    return K  # (m,n,1) LazyTensor


def quadratic_form_from_lazy(K, z):
    """
    Computes z^T K z without materializing K:
    v = K @ z via KeOps, then return z^T v.
    """
    z = z.contiguous().view(-1)               # (m,)
    z_j = LazyTensor(z[None, :, None])        # (1,n,1)
    v = (K * z_j).sum(dim=1).squeeze(-1)      # (m,)
    return (v * z).sum()


def compute_LTDR(cL,cD,cR):
    # cL is m x L x A 
    # cD is L x A
    # cR is m x L x A
    # L is m x A^L L_mx = \prod_p cL_m,p,x_p
    # R is m x A^L R_mx = \prod_p cR_m,p,x_p
    # D is A^L x A^L diagonal D_xx = \prod_p cD_x_p 
    # computes L^T D R by without explicitly constructing L, D, or R
    if cL is None:
        out = torch.einsum('la,nla->nl',cD,cR)
    else:
        out = torch.einsum('mla,la,nla->mnl',cL,cD,cR)
    out = torch.prod(out,dim=-1)
    return out

class KernelTrickComputer:
    """
    KernelTrickComputer: encapsulates kernel-site helpers and gauge-fixed posterior
    Uses the kernel object exposed by your EpiK model instances.
    Device-aware: all tensors are on the model's device.
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

        self.encoded_linear_subseqs = self._encode_subseqs(self.linear_subseqs)
        self.encoded_constant_subseq = self._encode_subseqs(self.constant_subseq)

        self._train_x = model.X.to(self.device).detach().clone()
        self._feature_dtype = self._train_x.dtype
        # new: computation dtype toggle
        self.comp_dtype = self._feature_dtype
        # ensure caches have consistent dtype
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)

        self.prior_mean = self._extract_prior_mean(model)


    # get a set of eyes and check on this to see what it is doing and how it works
    
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
        """Toggle internal computation precision and cast caches."""
        self.comp_dtype = torch.float64 if use_double else self._feature_dtype
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)
        self.prior_mean = self.prior_mean.to(self.comp_dtype)
        self._train_x = self._train_x.to(self.comp_dtype)

    # ---------- validation ----------
    def validate_sequences(self, sequences: Sequence[str], alphabet: str):
        alphabet_set = set(alphabet)
        for seq in sequences:
            assert len(seq) == self.seq_length
            assert set(seq).issubset(alphabet_set)

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
            assert torch.all(pis >= 0), (print(f"pis must be non-negative, got:\n{pis}"), f"pis must be non-negative, got:\n{pis}")[1]
            assert torch.allclose(
                torch.sum(pis, dim=-1),
                torch.ones(B, L, device=pis.device, dtype=pis.dtype),
                atol=1e-6,
            )

    def _as_batch_pis(self, pis: torch.Tensor):
        """
        Normalize `pis` to batched shape (B, L, A).

        Returns:
          pis_batched, squeeze_batch_dim
        where `squeeze_batch_dim=True` means the caller provided a single (L, A)
        and we should squeeze the leading batch dimension on outputs.
        """
        if pis.ndim == 2:
            return pis.unsqueeze(0), True
        if pis.ndim == 3:
            return pis, False
        raise ValueError(
            f"pis must have shape (seq_length, n_alleles) or (B, seq_length, n_alleles), got {tuple(pis.shape)}"
        )

    @staticmethod
    def _maybe_squeeze_batch(out, squeeze_batch_dim: bool):
        if not squeeze_batch_dim:
            return out
        if isinstance(out, tuple):
            return tuple(x.squeeze(0) for x in out)
        return out.squeeze(0)

    def _encode_subseqs(self, subseqs: Sequence[str]) -> torch.Tensor:
        """Encode subsequences and reshape to (n_subseq, seq_length, n_alleles+1)."""
        # Lazy-import from the selected epik checkout (fork vs dev).
        import epik_loader

        epik_loader.ensure_epik_loaded()
        from epik.epik.utils import encode_seqs

        encoded = encode_seqs(subseqs, self.extended_alphabet)
        encoded = torch.as_tensor(encoded, device=self.device)
        return encoded.reshape(-1, self.seq_length, self.n_alleles+1).contiguous()
    
    # ---------- main computational routines ----------

    def compute_zeta_pc(self, pis: torch.Tensor, eta: float):
        return eta * torch.einsum("pij,bpj->bpi", self.site_kernels, pis)

    def compute_zeta_p(self, pis1: torch.Tensor, pis2: torch.Tensor, eta: float):
        return (eta * eta) * torch.einsum("bpi,pij,bpj->bp", pis1, self.site_kernels, pis2)

    def compute_R(self, pis1: torch.Tensor, pis2: torch.Tensor, eta: float, subseqs: torch.Tensor):
       S = subseqs # r X L X (A+1)
       zetas_pc_1 = self.compute_zeta_pc(pis1, eta) # b x L x A
       zetas_pc_2 = self.compute_zeta_pc(pis2, eta) # b x L x A
       zeta_p = self.compute_zeta_p(pis1, pis2, eta) # b x L 

       A = self.site_kernels # L x A x A
       
       # contribution from K, a^p_(s_p,t_p) if they agree on a character in position p
       Y = torch.einsum("rla,lab,qlb->lrq", S[...,:-1], A, S[...,:-1])[None,...]  #b x L x r x r
       
       # contributions from zeta_p; appears in all terms 
       U = zeta_p[:, :, None, None] # b x L x 1 x 1
       
       # contribution from zeta_pc for s_p
       V = torch.einsum('bla,rla->blr', zetas_pc_1, S[...,:-1])  # b x L x r
       V = V[:,:,None,:]  # b x L x 1 x r
       
       # contribution from zeta_pc for t_p
       W = torch.einsum('bla,rla->blr', zetas_pc_2, S[...,:-1])  # b x L x r
       W = W[:,:,:, None]  # b x L x r x 1
       
       R1 = Y + U - W - V  # b x L x r x r
       M = (S[:, None, :, -1] == S[None, :, :, -1])  # r x r x L
       M = (2 * M - 1).permute(2, 0, 1).unsqueeze(0)  # 1 x L x r x r
       R = torch.prod(R1*M, dim=1)  # b x r x r
       return R # b x r x r
    
    def compute_R_various_pis_same_subseqs(self, pis: torch.Tensor, eta: float, subseqs: torch.Tensor):
        r = subseqs.shape[0]
        k = pis.shape[0]
        R = torch.zeros(k*r, k*r, device=self.device, dtype=self.site_kernels.dtype)
        for i in range(k):
            for j in range(k):
                R[i*r:(i+1)*r, j*r:(j+1)*r] = self.compute_R(pis[i:i+1], pis[j:j+1], eta, subseqs)
        return R
    
    def compute_z(self, pis: torch.Tensor, eta: float, subseqs: torch.Tensor):
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
       Z = torch.prod(Z1, dim=1)  # B x t x r
       return Z


######
###START HERE#######

#######
    def _gauge_fixed_posterior(
        self, pis: torch.Tensor, eta: float = 1.0, encoded_subseqs: torch.Tensor = None, calc_covar: bool = False
    ):
        dtype = self.comp_dtype
        pis = pis.to(device=self.device, dtype=dtype).contiguous()
        z = self.compute_z(pis, eta, encoded_subseqs.to(dtype))  # B x t x r
        mean = self.Qy.to(dtype) @ z  # (t) @ (B x t x r) -> B x r
        if not calc_covar:
            return mean
        else: 
            R = self.compute_R(pis, pis, eta, encoded_subseqs.to(dtype))  # b x r x r
            # contract along training dim to get b x r x r
            zS = torch.einsum('btr,tk->brk', z, self.S.to(dtype))  # b x r x t
            cov = R - torch.bmm(zS, zS.transpose(1, 2))  # b x r x r
            return mean, cov

    def _gauge_fixed_posterior_various_pis_same_subseqs(self, pis: torch.Tensor, eta: float = 1.0, encoded_subseqs: torch.Tensor = None, calc_covar: bool = False):
        dtype = self.comp_dtype
        pis = pis.to(device=self.device, dtype=dtype).contiguous()
        r = encoded_subseqs.shape[0]
        t = self._train_x.shape[0]
        B = pis.shape[0]
        z = torch.zeros(B*r, t, device=self.device, dtype=dtype) # B*r x t
        for i in range(pis.shape[0]):
            z[i*r:(i+1)*r, :] = self.compute_z(pis[i:i+1], eta, encoded_subseqs.to(dtype))[0,...].T # r x t
        #z = self.compute_z(pis[i], eta, encoded_subseqs.to(dtype))  # B x t x r
        # flatten batch dim
        #z = torch.flatten(z.permute(1, 0, 2), start_dim=1)  # t x (B*r)
        z = z.T  # t x (B*r)
        mean = self.Qy.to(dtype) @ z  # (B*r)
        if not calc_covar:
            return mean
        R = self.compute_R_various_pis_same_subseqs(pis, eta, encoded_subseqs.to(dtype))  # (B*r) x (B*r)
        zS = z.T @ self.S.to(dtype)  # (B*r) x t
        #print(zS.shape,R.shape)
        cov = R - zS @ zS.T# (B*r) x (B*r)
        return mean, cov


    def compute_R_various_pis_same_subseqs(self, pis: torch.Tensor, eta: float = 1.0, subseqs: torch.Tensor = None):
        r = subseqs.shape[0]
        k = pis.shape[0]
        R = torch.zeros(k*r, k*r, device=self.device, dtype=self.site_kernels.dtype)
        for i in range(k):
            for j in range(k):
                R[i*r:(i+1)*r, j*r:(j+1)*r] = self.compute_R(pis[i:i+1], pis[j:j+1], eta, subseqs)
        return R
    
    # GAUGE FIXED POSTERIOR FUNCTIONS FOR INDIVIDUAL PIS (batched)

    def gauge_fixed_posterior(
        self,
        pis: torch.Tensor,
        eta: float = 1.0,
        subseqs: Sequence[str] = (),
        calc_covar: bool = False,
        validate: bool = True,
        max_batch: int = None,
        max_subseq_batch: int = None,
    ):
        """
        If max_batch is provided and smaller than the number of pis, the computation
        is internally split along the batch dimension and stitched back together.

        If max_subseq_batch is provided and smaller than the number of subseqs, the
        computation is split along the subseq dimension (not supported with
        calc_covar=True).
        """
        raw_subseqs = tuple(subseqs)
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)
            self.validate_sequences(raw_subseqs, self.extended_alphabet)

        n_subseqs = len(raw_subseqs)
        if (
            max_subseq_batch is not None
            and max_subseq_batch > 0
            and n_subseqs > max_subseq_batch
        ):
            if calc_covar:
                raise ValueError(
                    "max_subseq_batch is not supported when calc_covar=True"
                )
            chunks = []
            for start in range(0, n_subseqs, max_subseq_batch):
                chunk_subseqs = raw_subseqs[start : start + max_subseq_batch]
                chunk_out = self.gauge_fixed_posterior(
                    pis,
                    eta=eta,
                    subseqs=chunk_subseqs,
                    calc_covar=calc_covar,
                    validate=False,
                    max_batch=max_batch,
                    max_subseq_batch=None,
                )
                chunks.append(chunk_out)
            out = torch.cat(chunks, dim=-1)
            return self._maybe_squeeze_batch(out, squeeze_batch_dim)

        encoded_subseqs = self._encode_subseqs(raw_subseqs)

        B = pis_batched.shape[0]
        if max_batch is None or max_batch <= 0 or max_batch >= B:
            out = self._gauge_fixed_posterior(pis_batched, eta, encoded_subseqs, calc_covar)
            return self._maybe_squeeze_batch(out, squeeze_batch_dim)

        chunks = []
        start = 0
        while start < B:
            end = min(start + max_batch, B)
            chunk_pis = pis_batched[start:end]
            chunk_out = self._gauge_fixed_posterior(
                chunk_pis,
                eta,
                encoded_subseqs,
                calc_covar,
            )
            chunks.append(chunk_out)
            start = end

        if isinstance(chunks[0], tuple):
            out = tuple(torch.cat([c[i] for c in chunks], dim=0) for i in range(len(chunks[0])))
        else:
            out = torch.cat(chunks, dim=0)

        return self._maybe_squeeze_batch(out, squeeze_batch_dim)

    def gauge_fixed_posterior_linear(
        self,
        pis: torch.Tensor,
        eta: float = 1.0,
        calc_covar: bool = False,
        validate: bool = True,
        max_batch: int = None,
    ):
        """
        Gauge-fixed posterior for linear subseqs.

        If max_batch is provided and smaller than the number of pis, the computation
        is internally split along the batch dimension and stitched back together to
        avoid OOM. Callers never need to manage chunks themselves.
        """
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)

        B = pis_batched.shape[0]
        if max_batch is None or max_batch <= 0 or max_batch >= B:
            out = self._gauge_fixed_posterior(pis_batched, eta, self.encoded_linear_subseqs, calc_covar)
            return self._maybe_squeeze_batch(out, squeeze_batch_dim)

        chunks = []
        start = 0
        while start < B:
            end = min(start + max_batch, B)
            chunk_pis = pis_batched[start:end]
            chunk_out = self._gauge_fixed_posterior(
                chunk_pis,
                eta,
                self.encoded_linear_subseqs,
                calc_covar,
            )
            chunks.append(chunk_out)
            start = end

        if isinstance(chunks[0], tuple):
            out = tuple(torch.cat([c[i] for c in chunks], dim=0) for i in range(len(chunks[0])))
        else:
            out = torch.cat(chunks, dim=0)

        return self._maybe_squeeze_batch(out, squeeze_batch_dim)


    def gauge_fixed_posterior_constant(
        self,
        pis: torch.Tensor,
        eta: float = 1.0,
        calc_covar: bool = False,
        validate: bool = True,
        max_batch: int = None,
    ):
        """
        Gauge-fixed posterior for constant subseqs.

        If max_batch is provided and smaller than the number of pis, the computation
        is internally split along the batch dimension and stitched back together to
        avoid OOM. Callers never need to manage chunks themselves.
        """
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)

        B = pis_batched.shape[0]
        if max_batch is None or max_batch <= 0 or max_batch >= B:
            out = self._gauge_fixed_posterior(pis_batched, eta, self.encoded_constant_subseq, calc_covar)
            return self._maybe_squeeze_batch(out, squeeze_batch_dim)

        chunks = []
        start = 0
        while start < B:
            end = min(start + max_batch, B)
            chunk_pis = pis_batched[start:end]
            chunk_out = self._gauge_fixed_posterior(
                chunk_pis,
                eta,
                self.encoded_constant_subseq,
                calc_covar,
            )
            chunks.append(chunk_out)
            start = end

        if isinstance(chunks[0], tuple):
            out = tuple(torch.cat([c[i] for c in chunks], dim=0) for i in range(len(chunks[0])))
        else:
            out = torch.cat(chunks, dim=0)

        return self._maybe_squeeze_batch(out, squeeze_batch_dim)
    
    def gauge_fixed_posterior_constant_linear(self, pis: torch.Tensor, eta: float = 1.0, calc_covar: bool = False, validate: bool = True):
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)
        subseqs = self.constant_subseq + self.linear_subseqs 
        encoded_subseqs = self._encode_subseqs(subseqs)
        out = self._gauge_fixed_posterior(pis_batched, eta, encoded_subseqs, calc_covar)
        return self._maybe_squeeze_batch(out, squeeze_batch_dim)

    # GAUGE FIXED POSTERIOR FOR MULTIPLE PIS; COMPUTE COVARIANCE ACROSS PIS

    def gauge_fixed_posterior_various_pis(self, pis: torch.Tensor, eta: float = 1.0, subseqs: Sequence[str] = (), calc_covar: bool = False, validate: bool = True
    ):
        raw_subseqs = tuple(subseqs)
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)
            self.validate_sequences(raw_subseqs, self.extended_alphabet)
        encoded_subseqs = self._encode_subseqs(raw_subseqs)
        out = self._gauge_fixed_posterior_various_pis_same_subseqs(pis_batched, eta, encoded_subseqs, calc_covar)
        return self._maybe_squeeze_batch(out, squeeze_batch_dim)
    
    def gauge_fixed_posterior_various_pis_linear(self, pis: torch.Tensor, eta: float = 1.0, calc_covar: bool = False, validate: bool = True):
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)
        out = self._gauge_fixed_posterior_various_pis_same_subseqs(pis_batched, eta, self.encoded_linear_subseqs, calc_covar)
        return self._maybe_squeeze_batch(out, squeeze_batch_dim)
    
    def gauge_fixed_posterior_various_pis_constant(self, pis: torch.Tensor, eta: float = 1.0, calc_covar: bool = False, validate: bool = True):
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)
        out = self._gauge_fixed_posterior_various_pis_same_subseqs(pis_batched, eta, self.encoded_constant_subseq, calc_covar)
        return self._maybe_squeeze_batch(out, squeeze_batch_dim)
    
    def gauge_fixed_posterior_various_pis_constant_linear(self, pis: torch.Tensor, eta: float = 1.0, calc_covar: bool = False, validate: bool = True):
        pis_batched, squeeze_batch_dim = self._as_batch_pis(pis)
        if validate:
            self.validate_pis(pis_batched)
        subseqs = self.constant_subseq + self.linear_subseqs
        encoded_subseqs = self._encode_subseqs(subseqs)
        out = self._gauge_fixed_posterior_various_pis_same_subseqs(pis_batched, eta, encoded_subseqs, calc_covar)
        return self._maybe_squeeze_batch(out, squeeze_batch_dim)
   
    # FUNCTIONS FOR COMPUTING VARIANCE EXPLAINED


    def _kx_and_qy(self, dtype):
        """Build KX (kernel-mapped training points) and the matching Qy weights."""
        train_x = self._train_x
        qy = self.Qy
        KX = torch.einsum(
            "lab,tla -> tlb",
            self.site_kernels.to(dtype),
            train_x.reshape(-1, self.seq_length, self.n_alleles).to(dtype),
        )
        return KX, qy.to(dtype)

    def E_MAP(self, pi):
        # dtype-aware: compute in the dtype of pi
        dtype = pi.dtype
        KX, qy = self._kx_and_qy(dtype)
        DKX = compute_LTDR(None, pi, KX)
        return DKX @ qy
    
    def E_MAP_2(self, pi, keops: bool = False):
        # dtype-aware: compute in the dtype of pi
        dtype = pi.dtype
        KX, qy = self._kx_and_qy(dtype)

        if keops:
            # Build K as a LazyTensor directly (no log space)
            K_lazy = compute_LTDR_lazy(KX, pi, KX)
            return quadratic_form_from_lazy(K_lazy, qy)
        else:
            KXTDKX = compute_LTDR(KX, pi, KX)
            return qy @ KXTDKX @ qy
        
    def _E_TRUNC(self, pi, weights, encoded_subseqs):
        # encoded subseqs r x L x (A+1)
        dtype = pi.dtype
        C = (encoded_subseqs[...,:-1] + encoded_subseqs[...,-1:]).to(dtype)  # r x L x A
        DC = compute_LTDR(None, pi, C) # r x r
        return DC @ weights.to(dtype)
    
    def E_TRUNC(self, pi, weights, subseqs):
        encoded_subseqs = self._encode_subseqs(subseqs)
        return self._E_TRUNC( pi, weights, encoded_subseqs)
    
    def _E_TRUNC_2(self, pi, weights, encoded_subseqs):
        # encoded subseqs r x L x (A+1)
        dtype = pi.dtype
        C = (encoded_subseqs[...,:-1] + encoded_subseqs[...,-1:]).to(dtype)  # r x L x A
        CTDC = compute_LTDR(C, pi, C) # r x r
        w = weights.to(dtype)
        return w @ CTDC @ w
    
    def E_TRUNC_2(self, pi, weights, subseqs):
        encoded_subseqs = self._encode_subseqs(subseqs)
        return self._E_TRUNC_2( pi, weights, encoded_subseqs)
    
    def _E_MAP_TRUNC(self, pi, weights, encoded_subseqs):
        dtype = pi.dtype
        C = (encoded_subseqs[...,:-1] + encoded_subseqs[...,-1:]).to(dtype)
        KX, qy = self._kx_and_qy(dtype)
        CTDKX = compute_LTDR(C, pi, KX)
        return weights.to(dtype) @ CTDKX @ qy

    def E_MAP_TRUNC(self, pi, weights, subseqs):
        encoded_subseqs = self._encode_subseqs(subseqs)
        return self._E_MAP_TRUNC(pi, weights, encoded_subseqs)
    
    def variance_explained(self, pi, weights, subseqs, use_double: bool = True, keops: bool = False):
        """
        Compute variance explained with optional double precision.
        When use_double=True, performs all internal ops in float64.
        Otherwise, uses the feature dtype (typically float32).
        """
        comp_dtype = torch.float64 if use_double else self._feature_dtype
        pi = pi.to(comp_dtype)
        weights = weights.to(comp_dtype)
        # compute all components in selected dtype
        E_map_2     = self.E_MAP_2(pi, keops=keops).to(comp_dtype)
        E_trunc     = self.E_TRUNC(pi, weights, subseqs).to(comp_dtype)
        E_trunc_2   = self.E_TRUNC_2(pi, weights, subseqs).to(comp_dtype)
        E_map_trunc = self.E_MAP_TRUNC(pi, weights, subseqs).to(comp_dtype)
        E_map       = self.E_MAP(pi).to(comp_dtype)
#        E_map_2     = self.E_MAP_2(pi).to(comp_dtype)
        den  = (E_map_2 - E_map * E_map)
        num1 = (E_trunc_2 + E_map_2 - 2.0 * E_map_trunc)
        num2 = (E_trunc - E_map) * (E_trunc - E_map)
        ve = 1.0 - ((num1 - num2) / den)
        return ve if use_double else ve.to(self._feature_dtype)


    def MSE(self, pi, weights, subseqs, use_double: bool = True, keops: bool = False):
        """
        Compute the analytic MSE = Var_pi(g - h), where g is pi's linear model
        and h is the GP posterior mean.

        When use_double=True, performs all internal ops in float64.
        Otherwise, uses the feature dtype (typically float32).
        """
        comp_dtype = torch.float64 if use_double else self._feature_dtype
        pi = pi.to(comp_dtype)
        weights = weights.to(comp_dtype)
        # compute all components in selected dtype
        E_map_2     = self.E_MAP_2(pi, keops=keops).to(comp_dtype)
        E_trunc     = self.E_TRUNC(pi, weights, subseqs).to(comp_dtype)
        E_trunc_2   = self.E_TRUNC_2(pi, weights, subseqs).to(comp_dtype)
        E_map_trunc = self.E_MAP_TRUNC(pi, weights, subseqs).to(comp_dtype)
        E_map       = self.E_MAP(pi).to(comp_dtype)
#       E_map_2     = self.E_MAP_2(pi).to(comp_dtype)
       
        num1 = (E_trunc_2 + E_map_2 - 2.0 * E_map_trunc)
        num2 = (E_trunc - E_map) * (E_trunc - E_map)
        num = num1 - num2
        return  num if use_double else num.to(self._feature_dtype)

    def variance_explained_pieces(self, pi, weights, subseqs, use_double: bool = True, keops: bool = False):
        """
        Compute variance explained with optional double precision.
        When use_double=True, performs all internal ops in float64.
        Otherwise, uses the feature dtype (typically float32).
        """
        comp_dtype = torch.float64 if use_double else self._feature_dtype
        pi = pi.to(comp_dtype)
        weights = weights.to(comp_dtype)
        # compute all components in selected dtype
        E_map_2     = self.E_MAP_2(pi, keops=keops).to(comp_dtype)
        E_trunc     = self.E_TRUNC(pi, weights, subseqs).to(comp_dtype)
        E_trunc_2   = self.E_TRUNC_2(pi, weights, subseqs).to(comp_dtype)
        E_map_trunc = self.E_MAP_TRUNC(pi, weights, subseqs).to(comp_dtype)
        E_map       = self.E_MAP(pi).to(comp_dtype)
#        E_map_2     = self.E_MAP_2(pi).to(comp_dtype)
        den  = (E_map_2 - E_map * E_map)
        num1 = (E_trunc_2 + E_map_2 - 2.0 * E_map_trunc)
        num2 = (E_trunc - E_map) * (E_trunc - E_map)
        num = num1 - num2
        print(num.item(), den.item())
        return den, num if use_double else (den.to(self._feature_dtype), num.to(self._feature_dtype))
    
    def compute_ve_matrix(self, pis, use_double: bool = True, keops: bool = False):
        """ve_mat[i,j] = variance explained in region pi_j by linear model from pi_i  """
        k = pis.shape[0]
        ve_mat = torch.zeros(k,k)
        subseqs = self.linear_subseqs + self.constant_subseq
        weights = self.gauge_fixed_posterior(pis, eta =1 , subseqs= subseqs)
        for i in range(k):
            for j in range(k):
                ve_mat[i,j] = self.variance_explained(pis[j], weights[i], subseqs, use_double= use_double, keops=keops)
        return ve_mat
    
    def compute_MSE_matrix(self, pis, use_double: bool = True, keops: bool = False):
        """ve_mat[i,j] = variance explained in region pi_j by linear model from pi_i  """
        k = pis.shape[0]
        ve_mat = torch.zeros(k,k)
        subseqs = self.linear_subseqs + self.constant_subseq
        weights = self.gauge_fixed_posterior(pis, eta =1 , subseqs= subseqs)
        for i in range(k):
            for j in range(k):
                ve_mat[i,j] = self.MSE(pis[j], weights[i], subseqs, use_double= use_double, keops=keops)
        return ve_mat

    def compute_MSE(self, pis_region, pis_model, use_double: bool = True, keops: bool = False):
        """MSE in region pi_region by linear model from pi_model."""
        subseqs = self.linear_subseqs + self.constant_subseq
        weights = self.gauge_fixed_posterior(pis_model, eta =1 , subseqs= subseqs)
        return self.MSE(pis_region, weights.squeeze(), subseqs, use_double= use_double, keops=keops)

    def compute_ve_matrix_diag(self, pis, use_double: bool = True, keops: bool = False):
        """ve_mat[i,j] = variance explained in region pi_j by linear model from pi_i  """
        k = pis.shape[0]
        ve_mat = torch.zeros(k)
        subseqs = self.linear_subseqs + self.constant_subseq
        weights = self.gauge_fixed_posterior(pis, eta =1 , subseqs= subseqs)
        for i in range(k):
            ve_mat[i] = self.variance_explained(pis[i], weights[i], subseqs, use_double= use_double, keops=keops)
        return ve_mat
    
    def compute_MSE_matrix_diag(self, pis, use_double: bool = True, keops: bool = False):
        """ve_mat[i,j] = variance explained in region pi_j by linear model from pi_i  """
        k = pis.shape[0]
        ve_mat = torch.zeros(k)
        subseqs = self.linear_subseqs + self.constant_subseq
        weights = self.gauge_fixed_posterior(pis, eta =1 , subseqs= subseqs)
        for i in range(k):
            ve_mat[i] = self.MSE(pis[i], weights[i], subseqs, use_double= use_double, keops=keops)
        return ve_mat

    
class KernelTrickComputerSingletonKeOps:
    """
    KernelTrickComputer: encapsulates kernel-site helpers and gauge-fixed posterior
    Uses the kernel object exposed by your EpiK model instances.
    Device-aware: all tensors are on the model's device.
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
        self.sigma2 = torch.exp(self.kernel.log_var.to(self.device)).detach()
        self.site_kernels = (
            self.kernel.get_site_kernels().to(self.device).detach()
            * (self.sigma2 ** (1.0 / self.seq_length))
        ).detach()

        self.S = model.gp.prediction_strategy.covar_cache.detach()
        self.Qy = model.gp.prediction_strategy.mean_cache.detach()

        self.linear_subseqs = [
            "-" * i + ch + "-" * (self.seq_length - i - 1)
            for i in range(self.seq_length)
            for ch in self.alphabet
        ]
        self.constant_subseq = ["-" * self.seq_length]

        self.encoded_linear_subseqs = self._encode_subseqs(self.linear_subseqs)
        self.encoded_constant_subseq = self._encode_subseqs(self.constant_subseq)

        self._train_x = model.X.to(self.device).detach().clone()
        self._feature_dtype = self._train_x.dtype
        # new: computation dtype toggle
        self.comp_dtype = self._feature_dtype
        # cast caches
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)
        self._train_x = self._train_x.to(self.comp_dtype)

        self.prior_mean = KernelTrickComputer._extract_prior_mean(model)

    def set_precision(self, use_double: bool = True):
        """Toggle internal computation precision and cast caches."""
        self.comp_dtype = torch.float64 if use_double else self._feature_dtype
        self.S = self.S.to(self.comp_dtype)
        self.Qy = self.Qy.to(self.comp_dtype)
        self.site_kernels = self.site_kernels.to(self.comp_dtype)
        self._train_x = self._train_x.to(self.comp_dtype)
        self.prior_mean = self.prior_mean.to(self.comp_dtype)

    # ---------- validation ----------
    def validate_sequences(self, sequences: Sequence[str], alphabet: str):
        alphabet_set = set(alphabet)
        for seq in sequences:
            assert len(seq) == self.seq_length
            assert set(seq).issubset(alphabet_set)

    def validate_pis(self, pis: torch.Tensor):
        with torch.no_grad():
            pis = pis.to(self.device)
            assert pis.ndim == 2, "pis must have shape (seq_length, n_alleles)"
            L, A = pis.shape
            assert L == self.seq_length
            assert A == self.n_alleles
            assert torch.all(pis >= 0)
            assert torch.allclose(
                torch.sum(pis, dim=-1), torch.ones(L, device=self.device), atol=1e-6
            )

    def _encode_subseqs(self, subseqs: Sequence[str]) -> torch.Tensor:
        """Encode subsequences and reshape to (seq_length*extended_size, n_subseq)."""
        encoded = encode_seqs(subseqs, self.extended_alphabet)
        encoded = torch.as_tensor(encoded, device=self.device)
        return encoded.T.contiguous()
    
     # ---------- selection ----------
    def select_site(self, x: torch.Tensor, site: int, num_ch: int):
        idx = torch.arange(site * num_ch, (site + 1) * num_ch, device=x.device)
        return x.index_select(-1, idx)
         

    # ---------- kernel covar func ----------
    def _sitewise_prod_covar_func(self, x1, x2, num_ch):
        x1_site = self.select_site(x1, site=0, num_ch=num_ch)
        x2_site = self.select_site(x2, site=0, num_ch=num_ch)
        x1_ = LazyTensor(x1_site[:, None, :])
        x2_ = LazyTensor(x2_site[None, :, :])
        K = (x1_ * x2_).sum(-1)
        for i in range(1, self.seq_length):
            x1_site = self.select_site(x1, site=i, num_ch=num_ch)
            x2_site = self.select_site(x2, site=i, num_ch=num_ch)
            x1_ = LazyTensor(x1_site[:, None, :])
            x2_ = LazyTensor(x2_site[None, :, :])
            K *= (x1_ * x2_).sum(-1)
        return K

    def get_sitewise_prod_covar_func(self, num_ch):
        def sitewise_prod_covar_func(x1, x2):
            return self._sitewise_prod_covar_func(x1, x2, num_ch)
        return sitewise_prod_covar_func

    # ---------- main computational routines ----------

    def compute_zeta_pc(self, pis: torch.Tensor, eta: float):
        return eta * torch.einsum("pij,pj->pi", self.site_kernels, pis)

    def compute_zeta_p(self, pis: torch.Tensor, eta: float):
        return (eta * eta) * torch.einsum("pi,pij,pj->p", pis, self.site_kernels, pis)
 
    def get_site_specific_values_for_R(self, pis: torch.Tensor, eta: float):
        dtype = pis.dtype
        device = self.device
        zeta_p = self.compute_zeta_p(pis, eta)
        zeta_pc = self.compute_zeta_pc(pis, eta)

        from_p = torch.ones(
            self.seq_length, self.n_alleles + 1, self.n_alleles + 1, device=device, dtype=dtype
        )
        from_p[:, -1, :] *= -1
        from_p[:, :, -1] *= -1
        from_p = zeta_p.view(self.seq_length, 1, 1) * from_p

        from_sp = zeta_pc[:, :, None] * torch.ones(
            self.seq_length, self.n_alleles, self.n_alleles + 1, device=device, dtype=dtype
        )
        from_sp[..., :-1] *= -1
        from_sp = torch.cat(
            [from_sp, torch.zeros(self.seq_length, 1, self.n_alleles + 1, device=device, dtype=dtype)],
            dim=-2,
        )

        from_tp = from_sp.transpose(-1, -2)

        from_a = torch.zeros(
            self.seq_length, self.n_alleles + 1, self.n_alleles + 1, device=device, dtype=dtype
        )
        from_a[:, : self.n_alleles, : self.n_alleles] = self.site_kernels

        return from_p + from_sp + from_tp + from_a
    
   

    def get_site_specific_values_for_z(self, pis: torch.Tensor, eta: float):
        dtype = pis.dtype
        device = self.device
        zetas = self.compute_zeta_pc(pis, eta)
        A_mat = zetas[:, :, None] * torch.ones(
            self.seq_length, self.n_alleles, self.n_alleles + 1, device=device, dtype=dtype
        )
        A_mat[..., :-1] *= -1
        kernel_to_add = torch.cat(
            [
                self.site_kernels,
                torch.zeros(self.seq_length, self.n_alleles, 1, device=device, dtype=dtype),
            ],
            dim=-1,
        )
        return A_mat + kernel_to_add
    

    def compute_z(self, pis: torch.Tensor, eta: float, subseqs: torch.Tensor):
        M_blocks = self.get_site_specific_values_for_z(pis, eta)
        subseqs = subseqs.contiguous().view(self.seq_length, self.extended_size, -1)
        transformed = torch.einsum("pij,pjk->pik", M_blocks, subseqs)
        transformed = transformed.view(self.seq_length * self.n_alleles, -1).T.contiguous()
        return KernelLinearOperator(
            self._train_x, transformed, covar_func=self.get_sitewise_prod_covar_func(self.n_alleles)
        )

    def compute_R(self, pis: torch.Tensor, eta: float, encoded_subseqs: torch.Tensor):
        x = encoded_subseqs.T.contiguous()
        x_blocks = x.reshape(x.size(0), self.seq_length, self.extended_size)
        R_blocks = self.get_site_specific_values_for_R(pis, eta)
        prod = torch.einsum("slj,ljk->slk", x_blocks, R_blocks)
        prod = prod.reshape(x.size(0), -1).contiguous()
        return KernelLinearOperator(
            prod, x, covar_func=self.get_sitewise_prod_covar_func(self.n_alleles + 1)
        )

    def _gauge_fixed_posterior(
        self, pis: torch.Tensor, eta: float, encoded_subseqs: torch.Tensor, calc_covar: bool = False
    ):
        dtype = self.comp_dtype
        pis = pis.to(device=self.device, dtype=dtype).contiguous()
        z = self.compute_z(pis, eta, encoded_subseqs.to(dtype))
        mean = self.Qy.to(dtype) @ z
        if not calc_covar:
            return mean
        R = self.compute_R(pis.to(dtype), eta, encoded_subseqs.to(dtype))
        zS = z.T @ self.S.to(dtype)
        cov = (R - (zS @ zS.T)).to_dense()
        return mean, cov

    def gauge_fixed_posterior(
        self, pis: torch.Tensor, eta: float, subseqs: Sequence[str], calc_covar: bool = False, validate: bool = True
    ):
        raw_subseqs = tuple(subseqs)
        if validate:
            self.validate_pis(pis)
            self.validate_sequences(raw_subseqs, self.extended_alphabet)
        encoded_subseqs = self._encode_subseqs(raw_subseqs)
        return self._gauge_fixed_posterior(pis, eta, encoded_subseqs, calc_covar)

    def gauge_fixed_posterior_linear(self, pis: torch.Tensor, eta: float, calc_covar: bool = False, validate: bool = True):
        if validate:
            self.validate_pis(pis)
        return self._gauge_fixed_posterior(pis, eta, self.encoded_linear_subseqs, calc_covar)

    def gauge_fixed_posterior_constant(self, pis: torch.Tensor, eta: float, calc_covar: bool = False, validate: bool = True):
        if validate:
            self.validate_pis(pis)
        return self._gauge_fixed_posterior(pis, eta, self.encoded_constant_subseq, calc_covar)



