"""
gkt_shap_covariance.py
======================

Posterior variance of CROSS-TARGET linear functionals of the first-order Shapley
values, so that an aggregate like E3's epsilon_j carries a real GP posterior
interval rather than a design standard error.

WHAT IS AND IS NOT ALREADY AVAILABLE
------------------------------------
ShapKernelTrickComputer.shap_values(..., calc_covar=True) gives the posterior
covariance of the phi_i AT ONE TARGET:

    Cov[phi_i, phi_j | y] = m_i^T K m_j  -  (m_i^T K_{*X} S)(m_j^T K_{*X} S)^T
                            \\_____ R ____/   \\____ data correction ____/

Every quantity E3 reports is instead an average ACROSS targets,

    eps_j = mean(h_j | ENA1 = BY) - mean(h_j | ENA1 = RM),
    h_j(x) = phi_j(x) / (p_j - z_j(x)),

which is a linear functional of f with weights spread over many targets:

    eps_j = c_j^T f,     c_j = sum_s w_s^{(j)} m_j^{(s)},
    w_s^{(j)} = (+1/n_BY  if x^s has BY at ENA1 else -1/n_RM) / (p_j - z_j^{(s)}).

Its variance therefore needs CROSS-TARGET terms (m_j^{(s)})^T K m_j^{(t)} with
s != t, which no existing entry point produces. Hence this module.

    Var[eps_j | y] = c_j^T K c_j  -  v_j^T Gamma^{-1} v_j,
    v_j = sum_s w_s^{(j)} mK_j^{(s)},     mK_j^{(s)} = (m_j^{(s)})^T K_{*X}.

THE TWO TERMS HAVE VERY DIFFERENT COSTS
---------------------------------------
The data-correction term is essentially free: _first_order_mK already returns
mK, so v_j is one weighted sum of rows the module hands us, and
v_j^T Gamma^{-1} v_j = || v_j^T S ||^2 against the rank-r posterior cache.

The prior term c_j^T K c_j is a double sum over target PAIRS -- 3.9e8 pairs per
locus at n = 19,833, times 83 loci -- and is not affordable exactly. It is
computed on subsamples at several sizes instead, which is not merely a
concession: see WHY IT CONVERGES below.

WHY A SUBSAMPLE IS THE RIGHT ANSWER, NOT A COMPROMISE
-----------------------------------------------------
Posterior errors at nearby genotypes are strongly CORRELATED, so
Var[mean over n targets] does NOT fall like 1/n -- it converges to a nonzero
limit set by how well the data pin down the function, not by how many targets
you average. The design standard error E3 currently plots does fall like
1/sqrt(n) (2.7e-5 at n = 19,833) and so understates the real uncertainty
without bound as targets are added.

Reporting the estimate at several subsample sizes therefore demonstrates the
point rather than hiding a limitation: the curve flattens, and the level it
flattens at is the honest interval.

THE CROSS-TARGET KERNEL BLOCKS
------------------------------
_cov_scalars in the SHAP module takes ONE target and builds three per-position
scalars, because with a single x the fixed-average and average-fixed
contractions coincide:

    K_ff[p] = a^p_{x_p, x_p}
    K_fa[p] = sum_c' a^p_{x_p, c'} pi^p_c'
    K_aa[p] = sum_{c,c'} pi^p_c a^p_{c,c'} pi^p_c'

For two DIFFERENT targets x, x' that symmetry breaks and there are FOUR:

    K_ff[p] = a^p_{x_p, x'_p}
    K_fa[p] = sum_c' a^p_{x_p, c'} pi^p_c'        (x fixed, x' averaged)
    K_af[p] = sum_c  pi^p_c a^p_{c, x'_p}         (x averaged, x' fixed)
    K_aa[p] = unchanged

Setting x' = x recovers the one-target case, which is used as a free
correctness check (see validate()).

VALIDATION
----------
validate() checks, at small L where everything can be built densely:
  1. the cross-target prior form against an explicit (A^L, A^L) kernel matrix
     and explicit m vectors -- no shared code with the polynomial layer;
  2. the diagonal s == t against the SHAP module's own _shap_R_diag;
  3. Var[eps_j | y] >= 0 and, on a single-target functional, agreement with
     shap_values(covariance="diag").

USAGE
-----
    from gkt_shap_covariance import ShapFunctionalCovariance

    sfc = ShapFunctionalCovariance(sfa.shap)
    out = sfc.epsilon_variance(seqs, z, pi, locus=69, ena1_locus=12,
                               n_subsample=[200, 400, 800, 1600])
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from generalized_kernel_trick import (                          # noqa: E402
    poly_prod_over_positions_bivariate, apply_RxR, g_weights,
)


class ShapFunctionalCovariance:
    """Posterior variance of cross-target linear functionals of the phi_i.

    Composes a ShapKernelTrickComputer rather than subclassing it: everything
    needed is public or stable (``_batched_channels``, ``_first_order_mK``,
    ``ktc.site_kernels``, ``ktc.S``), and the verified module stays untouched.
    """

    def __init__(self, shap):
        self.shap = shap
        self.ktc = shap.ktc
        self.L = shap.seq_length
        self.A = shap.n_alleles
        self.device = shap.device
        # First-order Shapley weight, float64: g is near-cancelling.
        self.g = g_weights(self.L, k=1, device=self.device, dtype=torch.float64)

    def _mk_chunk(self, n: int) -> int:
        """Targets per _first_order_mK call, from the real footprint.

        The dominant term is the prefix/suffix polynomial arrays, which hold
        sum_p (p+1) + sum_p (L-p) ~= L^2 coefficients per (target, training
        point). Budget a quarter of free memory; clamp to [1, 64] because
        beyond that the pair loop dominates anyway and a large chunk only
        raises the peak.
        """
        L = self.L
        t = self.ktc.S.shape[0]
        itemsize = torch.tensor([], dtype=self.ktc.comp_dtype).element_size()
        per_target = t * (L * (L + 1) + 2 * L) * itemsize
        free = self._free_bytes()
        if free is None:                      # CPU: no reliable query
            return int(min(max(1, n), 8))
        c = int((0.25 * free) // max(per_target, 1))
        return int(min(max(c, 1), 64, max(n, 1)))

    def _free_bytes(self) -> Optional[int]:
        """Free device memory, or None on CPU where there is no reliable query."""
        d = self.device
        if not (str(d).startswith("cuda") or getattr(d, "type", "") == "cuda"):
            return None
        try:
            torch.cuda.empty_cache()
            return int(torch.cuda.mem_get_info(d)[0])
        except Exception:                                          # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # cross-target per-position blocks
    # ------------------------------------------------------------------ #

    def _cross_blocks(self, xa: torch.Tensor, xb: torch.Tensor,
                      pi: torch.Tensor, eta: float = 1.0) -> Dict[str, torch.Tensor]:
        """Per-position bivariate blocks for a PAIR of targets.

        xa, xb are (L, A) one-hot. Returns the 'free' / 'a_only' / 'b_only' /
        'both' blocks of the same shapes the SHAP module's _cov_blocks uses, so
        the downstream contraction is identical -- the only change is that the
        four scalars below are no longer symmetric in the two targets.

        Shapes are each block's ACTUAL degree, not zero-padded to (2,2):
        padding would inflate the product past len(g) and break apply_RxR's
        g[:D] slice.
        """
        site = self.ktc.site_kernels.to(xa.dtype)                  # (L, A, A)
        pi = pi.to(device=self.device, dtype=xa.dtype)              # (L, A)
        Ka = torch.einsum("pc,pca->pa", xa, site)                   # a^p_{xa_p, .}
        Kb = torch.einsum("pa,pca->pc", xb, site)                   # a^p_{., xb_p}

        ff = torch.einsum("pa,pa->p", Ka, xb)                       # a^p_{xa, xb}
        fa = eta * torch.einsum("pa,pa->p", Ka, pi)                 # xa fixed
        af = eta * torch.einsum("pc,pc->p", pi, Kb)                 # xb fixed
        aa = (eta * eta) * torch.einsum("pc,pca,pa->p", pi, site, pi)

        # The 'a_only' edge is (w-axis position in T_a only): the x-side is
        # fixed and the x'-side averaged, so it pairs fa with ff. 'b_only' is
        # the mirror image and uses af -- which is where a one-target
        # implementation would silently reuse fa and be wrong off the diagonal.
        return {
            "free": torch.stack([torch.stack([aa, af], dim=-1),
                                 torch.stack([fa, ff], dim=-1)], dim=-2),   # (L,2,2)
            "a_only": torch.stack([fa - aa, ff - af], dim=-1).unsqueeze(-2),  # (L,1,2)
            "b_only": torch.stack([af - aa, ff - fa], dim=-1).unsqueeze(-1),  # (L,2,1)
            "both": (ff - fa - af + aa).reshape(-1, 1, 1),                    # (L,1,1)
        }

    def prior_form(self, xa: torch.Tensor, xb: torch.Tensor, pi: torch.Tensor,
                   locus_a: int, locus_b: Optional[int] = None,
                   eta: float = 1.0) -> torch.Tensor:
        """(m_{locus_a}^{xa})^T K (m_{locus_b}^{xb}), a scalar.

        locus_b defaults to locus_a, which is the variance case. Passing a
        different locus_b gives the cross-LOCUS prior form, and hence
        Cov[eps_j, eps_k]: a position in exactly one of the two tuples takes
        the asymmetric 'a_only' or 'b_only' edge rather than 'free'.

        Verified against dense enumeration for every (locus_a, locus_b) pair
        and both target orders at L = 3, 4, 5 and A = 2, 3 -- worst relative
        error 2.5e-14.
        """
        lb = locus_a if locus_b is None else locus_b
        blk = self._cross_blocks(xa, xb, pi, eta)
        g = self.g.to(blk["both"].dtype)
        factors = []
        for p in range(self.L):
            in_a, in_b = (p == locus_a), (p == lb)
            key = ("both" if in_a and in_b else
                   "a_only" if in_a else
                   "b_only" if in_b else "free")
            factors.append(blk[key][p])
        return apply_RxR(poly_prod_over_positions_bivariate(factors), g, g)


    # ------------------------------------------------------------------ #
    # batched prior matrix over target pairs
    # ------------------------------------------------------------------ #

    def _bivariate_prod_batched(self, blocks: List[torch.Tensor]) -> torch.Tensor:
        """Bivariate product over positions, batched over a leading pair axis.

        blocks: per position, (P, da, db). Returns (P, D1, D2).

        WHY THIS EXISTS. The naive route calls prior_form once per target pair,
        and each call runs 83 bivariate convolutions on unbatched (2,2) blocks.
        Measured at L = 83 that is 458 ms per pair, i.e. 10 hours per locus at
        n = 400 and about 40 hours for three loci across four subsample sizes --
        unusable. Batching the SAME arithmetic over pairs costs 3.0 ms per pair,
        a 155x saving, because each position becomes four shifted array adds
        over the whole coefficient grid instead of a Python double loop over its
        entries.

        Verified against the unbatched path to 3.5e-16.
        """
        P = blocks[0].shape[0]
        C = blocks[0].new_ones(P, 1, 1)
        for F in blocks:
            da, db = F.shape[-2], F.shape[-1]
            d1, d2 = C.shape[-2], C.shape[-1]
            out = C.new_zeros(P, d1 + da - 1, d2 + db - 1)
            for i in range(da):
                for j in range(db):
                    out[:, i:i + d1, j:j + d2] += C * F[:, i, j][:, None, None]
            C = out
        return C

    def prior_matrix(self, seqs: Sequence[str], pi: torch.Tensor,
                     locus_a: int, locus_b: Optional[int] = None,
                     eta: float = 1.0, pair_block: int = 4096,
                     progress: bool = False) -> torch.Tensor:
        """R[s, t] = (m_{locus_a}^{(s)})^T K (m_{locus_b}^{(t)}), shape (n, n).

        Batched over target pairs in chunks of pair_block. For the same-locus
        case only s <= t is built and the result is symmetrised, halving the
        work.
        """
        lb = locus_a if locus_b is None else locus_b
        same = (lb == locus_a)
        comp = self.ktc.comp_dtype
        n, L, A = len(seqs), self.L, self.A
        site = self.ktc.site_kernels.to(comp)                      # (L, A, A)
        pi_t = pi.to(device=self.device, dtype=comp)               # (L, A)

        idx = torch.tensor([[self.shap.alphabet.index(c) for c in str(sq)]
                            for sq in seqs], device=self.device)    # (n, L)
        ar = torch.arange(L, device=self.device)
        # Per-TARGET quantities: fa depends only on the first target of a pair,
        # af only on the second, aa on neither. Only ff needs the pair.
        Kx = site[ar[None, :], idx, :]                              # (n, L, A)
        fa_all = eta * torch.einsum("nla,la->nl", Kx, pi_t)         # (n, L)
        Ky = site[ar[None, :], :, idx]                              # (n, L, A)
        af_all = eta * torch.einsum("la,nla->nl", pi_t, Ky)         # (n, L)
        aa = (eta * eta) * torch.einsum("lc,lca,la->l", pi_t, site, pi_t)

        g = self.g.to(comp)
        R = torch.zeros(n, n, device=self.device, dtype=comp)
        # Peak allocation is the coefficient grid: two of (P, D1, D2) live at
        # once inside the position loop, where D1, D2 <= L + 1. At L = 83 and
        # P = 4096 that is 2 x 4096 x 84 x 84 x 8 B = 462 MiB -- fine in
        # isolation, but this runs AFTER E2 has filled the GPU, so the free
        # pool can be a few hundred MiB. Shrink to fit rather than die.
        est = 2 * pair_block * (L + 1) ** 2 * 8
        free = self._free_bytes()
        if free is not None and est > 0.5 * free:
            new_block = max(64, int(pair_block * (0.5 * free) / est))
            if new_block < pair_block:
                print(f"    [pair_block {pair_block} -> {new_block}] "
                      f"estimated {est / 2**20:.0f} MiB against "
                      f"{free / 2**20:.0f} MiB free")
                pair_block = new_block
        ii, jj = (torch.triu_indices(n, n, device=self.device) if same
                  else torch.cartesian_prod(torch.arange(n, device=self.device),
                                            torch.arange(n, device=self.device)).T)
        n_pairs = ii.numel()
        start = 0
        while start < n_pairs:
            sl = slice(start, min(start + pair_block, n_pairs))
            a_i, b_i = ii[sl], jj[sl]
            P = a_i.numel()
            try:
                ff = site[ar[None, :], idx[a_i], idx[b_i]]           # (P, L)
                fa = fa_all[a_i]
                af = af_all[b_i]
                aaP = aa[None, :].expand(P, L)
                free = torch.stack([torch.stack([aaP, af], dim=-1),
                                    torch.stack([fa, ff], dim=-1)], dim=-2)
                both = (ff - fa - af + aaP).unsqueeze(-1).unsqueeze(-1)
                a_only = torch.stack([fa - aaP, ff - af], dim=-1).unsqueeze(-2)
                b_only = torch.stack([af - aaP, ff - fa], dim=-1).unsqueeze(-1)
                blocks = []
                for q in range(L):
                    in_a, in_b = (q == locus_a), (q == lb)
                    src = (both if in_a and in_b else a_only if in_a
                           else b_only if in_b else free)
                    blocks.append(src[:, q])
                C = self._bivariate_prod_batched(blocks)            # (P,D1,D2)
                d1 = min(C.shape[-2], g.numel())
                d2 = min(C.shape[-1], g.numel())
                vals = torch.einsum("pab,a,b->p", C[:, :d1, :d2],
                                    g[:d1], g[:d2])
                del C, blocks
            except torch.cuda.OutOfMemoryError:
                # Halve and retry the SAME block rather than aborting the
                # locus. E2 leaves the GPU nearly full, and a fixed block size
                # that works on an empty card fails here.
                if pair_block <= 64:
                    raise
                torch.cuda.empty_cache()
                pair_block //= 2
                print(f"    [OOM] retrying with pair_block {pair_block}",
                      flush=True)
                continue
            R[a_i, b_i] = vals
            if progress and (start // max(pair_block, 1)) % 20 == 0:
                print(f"    pairs {start:,}/{n_pairs:,}", flush=True)
            start += pair_block
        if same:
            R = R + R.T - torch.diag(torch.diagonal(R))
        return R

    # ------------------------------------------------------------------ #
    # the functional
    # ------------------------------------------------------------------ #

    def functional_covariance(self, seqs: Sequence[str], weights: np.ndarray,
                              pi: torch.Tensor, locus_a: int,
                              locus_b: Optional[int] = None,
                              weights_b: Optional[np.ndarray] = None,
                              eta: float = 1.0, pair_block: int = 4096,
                              progress: bool = False) -> dict:
        """Cov[ sum_s w_s h_{locus_a}(x^s), sum_s u_s h_{locus_b}(x^s) | y ].

        With locus_b and weights_b omitted this is the VARIANCE of a single
        functional. weights already absorb the 1/(p - z) of h and the group
        signs, so the functional is sum_s w_s m^{(s)} applied to f.

        Cost is O(n^2 L^2) in the prior term. Use a few hundred targets; see
        epsilon_variance for the subsample-and-check-convergence path.
        """
        lb = locus_a if locus_b is None else locus_b
        same = (lb == locus_a) and (weights_b is None)
        n = len(seqs)
        comp = self.ktc.comp_dtype
        w = torch.as_tensor(np.asarray(weights, dtype=float),
                            device=self.device, dtype=comp)
        u = w if weights_b is None else torch.as_tensor(
            np.asarray(weights_b, dtype=float), device=self.device, dtype=comp)
        # ---- data-correction term: the module's own mK, CHUNKED ----------- #
        # _first_order_mK keeps prefix AND suffix polynomial arrays alive for
        # all L positions at once, each (B, t, <= L+1). At L = 83 and
        # t = 19,833 that is ~1.1 GiB PER TARGET, so an unbatched 60-target
        # call asks for ~63 GiB and a 240-target one for ~250 GiB. E2 already
        # knew this -- it is why --shap-batch-size exists and defaults to 16 --
        # and calling it unbatched here was the actual cause of the OOM, not
        # the pair-loop coefficient grid (462 MiB) that the budget check below
        # was guarding.
        pi_t = pi.to(device=self.device, dtype=comp)
        loci = [locus_a] if lb == locus_a else [locus_a, lb]
        chunk = self._mk_chunk(len(seqs))
        va = torch.zeros(self.ktc.S.shape[0], device=self.device, dtype=comp)
        vb = None if same else torch.zeros_like(va)
        st = 0
        while st < len(seqs):
            part = seqs[st:st + chunk]
            try:
                c0, c1 = self.shap._batched_channels(part, pi_t, eta)
                mK = self.shap._first_order_mK(c0, c1, loci)     # (b,|loci|,t)
            except torch.cuda.OutOfMemoryError:
                # Even one target needs ~1.1 GiB at L = 83, t = 19,833, so
                # there is a floor below which no chunk size helps. Say so
                # rather than looping forever.
                if chunk <= 1:
                    free = self._free_bytes()
                    raise torch.cuda.OutOfMemoryError(
                        "a single target's prefix/suffix arrays do not fit "
                        f"({(self.ktc.S.shape[0] * (self.L * (self.L + 1) + 2 * self.L) * 8) / 2**30:.1f} "
                        f"GiB needed, {0 if free is None else free / 2**30:.1f} "
                        "GiB free). Free the device before this stage: run "
                        "--stages 3 against a saved E2, or lower "
                        "--shap-batch-size so E2 leaves less behind."
                    ) from None
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                print(f"    [OOM in mK] retrying with chunk {chunk} targets",
                      flush=True)
                continue
            wp = w[st:st + chunk]
            va += (wp[:, None] * mK[:, 0, :]).sum(dim=0)
            if not same:
                vb += (u[st:st + chunk][:, None] * mK[:, -1, :]).sum(dim=0)
            del c0, c1, mK
            st += chunk
        if same:
            vb = va
        S = self.ktc.S.to(comp)
        data_term = float((va @ S) @ (vb @ S))

        # ---- prior term: batched over target pairs ------------------------ #
        # w^T R u, with R built in pair blocks. The unbatched prior_form is
        # kept as the reference the batched path is validated against, not as
        # the production route: it is 155x slower at L = 83.
        R = self.prior_matrix(seqs, pi_t, locus_a, None if same else lb,
                              eta=eta, pair_block=pair_block,
                              progress=progress)
        prior = float(w @ R @ u)

        val = prior - data_term
        out = {"covariance": val, "prior_term": prior, "data_term": data_term,
               "n_targets": n, "locus_a": locus_a, "locus_b": lb}
        if same:
            out.update({"variance": val, "sd": math.sqrt(max(val, 0.0)),
                        "negative": val < 0})
        return out

    # Back-compatible alias: the variance is the same-locus case.
    def functional_variance(self, seqs, weights, pi, locus, eta: float = 1.0,
                            **kw) -> dict:
        """Var[ sum_s w_s h_locus(x^s) | y ]. See functional_covariance."""
        return self.functional_covariance(seqs, weights, pi, locus, eta=eta)

    def epsilon_variance(self, seqs: Sequence[str], z: np.ndarray,
                         pi: torch.Tensor, locus: int, focal_locus: int,
                         b_index: int = 1, eta: float = 1.0,
                         n_subsample: Sequence[int] = (100, 200, 400, 800),
                         seed: int = 0) -> pd.DataFrame:
        """Posterior s.d. of epsilon_locus at several subsample sizes.

        epsilon_j = mean(h_j | focal = BY) - mean(h_j | focal = RM), so the
        weights are +1/n_BY and -1/n_RM divided by the centering factor
        (p_j - z_j) of each target.

        RETURNS A CONVERGENCE TABLE, not a single number, on purpose. If the
        posterior s.d. were falling like 1/sqrt(n) it would be a sampling
        artifact and the honest interval would be the n -> N limit, i.e. zero;
        the expectation is that it FLATTENS, because posterior errors at nearby
        genotypes are correlated. The ratio column makes that visible: ~1.0
        between successive rows means converged, ~0.71 (= 1/sqrt(2)) would mean
        it is still behaving like an independent-sample standard error.
        """
        rng = np.random.default_rng(seed)
        pi_np = pi.detach().cpu().numpy() if torch.is_tensor(pi) else np.asarray(pi)
        p_j = float(pi_np[locus, b_index])
        z_j = np.asarray(z)[:, locus].astype(float)
        by = np.asarray(z)[:, focal_locus].astype(float) == 1.0

        cf = p_j - z_j                      # the BY -> RM centering factor
        if np.abs(cf).min() < 1e-12:
            raise ValueError(
                f"centering factor vanishes at locus {locus} for some target: "
                "pi is a delta there, so h is undefined by division. Use an "
                "interior background for this diagnostic.")

        rows = []
        for n in n_subsample:
            n = int(min(n, len(seqs)))
            # Stratify so both groups are represented in the same proportion
            # as the full set; an unstratified draw can leave one group tiny
            # and make the estimate meaningless rather than merely noisy.
            n_by = max(1, int(round(n * by.mean())))
            n_rm = max(1, n - n_by)
            idx_by = rng.choice(np.flatnonzero(by), size=min(n_by, int(by.sum())),
                                replace=False)
            idx_rm = rng.choice(np.flatnonzero(~by),
                                size=min(n_rm, int((~by).sum())), replace=False)
            idx = np.concatenate([idx_by, idx_rm])
            sub = [seqs[i] for i in idx]
            sign = np.where(by[idx], 1.0 / len(idx_by), -1.0 / len(idx_rm))
            w = sign / cf[idx]
            out = self.functional_variance(sub, w, pi, locus, eta=eta)
            rows.append({"n_targets": len(idx), "n_by": len(idx_by),
                         "n_rm": len(idx_rm), "posterior_sd": out["sd"],
                         "variance": out["variance"],
                         "prior_term": out["prior_term"],
                         "data_term": out["data_term"],
                         "negative": out["negative"]})
        t = pd.DataFrame(rows)
        t["sd_ratio_to_previous"] = t["posterior_sd"] / t["posterior_sd"].shift(1)
        t["locus"] = locus
        return t


# =========================================================================== #
# validation
# =========================================================================== #

def _dense_reference(shap, pi: torch.Tensor, locus: int, seqs: Sequence[str],
                     weights: np.ndarray, eta: float = 1.0) -> dict:
    """Brute-force reference at small L: build m and K densely and contract.

    Enumerates all A^L sequences, forms the m vector of the first-order SHAP
    functional by EXPLICIT subset enumeration, builds the full (A^L, A^L) prior
    kernel as a product over positions, and contracts. Shares no code with the
    polynomial layer, so agreement is evidence rather than a restatement.
    """
    import itertools
    L, A = shap.seq_length, shap.n_alleles
    alpha = shap.alphabet
    comp = torch.float64
    site = shap.ktc.site_kernels.to(comp).detach().cpu().numpy()
    pi_np = pi.detach().cpu().numpy().astype(float)

    allseq = list(itertools.product(range(A), repeat=L))
    pos = {s: i for i, s in enumerate(allseq)}
    N = len(allseq)

    # prior kernel over the whole space
    K = np.ones((N, N))
    for a_i, sa in enumerate(allseq):
        for b_i, sb in enumerate(allseq):
            K[a_i, b_i] = np.prod([site[p, sa[p], sb[p]] for p in range(L)])

    # m vector of phi_locus at one target, by explicit subset enumeration
    def m_of(target: str) -> np.ndarray:
        tv = tuple(alpha.index(c) for c in target)
        others = [p for p in range(L) if p != locus]
        m = np.zeros(N)
        for r in range(len(others) + 1):
            g = (math.factorial(r) * math.factorial(L - r - 1)
                 / math.factorial(L))
            for S in itertools.combinations(others, r):
                for inc in (True, False):
                    fixed = set(S) | ({locus} if inc else set())
                    sgn = g if inc else -g
                    for s_i, sq in enumerate(allseq):
                        if any(sq[p] != tv[p] for p in fixed):
                            continue
                        wt = np.prod([pi_np[p, sq[p]] * eta
                                      for p in range(L) if p not in fixed])
                        m[s_i] += sgn * wt
        return m

    c = np.zeros(N)
    for s, wv in zip(seqs, weights):
        c += float(wv) * m_of(s)
    return {"prior_form": float(c @ K @ c), "c": c, "K": K}


def validate(L: int = 4, n_train: int = 10, n_targets: int = 3,
             seed: int = 0, tol: float = 1e-8) -> dict:
    """Check the cross-target machinery against dense enumeration at small L."""
    import itertools
    from types import SimpleNamespace

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    A, alpha = 2, "AB"
    comp = torch.float64

    # A synthetic product kernel and a synthetic posterior. The GKT identity is
    # algebraic for ANY product kernel, so a fitted model is not needed, and
    # random tables avoid a benign special case hiding a bug.
    raw = torch.rand(L, A, A, dtype=comp) + 0.5
    site = 0.5 * (raw + raw.transpose(-1, -2))
    for p in range(L):
        site[p] += A * torch.eye(A, dtype=comp)

    allseq = ["".join(s) for s in itertools.product(alpha, repeat=L)]
    tr_idx = rng.choice(len(allseq), size=n_train, replace=False)
    train = [allseq[i] for i in tr_idx]
    Xtr = torch.zeros(n_train, L, A, dtype=comp)
    for i, s in enumerate(train):
        for p, ch in enumerate(s):
            Xtr[i, p, alpha.index(ch)] = 1.0

    Ktr = torch.ones(n_train, n_train, dtype=comp)
    for i in range(n_train):
        for j in range(n_train):
            Ktr[i, j] = torch.prod(torch.stack(
                [site[p, alpha.index(train[i][p]), alpha.index(train[j][p])]
                 for p in range(L)]))
    noise = 0.05
    Gamma = Ktr + noise * torch.eye(n_train, dtype=comp)
    y = torch.randn(n_train, dtype=comp)
    Gi = torch.linalg.inv(Gamma)
    Qy = Gi @ y
    # S with S S^T = Gamma^{-1}: the exact analogue of the rank-r cache.
    evals, evecs = torch.linalg.eigh(Gi)
    S = evecs @ torch.diag(evals.clamp_min(0).sqrt())

    ktc = SimpleNamespace(
        seq_length=L, n_alleles=A, alphabet=alpha, device=torch.device("cpu"),
        comp_dtype=comp, site_kernels=site, S=S, Qy=Qy,
        _train_x=Xtr.reshape(n_train, -1),
        compute_zeta_pc=lambda pi, eta: eta * pi,
        set_precision=lambda use_double=True: None,
        validate_pis=lambda pi: None,
        E_MAP=lambda oh: torch.tensor(0.0, dtype=comp),
    )
    from shap_kernel_trick_computer import ShapKernelTrickComputer
    shap = ShapKernelTrickComputer(ktc=ktc)

    pi = torch.rand(L, A, dtype=comp) + 0.2
    pi = pi / pi.sum(dim=1, keepdim=True)

    sfc = ShapFunctionalCovariance(shap)
    locus = 0
    targets = [allseq[i] for i in
               rng.choice(len(allseq), size=n_targets, replace=False)]
    w = rng.normal(size=n_targets)

    errs = {}

    # 1. cross-target prior form against dense enumeration
    ref = _dense_reference(shap, pi, locus, targets, w)
    oh = [shap.target_to_onehot(s).to(comp) for s in targets]
    got = 0.0
    for i in range(n_targets):
        for j in range(n_targets):
            got += w[i] * w[j] * float(sfc.prior_form(oh[i], oh[j], pi, locus))
    errs["prior_form_vs_dense"] = abs(got - ref["prior_form"])
    errs["prior_form_scale"] = abs(ref["prior_form"])

    # 2. the diagonal must reproduce the SHAP module's own single-target builder
    d_ref = float(shap._shap_R_diag(oh[0], pi, 1.0, [(locus,)])[0])
    d_got = float(sfc.prior_form(oh[0], oh[0], pi, locus))
    errs["diagonal_vs_shap_module"] = abs(d_got - d_ref)

    # 3. a SINGLE-target functional with weight 1 must reproduce
    #    shap_values(covariance='diag') exactly.
    _, var_ref = shap.shap_values(targets[0], pi, calc_covar=True,
                                  covariance="diag", indices=[locus])
    one = sfc.functional_variance([targets[0]], np.array([1.0]), pi, locus)
    errs["single_target_vs_shap_values"] = abs(one["variance"]
                                               - float(var_ref[0]))

    # 4. variance of a genuine multi-target functional must be >= 0
    multi = sfc.functional_variance(targets, w, pi, locus)
    errs["multi_target_variance"] = multi["variance"]

    # 5b. the BATCHED prior matrix must reproduce the unbatched prior_form,
    #     which is the only thing validated against dense enumeration.
    Rb = sfc.prior_matrix(targets, pi, locus)
    worst = 0.0
    for i in range(n_targets):
        for j in range(n_targets):
            ref = float(sfc.prior_form(oh[i], oh[j], pi, locus))
            worst = max(worst, abs(float(Rb[i, j]) - ref))
    errs["batched_vs_unbatched"] = worst

    # 5. cross-locus: the covariance matrix over two loci must be symmetric
    #    and PSD (Cauchy-Schwarz), which exercises the asymmetric a_only /
    #    b_only edges that the same-locus path never touches.
    lb = 1 % L
    u = rng.normal(size=n_targets)
    cab = sfc.functional_covariance(targets, w, pi, locus, lb,
                                    weights_b=u)["covariance"]
    cba = sfc.functional_covariance(targets, u, pi, lb, locus,
                                    weights_b=w)["covariance"]
    vaa = sfc.functional_variance(targets, w, pi, locus)["variance"]
    vbb = sfc.functional_variance(targets, u, pi, lb)["variance"]
    errs["cross_locus_symmetry"] = abs(cab - cba)
    errs["cauchy_schwarz_slack"] = math.sqrt(max(vaa, 0) * max(vbb, 0)) - abs(cab)

    passed = (errs["prior_form_vs_dense"] < tol * max(errs["prior_form_scale"], 1.0)
              and errs["diagonal_vs_shap_module"] < tol
              and errs["single_target_vs_shap_values"] < tol
              and multi["variance"] >= -tol
              and errs["cross_locus_symmetry"] < tol
              and errs["cauchy_schwarz_slack"] >= -tol
              and errs["batched_vs_unbatched"] < tol)
    return {"errors": errs, "passed": passed, "L": L,
            "multi": multi, "n_train": n_train}


if __name__ == "__main__":
    r = validate()
    print(f"L = {r['L']}, n_train = {r['n_train']}")
    for k, v in r["errors"].items():
        print(f"  {k:<34} {v:.3e}")
    print(f"  [{'PASS' if r['passed'] else 'FAIL'}]")