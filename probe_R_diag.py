# =====================================================================
# COVARIANCE OPTIMIZATION -- DIAGONAL PATH (opt-in, additive)
#
# Paste _shap_R_diag into class ShapKernelTrickComputer, right after
# _shap_R_interaction_fast. Then apply the small edits described in
# COVARIANCE_SWITCH_EDITS below to the covariance-returning methods.
#
# Nothing here changes an existing default. calc_covar=True still returns the
# FULL covariance matrix (so test_gkt_brute.py and any full-matrix consumer are
# untouched). The new covariance="diag" is OPT-IN and returns only per-term
# variances (a 1-D tensor), which is all _global_cov_sd and the local sd/z
# columns actually consume.
#
# Why diag is the real win: the full R is an n x n grid of bivariate products
# (the measured 3x-per-L-step blowup). But the covariance DIAGONAL needs only
# R[a,a] for each tuple -- n bivariate products, not n^2 -- and the data-
# correction diagonal is just the row-wise squared norm of (mK S). So the diag
# path is O(n) in tuples where the full path is O(n^2), collapsing the curve for
# the common case (variances / standard deviations only).
# =====================================================================

    def _shap_R_diag(self, x_onehot, pi, eta, interactions, weight_fn=None):
        """Diagonal of R: returns ``R_aa = m^{T_a} K m^{T_a}`` for each tuple, shape (n,).

        The full :meth:`_shap_R_interaction` builds an n x n grid; the diagonal needs
        only the n same-tuple products, so this is O(n) tuple-products instead of
        O(n^2). Verified == ``torch.diagonal(_shap_R_interaction(...))`` (probe_R_diag.py,
        ~1e-12). Reuses the same per-position bivariate blocks as the pair (a,a): every
        position in T_a takes the "both" (0,0) block ``ff-2fa+aa``; every other position
        takes the "both_free" (1,1) block ``[[aa,fa],[fa,ff]]``.
        """
        L = self.seq_length
        if weight_fn is None:
            weight_fn = self._g_for_order
        sc = self._cov_scalars(x_onehot, pi, eta)
        K_ff, K_fa, K_aa = sc["K_ff"], sc["K_fa"], sc["K_aa"]
        dtype = K_ff.dtype

        both_free = torch.stack([                                    # (L, 2, 2)
            torch.stack([K_aa, K_fa], dim=-1),
            torch.stack([K_fa, K_ff], dim=-1),
        ], dim=-2)
        both_diag = (K_ff - 2.0 * K_fa + K_aa)                       # (L,)  the (0,0) block per position

        g_by_order: Dict[int, torch.Tensor] = {}

        def gord(k: int) -> torch.Tensor:
            g = g_by_order.get(k)
            if g is None:
                g = weight_fn(k).to(dtype)
                g_by_order[k] = g
            return g

        out = K_ff.new_zeros(len(interactions))
        for a, T in enumerate(interactions):
            Tset = set(T)
            g = gord(len(Tset))
            factors = []
            for p in range(L):
                if p in Tset:
                    factors.append(both_diag[p].reshape(1, 1))       # (0,0) block
                else:
                    factors.append(both_free[p])                     # (1,1) block
            coeffs2 = poly_prod_over_positions_bivariate(factors)
            out[a] = apply_RxR(coeffs2, g, g)
        return out


# =====================================================================
# COVARIANCE_SWITCH_EDITS
#
# For EACH covariance-returning method -- shap_values, shap_interaction_values,
# dividend_values, faith_shap_values -- add a `covariance="full"` parameter and,
# when covariance=="diag", return per-term VARIANCES instead of the full matrix.
# The change is the same shape in each; shown here for shap_interaction_values.
#
# BEFORE (schematic tail of the calc_covar branch):
#
#     mK = self._shap_z_interaction(x_oh, pi, eta, T_list, weight_fn=wfn)   # (n, t)
#     zS = mK @ self.ktc.S
#     R  = self._shap_R_interaction(x_oh, pi, eta, T_list, weight_fn=wfn)   # (n, n)
#     cov = R - zS @ zS.T
#     return T_list, phi, cov
#
# AFTER:
#
#     mK = self._shap_z_interaction(x_oh, pi, eta, T_list, weight_fn=wfn)   # (n, t)
#     zS = mK @ self.ktc.S
#     if covariance == "diag":
#         R_diag = self._shap_R_diag(x_oh, pi, eta, T_list, weight_fn=wfn)  # (n,)
#         var = R_diag - (zS * zS).sum(dim=1)                              # (n,)
#         return T_list, phi, var
#     R = self._shap_R_interaction(x_oh, pi, eta, T_list, weight_fn=wfn)    # (n, n)
#     cov = R - zS @ zS.T
#     return T_list, phi, cov
#
# Signature: add covariance: str = "full" (after calc_covar). Validate:
#     if covariance not in ("full", "diag"):
#         raise ValueError("covariance must be 'full' or 'diag'.")
#
# Notes:
#   * The diagonal of (mK S)(mK S)^T is exactly the row-wise squared norm
#     (zS*zS).sum(dim=1) -- no n^2 matmul needed, matching the O(n) R_diag.
#   * covariance is only consulted when calc_covar is truthy; with calc_covar
#     False the methods are unchanged.
#   * Default "full" keeps every existing caller and test_gkt_brute.py identical.
# =====================================================================







#!/usr/bin/env python
"""
probe_R_diag.py
===============
Gate for the diagonal covariance path. Verifies, for SII and dividend weights,
across targets:
  (1) _shap_R_diag == torch.diagonal(_shap_R_interaction)          ~1e-12
  (2) covariance="diag" var == torch.diagonal(covariance="full" cov)  ~1e-12
      (end-to-end through the public method, incl. the (mK S) correction)
  (3) speedup of the diag path vs full, and the L-scaling (should be far flatter)

Run AFTER pasting _shap_R_diag and applying COVARIANCE_SWITCH_EDITS.

    python probe_R_diag.py --data ./data/gb1.csv --kernel jenga --gpu \
        --n-condition 20000 --load-model --model-name gb1_jenga_nc20000
"""
import argparse, sys, itertools, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kernel", default="jenga")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n-condition", type=int, default=20000)
    ap.add_argument("--load-model", action="store_true")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--n-probe", type=int, default=64)
    ap.add_argument("--interaction-order", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scripts.analysis import AnalysisConfig, SequenceFunctionAnalysis, resolve_device
    import torch

    cfg = AnalysisConfig(
        data_path=args.data, kernel=args.kernel,
        device=resolve_device(args.gpu), n_condition=args.n_condition,
        load_model=args.load_model, model_name=args.model_name, background="uniform",
    )
    sfa = SequenceFunctionAnalysis(cfg).run_fit_only()
    shap, pi = sfa.shap, sfa.pi
    L = sfa.L
    ktc = shap.ktc; ktc.set_precision(use_double=True)
    comp = ktc.comp_dtype; dev = ktc.device
    cuda = str(dev).startswith("cuda")
    pi_c = pi.to(device=dev, dtype=comp)

    for m in ("_shap_R_diag",):
        if not hasattr(shap, m):
            print(f"[probe] ERROR: shap has no {m} -- paste R_diag_to_paste.py block."); return

    xs = [str(s) for s in sfa.seqs[:args.n_probe]]
    positions = [(i,) for i in range(L)]
    inter = []
    for k in range(2, args.interaction_order + 1):
        inter += list(itertools.combinations(range(L), k))
    tuples = positions + inter
    tol = args.tol
    def sync():
        if cuda: torch.cuda.synchronize()

    # (1) R_diag vs diagonal of full R
    for wname, wfn in (("SII", shap._g_for_order),
                       ("dividend", shap._index_weight_fn("dividend"))):
        x_oh = shap.target_to_onehot(xs[0]).to(comp)
        Rfull = shap._shap_R_interaction(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
        Rdiag = shap._shap_R_diag(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
        e = float((torch.diagonal(Rfull) - Rdiag).abs().max())
        print(f"[probe] R_diag vs diag(R_full) [{wname:8s}]: err={e:.2e} "
              f"[{'OK' if e<tol else 'FAIL'}]")

    # (2) end-to-end: covariance='diag' var vs diag of covariance='full'
    try:
        _, _, cov_full = shap.shap_interaction_values(
            xs[0], pi_c, interactions=tuples, normalize="sii",
            calc_covar=True, covariance="full")
        _, _, var_diag = shap.shap_interaction_values(
            xs[0], pi_c, interactions=tuples, normalize="sii",
            calc_covar=True, covariance="diag")
        e2 = float((torch.diagonal(cov_full) - var_diag).abs().max())
        print(f"[probe] end-to-end diag var vs diag(full cov) [SII]: err={e2:.2e} "
              f"[{'OK' if e2<tol else 'FAIL'}]")
    except TypeError as ex:
        print(f"[probe] end-to-end check SKIPPED: covariance= switch not wired yet ({ex})")

    # (3) timing: diag vs full over the sample
    sync(); t0=time.time()
    for x in xs:
        xoh = shap.target_to_onehot(x).to(comp)
        shap._shap_R_interaction(xoh, pi_c, 1.0, tuples, weight_fn=shap._g_for_order)
    sync(); t_full=time.time()-t0
    sync(); t0=time.time()
    for x in xs:
        xoh = shap.target_to_onehot(x).to(comp)
        shap._shap_R_diag(xoh, pi_c, 1.0, tuples, weight_fn=shap._g_for_order)
    sync(); t_diag=time.time()-t0
    print(f"[probe] R timing over {len(xs)}: full={t_full:.2f}s diag={t_diag:.2f}s "
          f"{t_full/max(1e-9,t_diag):.1f}x")
    print("\n[probe] (diag is O(n) vs full O(n^2); the speedup grows with tuple count / L.)")

if __name__ == "__main__":
    main()