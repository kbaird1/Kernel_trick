#!/usr/bin/env python
"""
probe_cov_breakdown.py
======================
Locate the cost in the per-target COVARIANCE path before optimizing it. For a
sample of targets it times the three components of shap_values / interaction
covariance separately:

    (1) channels / cov-scalars  -- the cheap per-position contractions
    (2) R = m^T K m^{T'}        -- the n x n tuple-pair bivariate loop
                                   (_shap_R_interaction), suspected bottleneck
    (3) mK build + (mK S)(mK S)^T -- the data-correction term

so we know whether the global-covariance win is (a) batching over targets (like
the mean path), (b) vectorizing the n x n R loop, or (c) both.

It also confirms, per target, that the summed components reproduce the covariance
returned by the public method (so the breakdown is faithful, not a re-derivation
that drifts).

Run (project root, allocation):

    python probe_cov_breakdown.py --data ./data/gb1.csv --kernel jenga --gpu \
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
    ap.add_argument("--n-probe", type=int, default=64,
                    help="targets to time the per-target cov path over")
    ap.add_argument("--interaction-order", type=int, default=2)
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
    ktc = shap.ktc
    ktc.set_precision(use_double=True)
    comp = ktc.comp_dtype
    dev = ktc.device
    cuda = str(dev).startswith("cuda")
    pi_c = pi.to(device=dev, dtype=comp)

    xs = [str(s) for s in sfa.seqs[:args.n_probe]]
    positions = [(i,) for i in range(L)]
    inter = []
    for k in range(2, args.interaction_order + 1):
        inter += list(itertools.combinations(range(L), k))
    tuples = positions + inter
    n = len(tuples)
    t = ktc._train_x.reshape(-1, L, sfa.A).shape[0]
    print(f"[probe] L={L} A={sfa.A} t={t} tuples(n)={n} targets={len(xs)} dev={dev}")

    S = ktc.S.to(comp)          # (t, t)
    def sync():
        if cuda: torch.cuda.synchronize()

    # component timers, accumulated over the target sample
    t_chan = t_R = t_mk = t_full = 0.0
    max_recon_err = 0.0
    wfn = shap._g_for_order    # SII weight (interaction path)

    for x in xs:
        x_oh = shap.target_to_onehot(x).to(comp)

        # (1) channels + cov-scalars
        sync(); s0 = time.time()
        c0, c1 = shap._mean_channels(x_oh, pi_c, 1.0)
        _ = shap._cov_scalars(x_oh, pi_c, 1.0)
        sync(); t_chan += time.time() - s0

        # (2) R = m^T K m^{T'}  (the n x n tuple-pair loop)
        sync(); s0 = time.time()
        R = shap._shap_R_interaction(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
        sync(); t_R += time.time() - s0

        # (3) mK build + data-correction (mK S)(mK S)^T
        sync(); s0 = time.time()
        mK = shap._shap_z_interaction(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)  # (n, t)
        zS = mK @ S
        cov_manual = R - zS @ zS.T
        sync(); t_mk += time.time() - s0

        # faithfulness: compare to the public interaction covariance (SII, unnormalized)
        sync(); s0 = time.time()
        _, _, cov_pub = shap.shap_interaction_values(
            x, pi_c, interactions=tuples, normalize="sii", calc_covar=True)
        sync(); t_full += time.time() - s0
        max_recon_err = max(max_recon_err, float((cov_manual - cov_pub).abs().max()))

    N = len(xs)
    print(f"[probe] per-target avg over {N} targets:")
    print(f"[probe]   (1) channels + cov_scalars : {t_chan/N*1e3:8.3f} ms")
    print(f"[probe]   (2) R (n x n pair loop)     : {t_R/N*1e3:8.3f} ms   <-- suspected bottleneck")
    print(f"[probe]   (3) mK + (mK S)(mK S)^T     : {t_mk/N*1e3:8.3f} ms")
    print(f"[probe]   full public cov call        : {t_full/N*1e3:8.3f} ms")
    print(f"[probe] recon max err (manual vs public) = {max_recon_err:.2e}  (want ~0)")
    print()
    frac_R = t_R / max(1e-9, (t_chan + t_R + t_mk))
    print(f"[probe] R is {frac_R:.0%} of the (1)+(2)+(3) cost.")
    if frac_R > 0.6:
        print("[probe] ==> R (the n x n tuple-pair loop) dominates. Vectorizing the "
              "covariance needs BOTH target-batching AND collapsing that n x n loop.")
    else:
        print("[probe] ==> R is NOT dominant; target-batching alone (like the mean "
              "path) likely captures most of the covariance win.")

if __name__ == "__main__":
    main()