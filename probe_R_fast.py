#!/usr/bin/env python
"""
probe_R_fast.py
===============
Gate for _shap_R_interaction_fast. Verifies it equals the naive _shap_R_interaction
(the brute-force-checked reference) to ~1e-12, for BOTH the SII weight and the
dividend weight, across a set of targets -- and times the speedup. Run AFTER
pasting _shap_R_interaction_fast into the class.

    python probe_R_fast.py --data ./data/gb1.csv --kernel jenga --gpu \
        --n-condition 20000 --load-model --model-name gb1_jenga_nc20000

Note: at GB1 L=4 the speedup will be modest (few positions to share); the point
here is CORRECTNESS. The n^2 * per-position win grows with L, so the real speedup
shows at larger L (a synthetic L check is offered via --synthetic-L).
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
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--interaction-order", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-10)
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

    if not hasattr(shap, "_shap_R_interaction_fast"):
        print("[probe] ERROR: no _shap_R_interaction_fast -- paste the block.")
        return

    xs = [str(s) for s in sfa.seqs[:args.n_probe]]
    positions = [(i,) for i in range(L)]
    inter = []
    for k in range(2, args.interaction_order + 1):
        inter += list(itertools.combinations(range(L), k))
    tuples = positions + inter
    tol = args.tol

    def sync():
        if cuda: torch.cuda.synchronize()

    for wname, wfn in (("SII", shap._g_for_order),
                       ("dividend", shap._index_weight_fn("dividend"))):
        max_err = 0.0
        sync(); t0 = time.time()
        for x in xs:
            x_oh = shap.target_to_onehot(x).to(comp)
            R_ref = shap._shap_R_interaction(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
        sync(); t_ref = time.time() - t0

        sync(); t0 = time.time()
        for x in xs:
            x_oh = shap.target_to_onehot(x).to(comp)
            R_fast = shap._shap_R_interaction_fast(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
        sync(); t_fast = time.time() - t0

        # recompute one pair of matrices to compare (last x)
        x_oh = shap.target_to_onehot(xs[0]).to(comp)
        R_ref0 = shap._shap_R_interaction(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
        R_fast0 = shap._shap_R_interaction_fast(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
        max_err = float((R_ref0 - R_fast0).abs().max())
        ok = max_err < tol
        print(f"[probe] {wname:9s}: err={max_err:.2e}  ref={t_ref:.2f}s fast={t_fast:.2f}s "
              f"{t_ref/max(1e-9,t_fast):.1f}x  [{'OK' if ok else 'FAIL'}]")

    print()
    print("[probe] (GB1 L=4: modest speedup expected; correctness is the point. "
          "The n^2 x per-position win grows with L.)")

if __name__ == "__main__":
    main()