#!/usr/bin/env python
"""
probe_R_scaling.py
==================
Measure the _shap_R_interaction_fast speedup as a function of L, on small SYNTHETIC
GPs (no cluster data needed). This answers the question GB1 (L=4) cannot: does the
memoized-free-product R actually deliver a large speedup at the L where covariance
matters (L ~ 12-36)? Correctness (== naive R) is re-checked at every L.

    python probe_R_scaling.py --gpu --Ls 4,8,12,16,20 --t 3000 --interaction-order 2

For each L it fits a quick jenga GP on random sequences, conditions on --t points,
then times both R implementations over a few targets at interaction-order 2.
"""
import argparse, sys, itertools, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--Ls", default="4,8,12,16,20")
    ap.add_argument("--A", type=int, default=20)
    ap.add_argument("--t", type=int, default=3000, help="conditioning points")
    ap.add_argument("--n-data", type=int, default=6000)
    ap.add_argument("--interaction-order", type=int, default=2)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = args.project_root or str(Path(__file__).resolve().parent)
    sys.path.insert(0, str(Path(root) / "modules"))
    sys.path.insert(0, root)
    import numpy as np, torch
    import epik_loader
    epik_loader.ensure_epik_loaded(epik_code_root=root)
    from epik.epik.kernel import JengaKernel
    from epik.epik.model import EpiK
    from epik.epik.utils import encode_seqs
    from modules.kernel_trick_computer_refactored import KernelTrickComputer
    from modules.shap_kernel_trick_computer import ShapKernelTrickComputer, make_background

    dev = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    alphabet = "ACDEFGHIKLMNPQRSTVWY"[:args.A]
    Ls = [int(x) for x in args.Ls.split(",")]
    cuda = dev == "cuda"
    def sync():
        if cuda: torch.cuda.synchronize()

    print(f"[probe] dev={dev} A={args.A} t={args.t} order={args.interaction_order} "
          f"targets={args.n_probe}")
    print(f"[probe] {'L':>4} {'n_tuples':>9} {'ref(s)':>9} {'fast(s)':>9} {'speedup':>8} {'err':>10}")
    rng = np.random.default_rng(0)
    for L in Ls:
        seqs = ["".join(rng.choice(list(alphabet), size=L)) for _ in range(args.n_data)]
        y = rng.standard_normal(args.n_data).astype(float)
        X = encode_seqs(seqs, alphabet=alphabet)
        yv = torch.zeros(args.n_data)
        c = rng.permutation(args.n_data)[:args.t]
        tx = X[c]; ty = torch.as_tensor(y[c], dtype=torch.float32); tyv = yv[c]
        model = EpiK(JengaKernel(n_alleles=args.A, seq_length=L), track_progress=False)
        model.set_data(tx, ty, tyv)
        model.fit(n_iter=20, learning_rate=0.05)
        _ = model.predict(tx[:min(args.t, 2000)], calc_variance=False)
        ktc = KernelTrickComputer(model, alphabet, L)
        if dev != "cpu":
            for nm, v in list(vars(ktc).items()):
                if torch.is_tensor(v): setattr(ktc, nm, v.to(dev))
            ktc.device = dev
        ktc.set_precision(use_double=True)
        shap = ShapKernelTrickComputer(ktc=ktc)
        comp = ktc.comp_dtype
        pi = make_background("uniform", L, alphabet, device=torch.device(dev), dtype=comp)

        positions = [(i,) for i in range(L)]
        inter = []
        for k in range(2, args.interaction_order + 1):
            inter += list(itertools.combinations(range(L), k))
        tuples = positions + inter
        xs = seqs[:args.n_probe]
        wfn = shap._g_for_order

        sync(); t0 = time.time()
        for x in xs:
            xoh = shap.target_to_onehot(x).to(comp)
            R_ref = shap._shap_R_interaction(xoh, pi, 1.0, tuples, weight_fn=wfn)
        sync(); t_ref = time.time() - t0

        sync(); t0 = time.time()
        for x in xs:
            xoh = shap.target_to_onehot(x).to(comp)
            R_fast = shap._shap_R_interaction_fast(xoh, pi, 1.0, tuples, weight_fn=wfn)
        sync(); t_fast = time.time() - t0

        err = float((R_ref - R_fast).abs().max())
        flag = "OK" if err < args.tol else "FAIL"
        print(f"[probe] {L:>4} {len(tuples):>9} {t_ref:>9.2f} {t_fast:>9.2f} "
              f"{t_ref/max(1e-9,t_fast):>7.1f}x {err:>10.1e} [{flag}]")

    print("\n[probe] If speedup climbs sharply with L and err stays ~0, the fast R "
          "is the right foundation for Step 2 (batching the covariance over targets).")

if __name__ == "__main__":
    main()