#!/usr/bin/env python
"""
probe_R_diag.py
===============
Gate for the diagonal covariance path. Every claim the covariance="diag" switch
makes, checked across multiple targets.

WIRING (handoff item 4)
  (0) all four covariance-returning methods actually accept `covariance`, and
      reject a bad value; the default is still "full" and returns a MATRIX.

R-LEVEL IDENTITY
  (1a) _shap_R_diag == diagonal(_shap_R_interaction), for SII and dividend weights.
  (1b) _shap_R_diag on SINGLETON tuples == diagonal(_shap_R_cov). Separate claim:
       the first-order path uses a DIFFERENT reference method, and relies on the
       SII weight at k==1 being self.g.

END-TO-END (public methods, incl. the (mK S) data correction)
  (2a) shap_values                diag var == diagonal(full cov)
  (2b) shap_interaction_values    diag var == diagonal(full cov), at normalize="sii"
       AND normalize="shap", AND on MIXED-order tuples. 'shap' is the only mode that
       exercises the (scale*scale) line; mixed orders are the only case that can catch
       a per-row misalignment of the scale vector.
  (2c) dividend_values            diag var == diagonal(full cov). Plus a distinctness
       assert vs the SII variances -- localizes a missing weight_fn=wfn forward.
  (2d) faith_shap_values          diag var == diagonal(full cov), for BOTH branches:
       tail-empty (cap==ell -> M is a selection matrix -> (M*M) @ a_var) and
       tail-non-empty (cap>ell -> full a_cov required -> ((M @ a_cov) * M).sum).
       Plus the ell==L -> dividends anchor, on means and variances.

TIMING
  (3) R-level and end-to-end diag vs full.

Run AFTER pasting _shap_R_diag and applying the covariance= switch edits.

    python probe_R_diag.py --data ./data/gb1.csv --kernel jenga --gpu \
        --n-condition 20000 --load-model --model-name gb1_jenga_nc20000

Exit code is 0 iff every check passed.
"""
import argparse, sys, itertools, time, inspect
from pathlib import Path

# Accumulated (label, passed) for the closing summary / exit code.
RESULTS = []


def _emit(label, err, tol, extra=""):
    """Print one probe line and record the outcome."""
    ok = err < tol
    RESULTS.append((label, ok))
    print(f"[probe] {label}: err={err:.2e} [{'OK' if ok else 'FAIL'}]{extra}")
    return ok


def _emit_bool(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"[probe] {label}: [{'OK' if ok else 'FAIL'}]{extra}")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kernel", default="jenga")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n-condition", type=int, default=20000)
    ap.add_argument("--load-model", action="store_true")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--n-probe", type=int, default=64,
                    help="targets used for the TIMING loop")
    ap.add_argument("--n-check", type=int, default=4,
                    help="targets used for the CORRECTNESS checks (max err reported)")
    ap.add_argument("--interaction-order", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--skip-faith", action="store_true",
                    help="skip the Faith-Shap checks (they build supersets up to L)")
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
    tol = args.tol

    def sync():
        if cuda: torch.cuda.synchronize()

    def maxerr(a, b):
        return float((a.double() - b.double()).abs().max())

    # ---------------------------------------------------------------- #
    # (0) WIRING: the method exists, all four public methods take
    #     `covariance`, a bad value raises, and the DEFAULT is still full.
    # ---------------------------------------------------------------- #
    print("\n=== (0) wiring ===")
    if not hasattr(shap, "_shap_R_diag"):
        print("[probe] ERROR: shap has no _shap_R_diag -- paste the R_diag block first.")
        sys.exit(1)
    _emit_bool("_shap_R_diag present", True)

    four = ("shap_values", "shap_interaction_values", "dividend_values", "faith_shap_values")
    for name in four:
        sig = inspect.signature(getattr(shap, name))
        has = "covariance" in sig.parameters
        dflt = sig.parameters["covariance"].default if has else None
        _emit_bool(f"{name:24s} accepts covariance= (default {dflt!r})",
                   has and dflt == "full")

    x0 = str(sfa.seqs[0])
    try:
        shap.shap_values(x0, pi_c, calc_covar=True, covariance="bogus")
        _emit_bool("bad covariance= value raises ValueError", False, "  (no raise)")
    except ValueError:
        _emit_bool("bad covariance= value raises ValueError", True)
    except Exception as ex:  # noqa: BLE001
        _emit_bool("bad covariance= value raises ValueError", False, f"  (raised {type(ex).__name__})")

    # Default must be identical to explicit "full", and 2-D.
    _, cov_default = shap.shap_values(x0, pi_c, calc_covar=True)
    _, cov_explicit = shap.shap_values(x0, pi_c, calc_covar=True, covariance="full")
    _emit_bool("default calc_covar returns a MATRIX (unchanged behaviour)",
               cov_default.dim() == 2 and cov_default.shape[0] == cov_default.shape[1])
    _emit("default == explicit covariance='full'", maxerr(cov_default, cov_explicit), tol)

    # ---------------------------------------------------------------- #
    # tuple sets used below
    # ---------------------------------------------------------------- #
    xs_check = [str(s) for s in sfa.seqs[:max(1, args.n_check)]]
    xs_time = [str(s) for s in sfa.seqs[:args.n_probe]]

    positions = [(i,) for i in range(L)]
    inter = []
    for k in range(2, min(args.interaction_order, L) + 1):
        inter += list(itertools.combinations(range(L), k))
    tuples = positions + inter                      # mixed orders 1..interaction_order

    # ---------------------------------------------------------------- #
    # (1a) R_diag vs diagonal of the full interaction R  (both weights)
    # ---------------------------------------------------------------- #
    print("\n=== (1) R-level identity ===")
    for wname, wfn in (("SII", shap._g_for_order),
                       ("dividend", shap._index_weight_fn("dividend"))):
        worst = 0.0
        for x in xs_check:
            x_oh = shap.target_to_onehot(x).to(comp)
            Rfull = shap._shap_R_interaction(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
            Rdiag = shap._shap_R_diag(x_oh, pi_c, 1.0, tuples, weight_fn=wfn)
            worst = max(worst, maxerr(torch.diagonal(Rfull), Rdiag))
        _emit(f"R_diag vs diag(_shap_R_interaction) [{wname:8s}]", worst, tol,
              f"  ({len(xs_check)} targets, {len(tuples)} tuples)")

    # ---------------------------------------------------------------- #
    # (1b) FIRST ORDER: singletons vs diagonal of _shap_R_cov.
    #      Different reference method; relies on g_1 == self.g.
    # ---------------------------------------------------------------- #
    idx = list(range(L))
    singletons = [(int(i),) for i in idx]
    worst = 0.0
    for x in xs_check:
        x_oh = shap.target_to_onehot(x).to(comp)
        R_cov = shap._shap_R_cov(x_oh, pi_c, 1.0, idx)
        R_dg = shap._shap_R_diag(x_oh, pi_c, 1.0, singletons)      # default = SII weight
        worst = max(worst, maxerr(torch.diagonal(R_cov), R_dg))
    _emit("R_diag singletons vs diag(_shap_R_cov) [first order]", worst, tol)

    # ---------------------------------------------------------------- #
    # (2a) shap_values end-to-end
    # ---------------------------------------------------------------- #
    print("\n=== (2) end-to-end public methods ===")
    worst = 0.0
    for x in xs_check:
        _, cov_f = shap.shap_values(x, pi_c, calc_covar=True, covariance="full")
        _, var_d = shap.shap_values(x, pi_c, calc_covar=True, covariance="diag")
        worst = max(worst, maxerr(torch.diagonal(cov_f), var_d))
    _emit("shap_values          diag var vs diag(full cov)", worst, tol)

    # ---------------------------------------------------------------- #
    # (2b) shap_interaction_values: BOTH normalize modes, single + mixed order.
    #      normalize="shap" is the only mode that exercises (scale*scale).
    # ---------------------------------------------------------------- #
    single_order = min(args.interaction_order, L)
    order_tuples = list(itertools.combinations(range(L), single_order))
    for norm in ("sii", "shap"):
        for tname, tset in (("single-order", order_tuples), ("MIXED-order", tuples)):
            worst = 0.0
            for x in xs_check:
                _, _, cov_f = shap.shap_interaction_values(
                    x, pi_c, interactions=tset, normalize=norm,
                    calc_covar=True, covariance="full")
                _, _, var_d = shap.shap_interaction_values(
                    x, pi_c, interactions=tset, normalize=norm,
                    calc_covar=True, covariance="diag")
                worst = max(worst, maxerr(torch.diagonal(cov_f), var_d))
            _emit(f"shap_interaction     diag vs diag(full) "
                  f"[normalize={norm:4s}, {tname}]", worst, tol)

    # Sanity on the normalization itself: at order k the 'shap' variance must be the
    # 'sii' variance / (k!)^2. Catches a scale applied once instead of twice.
    from math import factorial
    worst = 0.0
    for x in xs_check:
        _, _, v_sii = shap.shap_interaction_values(
            x, pi_c, interactions=tuples, normalize="sii",
            calc_covar=True, covariance="diag")
        _, _, v_shp = shap.shap_interaction_values(
            x, pi_c, interactions=tuples, normalize="shap",
            calc_covar=True, covariance="diag")
        fac = torch.tensor([1.0 / factorial(len(T)) ** 2 for T in tuples],
                           device=v_sii.device, dtype=v_sii.dtype)
        worst = max(worst, maxerr(v_shp, v_sii * fac))
    _emit("shap_interaction     diag var: shap == sii/(k!)^2", worst, tol)

    # ---------------------------------------------------------------- #
    # (2c) dividend_values end-to-end + distinctness from SII
    # ---------------------------------------------------------------- #
    worst = 0.0
    gap = float("inf")
    for x in xs_check:
        _, _, dcov_f = shap.dividend_values(
            x, pi_c, interactions=tuples, calc_covar=True, covariance="full")
        _, _, dvar_d = shap.dividend_values(
            x, pi_c, interactions=tuples, calc_covar=True, covariance="diag")
        worst = max(worst, maxerr(torch.diagonal(dcov_f), dvar_d))
        _, _, svar_d = shap.shap_interaction_values(
            x, pi_c, interactions=tuples, normalize="sii",
            calc_covar=True, covariance="diag")
        gap = min(gap, maxerr(dvar_d, svar_d))
    _emit("dividend_values      diag var vs diag(full cov)", worst, tol)
    _emit_bool("dividend diag var is DISTINCT from SII diag var",
               gap > 1e-12,
               f"  gap={gap:.2e}" + ("" if gap > 1e-12 else
                                     "  <- weight_fn=wfn forward likely MISSING"))

    # ---------------------------------------------------------------- #
    # (2d) faith_shap_values: both branches + the ell==L anchor
    # ---------------------------------------------------------------- #
    if args.skip_faith:
        print("[probe] Faith-Shap checks SKIPPED (--skip-faith)")
    else:
        ell = min(2, L)
        rep = list(itertools.combinations(range(L), min(single_order, ell)))
        cases = [("tail EMPTY   cap==ell", ell, ell),
                 ("tail EMPTY   ell==L  ", L, L)]
        if ell < L:
            cases.append(("tail NONEMPTY cap==L ", ell, L))
        for label, ell_c, cap_c in cases:
            rep_c = [T for T in rep if len(T) <= ell_c] or [(0,)]
            worst = 0.0
            for x in xs_check:
                _, _, fcov_f = shap.faith_shap_values(
                    x, pi_c, ell_c, interactions=rep_c, calc_covar=True,
                    order_cap=cap_c, covariance="full")
                _, _, fvar_d = shap.faith_shap_values(
                    x, pi_c, ell_c, interactions=rep_c, calc_covar=True,
                    order_cap=cap_c, covariance="diag")
                worst = max(worst, maxerr(torch.diagonal(fcov_f), fvar_d))
            _emit(f"faith_shap           diag vs diag(full) "
                  f"[ell={ell_c}, cap={cap_c}, {label}]", worst, tol)

        # Anchor: at ell == L, Faith-Shap IS the dividends -- means and variances.
        worst_m = worst_v = 0.0
        for x in xs_check:
            _, fphi, fvar = shap.faith_shap_values(
                x, pi_c, L, interactions=tuples, calc_covar=True, covariance="diag")
            _, dval, dvar = shap.dividend_values(
                x, pi_c, interactions=tuples, calc_covar=True, covariance="diag")
            worst_m = max(worst_m, maxerr(fphi, dval))
            worst_v = max(worst_v, maxerr(fvar, dvar))
        _emit("faith_shap(ell==L) mean vs dividend mean [anchor]", worst_m, tol)
        _emit("faith_shap(ell==L) diag var vs dividend diag var [anchor]", worst_v, tol)

    # ---------------------------------------------------------------- #
    # (3) TIMING: R-level and end-to-end
    # ---------------------------------------------------------------- #
    print("\n=== (3) timing ===")
    sync(); t0 = time.time()
    for x in xs_time:
        xoh = shap.target_to_onehot(x).to(comp)
        shap._shap_R_interaction(xoh, pi_c, 1.0, tuples, weight_fn=shap._g_for_order)
    sync(); t_full = time.time() - t0
    sync(); t0 = time.time()
    for x in xs_time:
        xoh = shap.target_to_onehot(x).to(comp)
        shap._shap_R_diag(xoh, pi_c, 1.0, tuples, weight_fn=shap._g_for_order)
    sync(); t_diag = time.time() - t0
    print(f"[probe] R-level     over {len(xs_time):4d} targets, {len(tuples)} tuples: "
          f"full={t_full:6.2f}s diag={t_diag:6.2f}s  {t_full/max(1e-9,t_diag):5.1f}x")

    n_e2e = min(len(xs_time), 16)
    sync(); t0 = time.time()
    for x in xs_time[:n_e2e]:
        shap.shap_interaction_values(x, pi_c, interactions=tuples, normalize="sii",
                                     calc_covar=True, covariance="full")
    sync(); e_full = time.time() - t0
    sync(); t0 = time.time()
    for x in xs_time[:n_e2e]:
        shap.shap_interaction_values(x, pi_c, interactions=tuples, normalize="sii",
                                     calc_covar=True, covariance="diag")
    sync(); e_diag = time.time() - t0
    print(f"[probe] end-to-end  over {n_e2e:4d} targets, {len(tuples)} tuples: "
          f"full={e_full:6.2f}s diag={e_diag:6.2f}s  {e_full/max(1e-9,e_diag):5.1f}x")
    print("[probe] (diag is O(n) vs full O(n^2); the speedup grows with tuple count / L.)")

    # ---------------------------------------------------------------- #
    # summary
    # ---------------------------------------------------------------- #
    failed = [lab for lab, ok in RESULTS if not ok]
    n_ok = len(RESULTS) - len(failed)
    print(f"\n[probe] ===== {n_ok}/{len(RESULTS)} checks passed =====")
    if failed:
        for lab in failed:
            print(f"[probe]   FAILED: {lab}")
        sys.exit(1)
    print("[probe] diag path is safe to use; analysis.py may now request covariance='diag'.")
    sys.exit(0)


if __name__ == "__main__":
    main()