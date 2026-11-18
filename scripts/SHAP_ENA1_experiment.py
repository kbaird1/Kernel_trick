"""
SHAP_ENA1_experiment.py
=======================

The "easy, obvious" Shapley analysis, built to answer four requests:

  1. cost against the NUMBER OF SEQUENCES rather than sequence length;
  2. the Shapley values of ONE sequence -- a parental strain -- at all 83 loci;
  3. how they compare with what can be done WITHOUT the GKT: single-mutant
     effects (equivalently the first-order gauge-fixed coefficients) around
     each parental wild type, plus a downstream calculation showing what the
     Shapley values capture that the single-mutant scans miss;
  4. Shapley values across genetic backgrounds.

THE IDEA THAT TIES THEM TOGETHER
--------------------------------
Take the BY parent as the target and the RM parent as the reference (a delta
background pi = RM). The first-order Shapley value is then

    phi_i(BY | RM) = sum_{S not containing i} g(|S|) [ f(BY on S+i, RM else)
                                                       - f(BY on S,   RM else) ],

the effect of switching locus i from RM to BY, AVERAGED OVER EVERY ORDER in
which the 83 loci could be switched -- i.e. over 2^82 intermediate genotypes
between the two parents. Its two extreme terms are exactly the two single-mutant
scans that need no GKT:

    S = {}           Delta_i^RM = f(RM with BY at i) - f(RM)      mutation in RM
    S = all others   Delta_i^BY = f(BY) - f(BY with RM at i)      mutation in BY

All three are RM -> BY effects of the same substitution, so they share units
and sign. For an additive landscape the three coincide; any spread between them
is epistasis of locus i with the genetic background along the RM -> BY path.

The downstream calculation is local accuracy. The Shapley values sum EXACTLY to
f(BY) - f(RM). The single-mutant effects do not, and the shortfall
sum_i Delta_i - [f(BY) - f(RM)] is the total epistasis the single-locus picture
leaves out.

Panel (e) widens the view to genetic backgrounds: the RM -> BY effect at each
locus, s_i(x) = phi_i(x) / (z_i(x) - p_i), across segregant targets under the
population (empirical) background, with the parental quantities overlaid.

POSTERIOR. By default the GP posterior is rebuilt EXACTLY (Cholesky of the full
19,833 x 19,833 system) rather than taken from GPyTorch's rank-100 LOVE cache,
which E5 showed misstates posterior traces at this N. Means and variances of
every quantity here then come from the same exact (S, eta). --no-exact-posterior
uses the cache instead.

STAGES
------
  selftest  small synthetic model (L = 10, CPU): the Shapley values for a
            delta background against brute-force subset enumeration, the two
            single-mutant scans against the S = {} and S = all subset terms,
            local accuracy, and Var[sum_i phi_i] against Var[f(BY) - f(RM)]
            from the module's full covariance. The compute stage refuses to
            run if this fails.
  compute   the real model. Writes CSVs + meta JSON.
  plot      draws the figure from those CSVs. Seconds, no GPU. With
            --read-from it reads another run's folder and writes to --out-dir,
            so earlier figures are never overwritten.

USAGE
-----
    python scripts/SHAP_ENA1_experiment.py --gpu \\
        --params-path models/qtls.li.hq/qtls_li_hq.full.1.Connectedness.model_params.pth \\
        --timing-csv output/ENA1_paper_rep1/E1_timing.csv \\
        --out-dir output/SHAP_ENA1_rep1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _d in (_HERE, _ROOT / "modules", _ROOT):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import ENA1_paper_analysis as epa                              # noqa: E402
from ENA1_paper_analysis import (                              # noqa: E402
    GENE_BY_LOCUS, RM_LETTER, BY_LETTER, C_RM, C_BY, C_NEU, C_ACC,
    PAPER_WT_FITNESS, banner, sub, write_csv, save_fig, _style, _panel,
    gene_label, set_seq_char,
)
from redraw_figures import _draw_cost_panel_seqspace           # noqa: E402

SCRIPT_REVISION = "shap-ena1-rev2-2026-10-05"
REQUIRED_EPA_REVISION = "rev9-2026-09-19"
C_PHI = "#2E3B55"
GPYTORCH_NOISE_FLOOR = 1e-4
DEFAULT_OUT = epa.REPO_ROOT / "output" / "SHAP_ENA1"


class _Tee:
    """Duplicate a stream to a file, flushed on every write."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = open(path, "w", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def close(self):
        try:
            self.file.close()
        except Exception:                                      # noqa: BLE001
            pass


# =========================================================================== #
# posterior quantities without the GKT: point functionals of f
# =========================================================================== #

class Posterior:
    """A GP posterior in the form every quantity here needs.

        mean(c)  = (c^T K_{*X}) eta
        var(c)   = c^T K c - || (c^T K_{*X}) S ||^2,     S S^T = Gamma^-1

    for any linear functional c of f. `site` are the per-position kernel
    tables (K = prod_p site[p]), `train_idx` the (t, L) training alleles.
    """

    def __init__(self, site, train_idx, S, eta, label: str):
        self.site, self.train_idx, self.S, self.eta = site, train_idx, S, eta
        self.label = label


def _seq_index(seqs, alphabet, device):
    import torch
    return torch.tensor([[alphabet.index(c) for c in str(s)] for s in seqs],
                        device=device, dtype=torch.long)


def _kmat(site, Ia, Ib):
    """Prior kernel between two allele-index sets: prod_p site[p][a_p, b_p]."""
    import torch
    out = torch.ones(Ia.shape[0], Ib.shape[0], dtype=site.dtype,
                     device=site.device)
    for p in range(site.shape[0]):
        out.mul_(site[p][Ia[:, p][:, None], Ib[:, p][None, :]])
    return out


def point_functionals(post: Posterior, seqs, W: np.ndarray, alphabet: str):
    """Mean and variance of c_f = sum_s W[f, s] delta_{seq_s}, for every row f.

    This is everything one can do with GP point predictions alone -- no GKT:
    single-mutant effects, the parental difference, and sums of them.
    """
    import torch
    dev, dt = post.site.device, post.site.dtype
    I = _seq_index(seqs, alphabet, dev)
    KsX = _kmat(post.site, I, post.train_idx)                     # (n, t)
    Kss = _kmat(post.site, I, I)                                  # (n, n)
    Wt = torch.as_tensor(np.asarray(W, dtype=float), device=dev, dtype=dt)
    mean = Wt @ (KsX @ post.eta)
    prior = ((Wt @ Kss) * Wt).sum(dim=1)
    D = (Wt @ KsX) @ post.S
    var = prior - (D * D).sum(dim=1)
    return mean.cpu().numpy(), var.cpu().numpy()


def shap_one_target(sfa, post: Posterior, x: str, pi, full_cov: bool = False):
    """phi_i(x) for every locus, with posterior variance, from (S, eta).

    Uses the SHAP module's own GKT builders for the mean rows (mK = m^T K_{*X})
    and the prior form (R = m^T K m), and contracts them with THIS posterior's
    S and eta -- which is what lets the exact rebuild replace the LOVE cache
    without touching the module.
    """
    shap = sfa.shap
    pi_c, comp = shap._prepare_pi(pi, True)
    L = sfa.L
    idx = list(range(L))
    c0, c1 = shap._batched_channels([x], pi_c, 1.0)
    mK = shap._first_order_mK(c0, c1, idx)[0].to(post.S.dtype)    # (L, t)
    mean = mK @ post.eta
    zS = mK @ post.S
    x_oh = shap.target_to_onehot(x).to(comp)
    tuples = [(i,) for i in idx]
    Rd = shap._shap_R_diag(x_oh, pi_c, 1.0, tuples).to(post.S.dtype)
    var = Rd - (zS * zS).sum(dim=1)
    cov = None
    if full_cov:
        Rf = shap._shap_R_interaction(x_oh, pi_c, 1.0, tuples).to(post.S.dtype)
        cov = (Rf - zS @ zS.T).cpu().numpy()
    return mean.cpu().numpy(), var.cpu().numpy(), cov


def population_effects(sfa, post: Posterior, seqs, pi, batch: int = 4,
                       progress: bool = True) -> np.ndarray:
    """phi_i(x) for many targets (means only) from this posterior's eta."""
    import torch
    shap = sfa.shap
    pi_c, _ = shap._prepare_pi(pi, True)
    idx = list(range(sfa.L))
    out = np.empty((len(seqs), sfa.L))
    n_blocks = math.ceil(len(seqs) / batch)
    t0 = time.perf_counter()
    for bi, s0 in enumerate(range(0, len(seqs), batch)):
        part = [str(s) for s in seqs[s0:s0 + batch]]
        c0, c1 = shap._batched_channels(part, pi_c, 1.0)
        mK = shap._first_order_mK(c0, c1, idx).to(post.eta.dtype)  # (b, L, t)
        out[s0:s0 + len(part)] = (mK @ post.eta).cpu().numpy()
        del c0, c1, mK
        if progress and (bi % max(1, n_blocks // 10) == 0
                         or bi == n_blocks - 1):
            el = time.perf_counter() - t0
            print(f"    targets {min(s0 + batch, len(seqs)):>6,}/{len(seqs):,}"
                  f"   {el:7.1f} s", flush=True)
    if str(post.S.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def parental_design(L: int):
    """Sequences and weight matrix for every point functional the figure uses.

    -> (seqs, W, names). Rows of W:
         f_RM, f_BY, gap = f(BY) - f(RM),
         d_rm_i = f(RM with BY at i) - f(RM)          (mutation in RM),
         d_by_i = f(BY) - f(BY with RM at i)          (mutation in BY),
         sum_d_rm, sum_d_by.
    """
    RM, BY = RM_LETTER * L, BY_LETTER * L
    seqs = ([RM, BY] + [set_seq_char(RM, i, BY_LETTER) for i in range(L)]
            + [set_seq_char(BY, i, RM_LETTER) for i in range(L)])
    rm_off, by_off = 2, 2 + L
    n = len(seqs)
    rows, names = [], []

    def row():
        r = np.zeros(n)
        rows.append(r)
        return r
    r = row(); r[0] = 1; names.append("f_RM")
    r = row(); r[1] = 1; names.append("f_BY")
    r = row(); r[1] = 1; r[0] = -1; names.append("gap")
    for i in range(L):
        r = row(); r[rm_off + i] = 1; r[0] = -1; names.append(f"d_rm_{i}")
    for i in range(L):
        r = row(); r[1] = 1; r[by_off + i] = -1; names.append(f"d_by_{i}")
    W = np.vstack(rows)
    s_rm = W[3:3 + L].sum(axis=0)
    s_by = W[3 + L:3 + 2 * L].sum(axis=0)
    W = np.vstack([W, s_rm, s_by])
    names += ["sum_d_rm", "sum_d_by"]
    return seqs, W, names


# =========================================================================== #
# selftest: small synthetic model against brute force
# =========================================================================== #

def stage_selftest(args) -> bool:
    import torch
    banner(f"self-test on a synthetic model (L = {args.selftest_L}, CPU)")
    ep = epa.bootstrap_epik(args.project_root)
    L = int(args.selftest_L)
    sfa = epa.build_synthetic_model(ep, L, args.selftest_n_train, args.seed,
                                    device="cpu", family="Connectedness")
    epa.ensure_double(sfa)
    ktc = sfa.ktc
    comp = ktc.comp_dtype
    site = ktc.site_kernels.to(torch.float64)
    A = sfa.A
    train_idx = ktc._train_x.reshape(-1, L, A).argmax(dim=-1)
    post = Posterior(site, train_idx, ktc.S.to(torch.float64),
                     ktc.Qy.to(torch.float64).reshape(-1), "cache")
    RM, BY = RM_LETTER * L, BY_LETTER * L
    pi_rm = epa.make_pi(sfa, "rm_wildtype")

    phi, var, cov = shap_one_target(sfa, post, BY, pi_rm, full_cov=True)
    phi_mod, var_mod = sfa.shap.shap_values(BY, pi_rm, calc_covar=True,
                                            covariance="diag", use_double=True)
    phi_mod, var_mod = epa._to_np(phi_mod), epa._to_np(var_mod)

    v = epa.brute_force_subset_table(sfa, BY, pi_rm)
    phi_bf = epa.bf_shap_from_table(v, L)
    full = (1 << L) - 1
    d_rm_bf = np.array([v[1 << i] - v[0] for i in range(L)])
    d_by_bf = np.array([v[full] - v[full ^ (1 << i)] for i in range(L)])

    seqs, W, names = parental_design(L)
    mean, pvar = point_functionals(post, seqs, W, sfa.alphabet)
    nm = {k: j for j, k in enumerate(names)}
    d_rm = mean[[nm[f"d_rm_{i}"] for i in range(L)]]
    d_by = mean[[nm[f"d_by_{i}"] for i in range(L)]]
    gap = mean[nm["gap"]]

    scale = max(float(np.abs(phi_bf).max()), 1e-12)
    vscale = max(float(np.abs(var_mod).max()), 1e-30)
    checks = {
        "phi: exact route vs brute force": float(np.abs(phi - phi_bf).max()) / scale,
        "phi: exact route vs shap_values": float(np.abs(phi - phi_mod).max()) / scale,
        "var(phi): exact route vs shap_values":
            float(np.abs(var - var_mod).max()) / vscale,
        "Delta^RM vs subset term S = {}": float(np.abs(d_rm - d_rm_bf).max()) / scale,
        "Delta^BY vs subset term S = all": float(np.abs(d_by - d_by_bf).max()) / scale,
        "local accuracy: sum phi - [f(BY) - f(RM)]":
            abs(float(phi.sum()) - float(gap)) / scale,
        "Var[sum phi] (full cov) vs Var[f(BY) - f(RM)]":
            abs(float(cov.sum()) - float(pvar[nm["gap"]]))
            / max(abs(float(pvar[nm["gap"]])), 1e-30),
    }
    ok = True
    for k, e in checks.items():
        good = e < 1e-8
        ok &= good
        print(f"  {k:<48} rel err {e:.2e}  [{'PASS' if good else 'FAIL'}]")
    print(f"\n  [{'PASS' if ok else 'FAIL'}] self-test")
    return bool(ok)


# =========================================================================== #
# compute
# =========================================================================== #

def _extra_noise(sfa) -> float:
    try:
        return float(np.asarray(epa._to_np(
            sfa.model.likelihood.second_noise_covar.noise)).reshape(-1)[0])
    except Exception:                                          # noqa: BLE001
        return 0.0


def build_posterior(args, sfa, df) -> Posterior:
    """The exact posterior (default) or GPyTorch's cache."""
    import torch
    ktc = sfa.ktc
    epa.ensure_double(sfa)
    dev = sfa.shap.device
    site = ktc.site_kernels.to(device=dev, dtype=torch.float64)
    L, A = sfa.L, sfa.A
    train_idx = ktc._train_x.reshape(-1, L, A).argmax(dim=-1).to(dev)
    eta_cache = ktc.Qy.to(device=dev, dtype=torch.float64).reshape(-1)
    if not args.exact_posterior:
        print("  posterior: GPyTorch LOVE cache "
              f"(rank {int(ktc.S.shape[-1])})")
        return Posterior(site, train_idx,
                         ktc.S.to(device=dev, dtype=torch.float64), eta_cache,
                         f"GPyTorch cache, rank {int(ktc.S.shape[-1])}")

    sub(f"exact posterior: Cholesky of the {len(df):,} x {len(df):,} system")
    t0 = time.perf_counter()
    y = torch.as_tensor(df["y"].to_numpy(dtype=float) - sfa.y_offset,
                        device=dev, dtype=torch.float64)
    extra = _extra_noise(sfa)
    noise = np.maximum(df["y_var"].to_numpy(dtype=float) * args.yvar_scale,
                       GPYTORCH_NOISE_FLOOR) + extra
    K = _kmat(site, train_idx, train_idx)
    K.diagonal().add_(torch.as_tensor(noise, device=dev, dtype=torch.float64))
    Lc = torch.linalg.cholesky(K)
    del K
    t = Lc.shape[0]
    Linv = torch.linalg.solve_triangular(
        Lc, torch.eye(t, device=dev, dtype=torch.float64), upper=False)
    del Lc
    eta = Linv.T @ (Linv @ y)
    S = Linv.T                                     # S S^T = Gamma^-1
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    rel = float(torch.linalg.norm(eta - eta_cache) / torch.linalg.norm(eta))
    print(f"  done in {time.perf_counter() - t0:.1f} s  (extra noise "
          f"{extra:.3e}, noise floor {GPYTORCH_NOISE_FLOOR:g})")
    print(f"  cross-check: ||eta_exact - eta_gpytorch|| / ||eta_exact|| = "
          f"{rel:.2e}   (~1e-3 is float32 GPyTorch; >1e-2 means the rebuild "
          "does not match the fitted model -- check --yvar-scale)")
    return Posterior(site, train_idx, S, eta, "exact float64 posterior")


def stage_compute(args, out_dir: Path) -> dict:
    import torch
    banner("compute: parental Shapley values and single-mutant scans")
    df = epa.load_segregants(Path(args.data_dir) / "qtls_li_hq.csv")
    sfa = epa.load_model_and_shap(args, df)
    if sfa is None:
        print("  model unavailable")
        return {}
    epa.check_base_values(sfa)
    L = sfa.L
    post = build_posterior(args, sfa, df)

    # ---- the BY parent against the RM reference --------------------------- #
    sub("Shapley values of the BY parent against the RM reference (GKT)")
    pi_rm = epa.make_pi(sfa, "rm_wildtype")
    t0 = time.perf_counter()
    phi, phi_var, phi_cov = shap_one_target(sfa, post, BY_LETTER * L, pi_rm,
                                            full_cov=args.full_cov_check)
    t_phi = time.perf_counter() - t0
    print(f"  {L} Shapley values, each a 2^{L - 1}-term sum, with posterior "
          f"variances: {t_phi:.1f} s")
    neg = int((phi_var < 0).sum())
    if neg:
        print(f"  !! {neg} negative variances (min {phi_var.min():+.2e}); "
              "set to NaN")
    phi_sd = np.sqrt(np.where(phi_var >= 0, phi_var, np.nan))

    if args.compare_cache:
        mod = epa._to_np(sfa.shap.shap_values(BY_LETTER * L, pi_rm,
                                              use_double=True))
        print(f"  cross-check against the module's own shap_values (cache "
              f"route): max |diff| / max |phi| = "
              f"{np.abs(mod - phi).max() / np.abs(phi).max():.2e}")

    # ---- the single-mutant scans: point predictions only ------------------ #
    sub("single-mutant scans around each parent (GP point predictions, no GKT)")
    seqs, W, names = parental_design(L)
    mean, var = point_functionals(post, seqs, W, sfa.alphabet)
    nm = {k: j for j, k in enumerate(names)}
    sd = np.sqrt(np.clip(var, 0.0, None))
    for k in ("f_RM", "f_BY"):
        mean[nm[k]] += sfa.y_offset
    gap = float(mean[nm["gap"]])
    resid = float(phi.sum()) - gap
    print(f"  f(RM) = {mean[nm['f_RM']]:+.4f} +- {sd[nm['f_RM']]:.4f}    "
          f"f(BY) = {mean[nm['f_BY']]:+.4f} +- {sd[nm['f_BY']]:.4f}")
    print(f"  f(BY) - f(RM) = {gap:+.4f} +- {sd[nm['gap']]:.4f}   "
          f"(source paper {PAPER_WT_FITNESS['BY'] - PAPER_WT_FITNESS['RM']:+.2f})")
    print(f"  sum_i phi_i   = {phi.sum():+.4f}   local-accuracy residual "
          f"{resid:+.2e}")
    if phi_cov is not None:
        v_sum = float(phi_cov.sum())
        print(f"  Var[sum phi] from the full Shapley covariance {v_sum:.4e} vs "
              f"Var[f(BY) - f(RM)] {var[nm['gap']]:.4e}  (must agree: the two "
              "are the same linear functional)")
    print(f"  sum_i Delta_i^RM = {mean[nm['sum_d_rm']]:+.4f} +- "
          f"{sd[nm['sum_d_rm']]:.4f}   (miss {mean[nm['sum_d_rm']] - gap:+.4f})")
    print(f"  sum_i Delta_i^BY = {mean[nm['sum_d_by']]:+.4f} +- "
          f"{sd[nm['sum_d_by']]:.4f}   (miss {mean[nm['sum_d_by']] - gap:+.4f})")
    print("  The misses are the epistasis the single-locus scans leave out; the "
          "Shapley values close the books exactly.")

    par = pd.DataFrame({
        "locus": np.arange(L), "gene": [GENE_BY_LOCUS.get(i) for i in range(L)],
        "phi": phi, "phi_sd": phi_sd,
        "d_rm": mean[[nm[f"d_rm_{i}"] for i in range(L)]],
        "d_rm_sd": sd[[nm[f"d_rm_{i}"] for i in range(L)]],
        "d_by": mean[[nm[f"d_by_{i}"] for i in range(L)]],
        "d_by_sd": sd[[nm[f"d_by_{i}"] for i in range(L)]],
    })
    par["spread_rm_by"] = par["d_by"] - par["d_rm"]
    top = par.reindex(par["phi"].abs().sort_values(ascending=False).index)
    print("\n  top loci by |phi|:")
    print(top.head(12)[["locus", "gene", "phi", "phi_sd", "d_rm", "d_by"]]
          .to_string(index=False))
    write_csv(par, out_dir / "SHAP_ENA1_parental.csv")
    summ = pd.DataFrame({
        "quantity": ["f_RM", "f_BY", "gap", "sum_phi", "sum_d_rm", "sum_d_by",
                     "local_accuracy_residual"],
        "mean": [mean[nm["f_RM"]], mean[nm["f_BY"]], gap, float(phi.sum()),
                 mean[nm["sum_d_rm"]], mean[nm["sum_d_by"]], resid],
        "sd": [sd[nm["f_RM"]], sd[nm["f_BY"]], sd[nm["gap"]], sd[nm["gap"]],
               sd[nm["sum_d_rm"]], sd[nm["sum_d_by"]], np.nan],
    })
    write_csv(summ, out_dir / "SHAP_ENA1_summary.csv")

    # ---- across genetic backgrounds -------------------------------------- #
    n_bg = 0
    if args.e2_long_csv:
        sub(f"backgrounds: reusing {args.e2_long_csv}")
        lg = pd.read_csv(args.e2_long_csv, usecols=["target_index", "locus", "h"])
        lg["s"] = -lg["h"]           # E2's h is BY -> RM; s is RM -> BY
        write_csv(lg[["target_index", "locus", "s"]],
                  out_dir / "SHAP_ENA1_backgrounds.csv")
        n_bg = int(lg["target_index"].nunique())
    elif args.n_backgrounds != 0:
        rng = np.random.default_rng(args.seed)
        n_bg = (len(df) if args.n_backgrounds < 0
                else min(int(args.n_backgrounds), len(df)))
        pick = np.sort(rng.choice(len(df), size=n_bg, replace=False))
        bg_seqs = [str(s) for s in df["seq"].to_numpy()[pick]]
        sub(f"Shapley values across {n_bg:,} segregant backgrounds "
            "(population background pi = empirical frequencies)")
        X = epa.genotype_matrix(df["seq"])
        counts = epa.empirical_counts_from_X(X, sfa.alphabet)
        pi_emp = epa.make_pi(sfa, "empirical", empirical_counts=counts)
        p = epa._to_np(pi_emp)[:, epa.b_index(sfa.alphabet)]
        phi_bg = population_effects(sfa, post, bg_seqs, pi_emp,
                                    batch=args.shap_batch_size)
        z = epa.allele_indicator(bg_seqs, L).astype(float)
        s = phi_bg / (z - p[None, :])           # RM -> BY effect at each locus
        lg = pd.DataFrame({"target_index": np.repeat(pick, L),
                           "locus": np.tile(np.arange(L), n_bg),
                           "s": s.ravel()})
        write_csv(lg, out_dir / "SHAP_ENA1_backgrounds.csv")

    if args.timing_csv and Path(args.timing_csv).exists():
        pd.read_csv(args.timing_csv).to_csv(out_dir / "E1_timing.csv",
                                            index=False)
        print(f"  copied {args.timing_csv} -> {out_dir / 'E1_timing.csv'}")
    meta = {"revision": SCRIPT_REVISION, "N": int(len(df)), "L": int(L),
            "posterior": post.label, "yvar_scale": float(args.yvar_scale),
            "n_backgrounds": n_bg, "seconds_parental_shap": t_phi}
    (out_dir / "SHAP_ENA1_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_dir / 'SHAP_ENA1_meta.json'}")
    return meta


def stage_parental(args, out_dir: Path) -> bool:
    """Shapley values phi_i of each parental genotype at every locus.

    Target: the all-RM (and all-BY) parent. Background: the empirical allele
    frequencies, i.e. the same pi as the main SHAP analysis (E2), so these
    phi_i are directly comparable to the paper's other Shapley numbers. These
    are the raw phi_i, NOT h_i: for the RM parent z_i = 0 at every locus, so
    phi_i = (1[x_i = RM] - pi^i_RM) h_i = p_i h_i with p_i the BY frequency.

    Also records phi_0 = E_pi[f] and f(x), and checks local accuracy,
    f(x) = phi_0 + sum_i phi_i, from the same posterior.
    """
    import torch
    banner("parental: Shapley values of each parent (empirical background)")
    df = epa.load_segregants(Path(args.data_dir) / "qtls_li_hq.csv")
    sfa = epa.load_model_and_shap(args, df)
    if sfa is None:
        print("  model unavailable")
        return False
    L = sfa.L
    post = build_posterior(args, sfa, df)
    X = epa.genotype_matrix(df["seq"])
    counts = epa.empirical_counts_from_X(X, sfa.alphabet)
    pi = epa.make_pi(sfa, "empirical", empirical_counts=counts)
    pi_d = pi.to(device=post.site.device, dtype=post.site.dtype)
    ib = epa.b_index(sfa.alphabet)
    p_by = epa._to_np(pi)[:, ib]

    # phi_0 = E_pi[f]: the kernel features of the product distribution pi are
    # prod_p (pi_p^T site_p)[y_tp] for each training sequence y_t.
    w = torch.einsum("pa,pab->pb", pi_d, post.site)               # (L, A)
    Kpi = torch.ones(post.train_idx.shape[0], dtype=post.site.dtype,
                     device=post.site.device)
    for q in range(L):
        Kpi = Kpi * w[q][post.train_idx[:, q]]
    phi0 = float(Kpi @ post.eta) + sfa.y_offset

    rows, summ = [], []
    for name, letter_ in (("RM", RM_LETTER), ("BY", BY_LETTER)):
        x = letter_ * L
        t0 = time.perf_counter()
        phi, var, _ = shap_one_target(sfa, post, x, pi)
        fx_mean, _ = point_functionals(post, [x], np.ones((1, 1)), sfa.alphabet)
        fx = float(fx_mean[0]) + sfa.y_offset
        resid = fx - phi0 - float(phi.sum())
        sd = np.sqrt(np.where(var >= 0, var, np.nan))
        print(f"  {name} parent: f(x) = {fx:+.4f}, phi_0 = {phi0:+.4f}, "
              f"sum phi = {phi.sum():+.4f}, local-accuracy residual "
              f"{resid:+.2e}  [{time.perf_counter() - t0:.1f} s]")
        top = np.argsort(-np.abs(phi))[:6]
        print("    largest |phi|: " + ", ".join(
            f"{epa.gene_label(i)} {phi[i]:+.4f}" for i in top))
        for i in range(L):
            rows.append({"target": name, "locus": i,
                         "gene": GENE_BY_LOCUS.get(i), "phi": phi[i],
                         "phi_sd": sd[i], "p_BY": p_by[i],
                         "allele": letter_})
        summ.append({"target": name, "f_x": fx, "phi_0": phi0,
                     "sum_phi": float(phi.sum()),
                     "local_accuracy_residual": resid})
    write_csv(pd.DataFrame(rows), out_dir / "SHAP_parental_targets.csv")
    write_csv(pd.DataFrame(summ), out_dir / "SHAP_parental_summary.csv")
    (out_dir / "SHAP_parental_meta.json").write_text(json.dumps({
        "revision": SCRIPT_REVISION, "N": int(len(df)), "L": int(L),
        "background": "empirical", "posterior": post.label,
        "yvar_scale": float(args.yvar_scale)}, indent=2))
    return True


# =========================================================================== #
# figure
# =========================================================================== #

def stage_plot(args, out_dir: Path) -> None:
    src = Path(args.read_from) if args.read_from else out_dir
    if not src.is_absolute():
        src = epa.REPO_ROOT / src
    banner(f"plot: reading {src}, writing {out_dir}")
    par = pd.read_csv(src / "SHAP_ENA1_parental.csv")
    summ = pd.read_csv(src / "SHAP_ENA1_summary.csv").set_index("quantity")
    meta = json.loads((src / "SHAP_ENA1_meta.json").read_text())
    bpath = src / "SHAP_ENA1_backgrounds.csv"
    bg = pd.read_csv(bpath) if bpath.exists() else None
    tpath = (Path(args.timing_csv) if args.timing_csv
             else src / "E1_timing.csv")
    timing = pd.read_csv(tpath) if tpath.exists() else None
    if timing is None:
        print(f"  [missing] {tpath}: panel (a) will be blank")
    if src.resolve() != out_dir.resolve():
        (out_dir / "SHAP_ENA1_meta.json").write_text(json.dumps(
            dict(meta, replotted_from=str(src)), indent=2))
    fig_shap_ena1(par, summ, bg, timing, meta, args, out_dir)


def fig_shap_ena1(par, summ, bg, timing, meta, args, out_dir: Path) -> None:
    _style()
    L = int(meta["L"])
    loci = par["locus"].to_numpy()
    phi, phi_sd = par["phi"].to_numpy(), par["phi_sd"].to_numpy()
    d_rm, d_by = par["d_rm"].to_numpy(), par["d_by"].to_numpy()
    order = np.argsort(-np.abs(phi))
    topk = [int(i) for i in order[:int(args.top_k)]]

    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=0.95, left=0.055,
                          right=0.985, top=0.86, bottom=0.12)

    # ---- (a) cost against the number of sequences ----------------------- #
    ax = fig.add_subplot(gs[0, 0:2])
    if timing is not None and len(timing):
        _draw_cost_panel_seqspace(ax, timing, fontsize=6.4)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "pass --timing-csv for the cost panel",
                ha="center", va="center", fontsize=9)
    _panel(ax, "a", "Per-target wall-clock cost")

    # ---- (b) Shapley values of the BY parent at all loci ---------------- #
    ax = fig.add_subplot(gs[0, 2:6])
    ax.bar(loci, phi, width=0.8, color=C_PHI, zorder=3,
           label="Shapley value $\\phi_i$(BY | RM)  (GKT)",
           yerr=np.nan_to_num(phi_sd), error_kw=dict(lw=0.6, ecolor="0.35",
                                                    capsize=0))
    ax.plot(loci, d_rm, "o", ms=3.4, mfc="none", mec=C_RM, mew=0.9, zorder=4,
            label="single mutant in RM, $\\Delta_i^{RM}$  (no GKT)")
    ax.plot(loci, d_by, "^", ms=3.6, mfc="none", mec=C_BY, mew=0.9, zorder=4,
            label="single mutant in BY, $\\Delta_i^{BY}$  (no GKT)")
    ax.axhline(0, color="k", lw=0.7)
    named = [i for i in range(L) if GENE_BY_LOCUS.get(i)]
    ax.set_xticks(named)
    ax.set_xticklabels([gene_label(i) for i in named], rotation=90,
                       fontsize=7)
    ax.set_xlim(-1, L)
    ax.set_xlabel("locus (genome order; mapped QTLs labelled)")
    ax.set_ylabel("RM$\\to$BY effect at locus $i$")
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    _panel(ax, "b", "Shapley values of the BY parent (reference: RM parent), "
                    "all loci")

    # ---- (c) the books: does each decomposition sum to f(BY) - f(RM)? ---- #
    ax = fig.add_subplot(gs[1, 0:2])
    keys = ["gap", "sum_phi", "sum_d_rm", "sum_d_by"]
    labs = ["$f$(BY)$-f$(RM)", "$\\sum_i\\phi_i$", "$\\sum_i\\Delta_i^{RM}$",
            "$\\sum_i\\Delta_i^{BY}$"]
    cols = ["0.55", C_PHI, C_RM, C_BY]
    vals = [float(summ.loc[k, "mean"]) for k in keys]
    sds = [float(summ.loc[k, "sd"]) for k in keys]
    ax.bar(range(4), vals, yerr=sds, color=cols, width=0.65, zorder=3,
           error_kw=dict(lw=0.9, ecolor="0.2", capsize=3))
    paper_gap = PAPER_WT_FITNESS["BY"] - PAPER_WT_FITNESS["RM"]
    ax.axhline(paper_gap, color="k", ls="--", lw=0.8, zorder=1,
               label=f"source paper ({paper_gap:+.2f})")
    ax.legend(fontsize=7, loc="upper left")
    ax.axhline(0, color="k", lw=0.7)
    top_ = max(abs(v) + (0 if not np.isfinite(e) else e)
               for v, e in zip(vals, sds))
    for xx, v, e in zip(range(4), vals, sds):
        e = e if np.isfinite(e) else 0.0
        off = e + 0.03 * top_
        ax.text(xx, v + off if v >= 0 else v - off, f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.set_ylim(min(0.0, min(vals) - 0.25 * top_), 1.18 * top_)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("fitness difference")
    _panel(ax, "c", "Sum over loci vs the parental difference")

    # ---- (d) Shapley value against the two single-mutant effects -------- #
    ax = fig.add_subplot(gs[1, 2:4])
    lim = float(np.nanmax(np.abs(np.r_[phi, d_rm, d_by]))) * 1.12
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.9, ls="--", zorder=1)
    for i in range(L):
        ax.plot([d_rm[i], d_by[i]], [phi[i], phi[i]], "-", color="0.7",
                lw=0.8, zorder=2)
    ax.plot(d_rm, phi, "o", ms=3.6, mfc="none", mec=C_RM, mew=0.9, zorder=3,
            label="$\\Delta_i^{RM}$")
    ax.plot(d_by, phi, "^", ms=3.8, mfc="none", mec=C_BY, mew=0.9, zorder=3,
            label="$\\Delta_i^{BY}$")
    for i in topk[:6]:
        ax.annotate(gene_label(i), (max(d_rm[i], d_by[i]), phi[i]),
                    xytext=(4, -2), textcoords="offset points", fontsize=7,
                    color="0.25")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("single-mutant effect $\\Delta_i$  (no GKT)")
    ax.set_ylabel("Shapley value $\\phi_i$(BY | RM)")
    ax.legend(fontsize=7.5, loc="upper left")
    _panel(ax, "d", "Shapley value vs single-mutant scans")

    # ---- (e) across genetic backgrounds -------------------------------- #
    ax = fig.add_subplot(gs[1, 4:6])
    if bg is not None and len(bg):
        show = [i for i in topk if i in set(bg["locus"].astype(int))]
        groups = [bg.loc[bg["locus"] == i, "s"].to_numpy() for i in show]
        xs = np.arange(len(show))
        parts = ax.violinplot(groups, positions=xs, widths=0.8,
                              showextrema=False, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(C_NEU)
            pc.set_alpha(0.35)
            pc.set_linewidth(0)
        ax.plot(xs, phi[show], "D", ms=4.2, color=C_PHI, zorder=4,
                label="$\\phi_i$(BY | RM)")
        ax.plot(xs, d_rm[show], "o", ms=3.8, mfc="none", mec=C_RM, mew=1.0,
                zorder=4, label="$\\Delta_i^{RM}$")
        ax.plot(xs, d_by[show], "^", ms=4.0, mfc="none", mec=C_BY, mew=1.0,
                zorder=4, label="$\\Delta_i^{BY}$")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([gene_label(i) for i in show], rotation=90,
                           fontsize=7.5)
        ax.set_ylabel("RM$\\to$BY effect at locus $i$")
        ax.legend(fontsize=7, loc="upper right")
        nbg = int(meta.get("n_backgrounds", 0)) or int(bg["target_index"]
                                                     .nunique())
        _panel(ax, "e", f"Effect across {nbg:,} genetic backgrounds")
    else:
        ax.axis("off")
        _panel(ax, "e", "Effect across genetic backgrounds")

    N = int(meta.get("N", 0))
    fig.suptitle("First-order Shapley values of the parental strains against "
                 "single-mutant scans\n"
                 f"GP conditioned on {N:,} measured segregants "
                 f"({meta.get('posterior', '')}); whiskers $\\pm$1 posterior "
                 "s.d.", fontsize=11.5, y=0.975)
    save_fig(fig, out_dir, "fig_SHAP_ENA1")


# =========================================================================== #
# CLI
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="selftest,compute,plot",
                    help="comma-separated: selftest, compute, plot, parental "
                         "(parental = Shapley values of each parental "
                         "genotype under the empirical background)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--read-from", type=Path, default=None,
                    help="plot stage only: read CSVs from this earlier run "
                         "and write the figure to --out-dir")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path,
                    default=epa.REPO_ROOT / "data" / "qtls_li_hq")
    ap.add_argument("--params-path", type=Path, default=None)
    ap.add_argument("--project-root", type=Path, default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-center-y", action="store_true")
    ap.add_argument("--yvar-scale", type=float, default=0.05,
                    help="MUST match the main SHAP run (its run.log prints "
                         "'y_var scaled by ...')")
    ap.add_argument("--root-decomp-size", type=int, default=None)
    ap.add_argument("--exact-posterior", dest="exact_posterior",
                    action="store_true", default=True)
    ap.add_argument("--no-exact-posterior", dest="exact_posterior",
                    action="store_false",
                    help="use GPyTorch's rank-100 cache instead of the exact "
                         "Cholesky rebuild")
    ap.add_argument("--full-cov-check", dest="full_cov_check",
                    action="store_true", default=True)
    ap.add_argument("--no-full-cov-check", dest="full_cov_check",
                    action="store_false",
                    help="skip the Var[sum phi] = Var[f(BY)-f(RM)] check on the "
                         "real model (one 83 x 83 Shapley covariance)")
    ap.add_argument("--compare-cache", action="store_true", default=True)
    ap.add_argument("--n-backgrounds", type=int, default=2000,
                    help="segregant targets for panel (e); 0 = skip, -1 = all")
    ap.add_argument("--e2-long-csv", type=Path, default=None,
                    help="reuse E2_phi_and_h_long.csv from the main run for "
                         "panel (e) instead of recomputing (covers the loci "
                         "that file kept)")
    ap.add_argument("--shap-batch-size", type=int, default=4)
    ap.add_argument("--timing-csv", type=Path, default=None,
                    help="E1_timing.csv from the main run, for panel (a)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="loci shown in panel (e) (largest |phi|)")
    ap.add_argument("--selftest-L", type=int, default=10)
    ap.add_argument("--selftest-n-train", type=int, default=64)
    return ap


def main() -> int:
    a = build_parser().parse_args()
    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = epa.REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a.no_log:
        import atexit
        tag = ("_plot" if stages == {"plot"} else
               "_parental" if "parental" in stages and "compute" not in stages
               else "")
        tees = []
        for name, fname in (("stdout", f"run{tag}.log"),
                            ("stderr", f"run{tag}.err")):
            t = _Tee(getattr(sys, name), out_dir / fname)
            setattr(sys, name, t)
            tees.append((name, t))

        def _restore():
            for nm, tee in tees:
                setattr(sys, nm, tee.stream)
                tee.close()
        atexit.register(_restore)

    banner(f"SHAP_ENA1 -- stages {sorted(stages)}")
    print(f"  script   {Path(__file__).resolve()}")
    print(f"  revision {SCRIPT_REVISION}")
    print(f"  ENA1_paper_analysis revision {getattr(epa, 'SCRIPT_REVISION', '?')}")
    print(f"  out      {out_dir}")

    if "selftest" in stages:
        if not stage_selftest(a):
            print("\n  self-test FAILED; refusing to run the real model.")
            return 4
    if "compute" in stages:
        if not stage_compute(a, out_dir):
            return 1
    if "parental" in stages:
        if not stage_parental(a, out_dir):
            return 1
    if "plot" in stages:
        stage_plot(a, out_dir)
    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())