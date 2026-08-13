#!/usr/bin/env python
"""
GB1_fitness.py
==============
Reproduce the Fig. 5 result of Zhou & McCandlish 2020 ("Minimum epistasis
interpolation for sequence-function relationships", Nat. Commun. 11:1782) on the
GB1 landscape, and use our GP+SHAP tool (analysis.py) to give it a QUANTITATIVE
reading.

The paper's claim, in three parts:
    (1) The GB1 landscape decomposes into THREE high-binding regions that are far
        apart in evolutionary time.
    (2) A single GLOBAL additive model fits poorly (R^2 ~ 0.49), but a SEPARATE
        additive model fit WITHIN each region fits well (R^2 ~ 0.77-0.80) --
        i.e. the landscape is locally, not globally, additive.
    (3) The regions differ mainly in the identity/placement of a small residue at
        sites 41 and 54 (positions 2 and 3 of VDGV), which physically contact.

Why the pipeline is ordered the way it is (READ THIS)
-----------------------------------------------------
The regions are defined by the paper's own object -- the diffusion embedding of a
weak-mutation evolutionary Markov chain on the landscape (McCandlish 2011), and its
metastable sets. That construction never looks at any per-site attribution. Only
AFTER the regions exist do we bring in SHAP, to test whether sites 41/54 fall out
as the differentiators.

This ordering is the whole point. If we instead clustered sequences on their SHAP
vectors and then "found" that SHAP differs between clusters, we would have proven
nothing -- the clusters would have been defined by that variation. Defining
regions independently (diffusion) and letting SHAP recover 41/54 makes the 41/54
result a genuine prediction the tool passes.

Region-definition methods (``--cluster-method``)
------------------------------------------------
    * 'pcca'   (default) -- PCCA+ metastable-set decomposition of the evolutionary
                 Markov chain (Deuflhard & Weber inner-simplex spectral PCCA+).
                 This is the most faithful automatic analogue of how the paper's
                 regions arise: they ARE the metastable basins of the chain. Runs
                 on the full 160k landscape because it only needs the sparse
                 slow-mode eigenvectors (the library PCCA+ implementations require
                 a dense transition matrix and do not scale here).
    * 'kmeans'          -- KMeans on the 2-D diffusion coordinates of the
                 high-binding sequences. Simpler; approximates the hand-drawn boxes
                 in the paper by the spatial separation of the arms.

Both define regions from the landscape GEOMETRY, never from SHAP.

Pipeline
--------
    1. Fit / load the GP on GB1 (via analysis.SequenceFunctionAnalysis).
    2. Predict fitness for every sequence (the GP posterior mean = the paper's
       smoothed landscape analogue).
    3. Diffusion spectrum (Fig. 5 axes + metastable eigenvectors) from that fitness.
    4. Define regions (pcca or kmeans) -- geometry only, no SHAP.
    5. Fidelity: top-N binder capture + above-WT capture vs the paper's ~97.5%.
    6. Additivity: within-region vs global additive R^2 + randomization test.
    7. Sequence logos per region (expect sites 2,3 to carry the differences).
    8. SHAP explanation: dividends for the region-assigned sequences; per-region
       main effects per site and the 41x54 (2x3) interaction.
    9. Plots + a text summary lining each number up against the paper.

Paths
-----
Runs correctly from the project root (Kernel_trick/) OR from inside scripts/.
--data defaults to <project_root>/data/gb1.csv; relative --model-dir / --out-dir
are resolved against the project root so a run from either place behaves the same.

Requires scikit-learn (KMeans, LinearRegression) and scipy (sparse eigen-solve).

Usage
-----
    # simplest: everything defaults to the bundled GB1 data + output/GB1_fitness_analysis/
    python scripts/GB1_fitness.py --kernel jenga --gpu \
        --n-condition 20000 --load-model --model-name gb1_jenga_nc20000

    # match the old smoothed run (additive fit to the GP surface) with kmeans regions:
    python scripts/GB1_fitness.py --kernel jenga --gpu \
        --n-condition 20000 --load-model --model-name gb1_jenga_nc20000 \
        --additive-target predicted --cluster-method kmeans \
        --out-dir output/real_nc20k_smoothed

    # quick trial on a subsample before the full 160k run:
    python scripts/GB1_fitness.py --kernel jenga --n-condition 2000 --max-seqs 20000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import (
    SequenceFunctionAnalysis,
    _enumerate_interactions, _to_np, add_common_args, build_config_from_args,
)


# --------------------------------------------------------------------------- #
# GB1 constants from Zhou & McCandlish 2020, Fig. 5
# --------------------------------------------------------------------------- #
GB1_SITES = ("39", "40", "41", "54")     # the four mutated positions
GB1_WT = "VDGV"
# the two sites the paper singles out (41, 54) map to positions 2, 3
KEY_POSITIONS = (2, 3)

# landmark sequences the paper labels in Fig. 5 (overlaid on the scatter if present)
PAPER_LANDMARKS = {
    "VDGV": "WT (region 1)",
    "VWGM": "region 1",
    "LYGV": "region 1",
    "WWLG": "region 2",
    "INAA": "region 3",
    "WRCA": "region 3",
    "LICA": "region 3",
    "RCSE": "low-binding core",
}


# --------------------------------------------------------------------------- #
# path resolution (run from project root OR from scripts/)
# --------------------------------------------------------------------------- #
def find_project_root(explicit: Optional[str] = None) -> Path:
    """Locate the Kernel_trick project root (the dir that contains ``modules/``).

    Works whether the script is launched from the project root or from inside
    scripts/. Walks up from this file until it finds a directory holding
    ``modules/``; falls back sensibly if the layout is unusual. Does not import
    anything heavy, so it is safe to call before building the analysis config.
    """
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "modules").is_dir():
            return cand
    return here.parent if here.name == "scripts" else here


def _resolve_data_path(data_arg: Optional[str], root: Path) -> str:
    """Default --data to <root>/data/gb1.csv; resolve a relative arg sensibly."""
    if data_arg is None:
        p = root / "data" / "gb1.csv"
    else:
        p = Path(data_arg)
        if not p.is_absolute():
            cwd_rel = Path.cwd() / p
            p = cwd_rel if cwd_rel.exists() else (root / p)
    if not p.exists():
        raise FileNotFoundError(
            f"GB1 data not found at {p}. Pass --data explicitly, or place the CSV "
            f"at {root / 'data' / 'gb1.csv'}.")
    return str(p.resolve())


def _resolve_dir(d: str, root: Path) -> str:
    """Resolve a relative directory (out-dir / model-dir) against the project root."""
    p = Path(d)
    return str(p if p.is_absolute() else (root / p))


@dataclass
class GB1Config:
    """GB1-specific knobs (the GP/SHAP knobs live in the AnalysisConfig)."""
    n_regions: int = 3                     # the paper found 3
    cluster_method: str = "pcca"           # 'pcca' | 'kmeans' : region definition
    max_seqs: Optional[int] = None         # None => all sequences (the full graph)
    shap_batch: int = 2000                 # progress granularity for the SHAP step
    high_binding: str = "pred_wt"          # 'pred_wt' | 'wt' | 'qX' : high-binding selector
    diffusion_beta: float = 1.0            # population-scaled selection ("temperature")
    additive_target: str = "measured"      # 'measured' (raw y, as the paper) | 'predicted'
    top_n: int = 100                       # top-N binder capture reported in the fidelity block
    n_perm: int = 200                      # randomization-test subsets (paper used 1000)
    signature_covariance: bool = False     # also compute per-sequence posterior sd of the
                                           # region SHAP signature (adds a model-confidence
                                           # column; SEM error bars are always shown)


# =========================================================================== #
# The reproduction
# =========================================================================== #

class GB1FitnessLandscape:
    """Diffusion/metastable regions (paper geometry) + SHAP explanation (our tool)."""

    def __init__(self, sfa: SequenceFunctionAnalysis, gb1: GB1Config):
        self.sfa = sfa
        self.gb1 = gb1
        self.cfg = sfa.cfg
        self.L = sfa.L
        self.A = sfa.A
        self.alphabet = sfa.alphabet
        self.out_dir = Path(sfa.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._require_deps()
        # diffusion spectrum, populated by _diffusion_spectrum()
        self._lam: Optional[np.ndarray] = None      # eigenvalues, closest-to-0 first
        self._psi: Optional[np.ndarray] = None      # right eigenvectors of Q (N x nev)
        self._sqrt_pi: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        seqs, y = self._select_sequences()
        pred = self._predicted_fitness(seqs)
        wt_pred, wt_meas = self._wt_reference(seqs, y, pred)

        # --- regions: the paper's geometry, no SHAP involved ---
        self._diffusion_spectrum(seqs, pred)             # sets self._lam/_psi/_sqrt_pi
        emb, slow_eig = self._embedding_from_spectrum()
        high = self._high_binding_mask(y, pred, wt_pred, wt_meas)
        labels = self._define_regions(emb, high)
        n_found = len(set(labels[labels >= 0].tolist()))
        print(f"[gb1] {int(high.sum())}/{len(seqs)} high-binding; "
              f"{n_found} regions defined via {self.gb1.cluster_method} in diffusion space.")

        # --- checks against the paper ---
        lines: List[str] = []
        lines += self._fidelity(seqs, y, labels, high, wt_meas)
        lines += self._additivity(seqs, y, pred, labels)
        lines += self._logos(seqs, labels)

        # --- SHAP explanation, computed ONLY on region-assigned sequences ---
        feats, feats_sd, singleton_cols, pair_col_2x3 = self._shap_on_regions(seqs, labels)
        lines += self._shap_signature(feats, singleton_cols, pair_col_2x3, labels,
                                      feats_sd=feats_sd)

        # --- outputs ---
        self._plot_scatter(seqs, emb, pred, labels, high)
        self._dump_embedding_csv(seqs, emb, pred, y, labels, high)
        summ = self.out_dir / "GB1_SUMMARY.txt"
        with open(summ, "w") as fh:
            fh.write(self._render_summary(lines, slow_eig))
        print(f"[gb1] wrote reproduction artifacts to {self.out_dir}")
        return {"embedding": emb, "labels": labels, "high_binding": high}

    # ------------------------------------------------------------------ #
    # data selection + GP predictions
    # ------------------------------------------------------------------ #
    def _select_sequences(self):
        seqs = np.asarray([str(s) for s in self.sfa.seqs])
        y = np.asarray(self.sfa.y, dtype=float)
        cap = self.gb1.max_seqs
        if cap is not None and 0 < cap < len(seqs):
            rng = np.random.default_rng(self.cfg.seed)
            idx = set(rng.choice(len(seqs), size=cap, replace=False).tolist())
            idx |= {i for i, s in enumerate(seqs) if s in PAPER_LANDMARKS}
            idx = np.array(sorted(idx))
            seqs, y = seqs[idx], y[idx]
            print(f"[gb1] subsampled to {len(seqs)} sequences (--max-seqs={cap}); "
                  "the diffusion geometry is only faithful on the full 160k landscape.")
        else:
            print(f"[gb1] using ALL {len(seqs)} sequences.")
        return seqs, y

    def _predicted_fitness(self, seqs):
        X = self.sfa._encode_seqs(list(seqs), alphabet=self.alphabet)
        return self.sfa._predict_batched(self.sfa.model, X)

    def _wt_reference(self, seqs, y, pred):
        wt = self.cfg.wildtype or GB1_WT
        hits = np.where(seqs == wt)[0]
        if len(hits):
            i = hits[0]
            return float(pred[i]), float(y[i])
        wt_meas = 0.0 if (y.min() < 0 < y.max()) else float(np.median(y))
        print(f"[gb1][warn] WT '{wt}' not in the selected set; using {wt_meas:.3f} as ref.")
        return wt_meas, wt_meas

    def _high_binding_mask(self, y, pred, wt_pred, wt_meas):
        mode = self.gb1.high_binding
        if mode == "pred_wt":
            return pred > wt_pred
        if isinstance(mode, str) and mode.startswith("q"):
            return y >= np.quantile(y, float(mode[1:]))
        return y > wt_meas  # 'wt'

    # ------------------------------------------------------------------ #
    # diffusion spectrum  (McCandlish 2011; the paper's Fig. 5 axes)
    # ------------------------------------------------------------------ #
    def _diffusion_spectrum(self, seqs, pred) -> None:
        """Eigen-decompose a weak-mutation evolutionary rate matrix on the Hamming
        graph, using GP-predicted fitness as the (smoothed) phenotype.

        Origin-fixation / Sella-Hirsh model: the substitution process is reversible
        with stationary distribution pi_i ~ exp(beta * f_i), rate between adjacent
        genotypes Q_ij = phi(S_ij) with S_ij = beta*(f_j - f_i), phi(x)=x/(1-e^-x).

        Stores, sorted closest-to-0 first (i.e. the DOMINANT / slowest modes):
            self._lam    generator eigenvalues (<= 0); lam[0] ~ 0 is stationary
            self._psi    right eigenvectors of Q (N x nev); psi[:,0] ~ constant
            self._sqrt_pi

        The 2 slowest NONZERO modes give the Fig. 5 plot axes; the first
        ``n_regions`` columns (incl. the constant Perron mode) feed PCCA+.

        Solved matrix-free (Lanczos on the sparse symmetrized operator), so the full
        ~160k graph fits in a few hundred MB.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        N = len(seqs)
        ri, ci = self._hamming1_edges(seqs)
        if len(ri) == 0:
            raise RuntimeError("no Hamming-1 edges among the selected sequences; "
                               "the diffusion geometry needs a connected landscape "
                               "(run on the full GB1 set, not a sparse subsample).")
        print(f"[gb1] diffusion graph: {N} nodes, {len(ri)} undirected edges")

        beta = float(self.gb1.diffusion_beta)
        f = np.asarray(pred, dtype=float)
        logpi = beta * (f - f.max())
        logpi -= np.log(np.sum(np.exp(logpi)))       # normalize
        pi = np.maximum(np.exp(logpi), 1e-300)       # floor: avoid inf in sqrt(pi_i/pi_j)
        sqrt_pi = np.sqrt(pi)

        def phi(x):                                   # x/(1-e^-x), stable, ->1 at 0
            out = np.ones_like(x)
            m = np.abs(x) > 1e-8
            out[m] = x[m] / (1.0 - np.exp(-x[m]))
            return out

        S = beta * (f[ci] - f[ri])
        w_ij = phi(S)                                 # rate i->j
        w_ji = phi(-S)                                # rate j->i
        # symmetric off-diagonals of Qsym = D^{1/2} Q D^{-1/2}
        s_ij = (sqrt_pi[ri] / sqrt_pi[ci]) * w_ij
        s_ji = (sqrt_pi[ci] / sqrt_pi[ri]) * w_ji
        rows = np.concatenate([ri, ci])
        cols = np.concatenate([ci, ri])
        vals = np.concatenate([s_ij, s_ji])
        Ssym = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        Ssym = 0.5 * (Ssym + Ssym.T)                  # enforce numerical symmetry

        # CORRECT generator diagonal = -(original out-rates) = -sum_j w_ij.
        # NOT the symmetrized row sums (symmetrization changes row sums).
        dout = np.zeros(N)
        np.add.at(dout, ri, w_ij)                     # i -> j out-rate accrues to i
        np.add.at(dout, ci, w_ji)                     # j -> i out-rate accrues to j

        # Matrix-free dominant-mode solve: largest eigenvalues of P = I + Qsym/c
        # (Qsym = Ssym - diag(dout), c = max out-rate). sqrt(pi) is the eigenvector
        # with eigenvalue 1 (stationary); the next-largest are the slow modes.
        n_modes = max(2, int(self.gb1.n_regions))     # enough for the plot AND PCCA+
        c = float(dout.max()) if dout.max() > 0 else 1.0
        Psym = (Ssym / c) + sp.diags(1.0 - dout / c)
        nev = min(N - 1, n_modes + 5)
        ncv = min(N - 1, max(2 * nev + 1, 20))
        muP, vec = eigsh(Psym, k=nev, which="LA", ncv=ncv, maxiter=5000)
        lam = c * (muP - 1.0)                          # generator eigenvalues (<= 0)
        order = np.argsort(-lam)                       # closest to 0 first (dominant)
        lam, vec = lam[order], vec[:, order]
        psi = vec / sqrt_pi[:, None]                   # right eigenvectors of Q

        self._lam, self._psi, self._sqrt_pi = lam, psi, sqrt_pi

    def _embedding_from_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """The 2-D Fig. 5 plot coordinates: the 2 slowest NONZERO modes, each scaled
        by 1/sqrt(-lambda_k) (diffusion-map scaling). Returns (emb, slow_eigenvalues).
        """
        lam, psi = self._lam, self._psi
        slow = np.where(lam < -1e-9)[0][:2]            # drop stationary/disconnected (~0)
        if len(slow) < 2:
            raise RuntimeError(f"graph too disconnected: only {len(slow)} slow modes; "
                               "run on the full landscape.")
        emb = np.zeros((psi.shape[0], 2))
        for axis, gi in enumerate(slow):
            emb[:, axis] = psi[:, gi] / np.sqrt(-lam[gi])
        slow_eig = lam[slow]
        print(f"[gb1] diffusion slow-mode eigenvalues: "
              f"{', '.join(f'{v:.2e}' for v in slow_eig)}")
        self._check_beta(slow_eig)
        return emb, slow_eig

    def _hamming1_edges(self, seqs):
        """Undirected Hamming-distance-1 edges present in the sequence set.

        Vectorized via a dense key->index table when the code space A**L is small
        enough to index directly (true for GB1: 20**4 = 160000). Falls back to a
        dict for larger spaces.
        """
        amap = {c: i for i, c in enumerate(self.alphabet)}
        L, A = self.L, self.A
        codes = np.array([[amap[c] for c in s] for s in seqs], dtype=np.int64)
        powers = (A ** np.arange(L)).astype(np.int64)
        keys = codes @ powers
        N = len(seqs)

        space = int(A) ** int(L)
        dense = space <= 50_000_000                    # ~400 MB int64 ceiling
        if dense:
            idx_of_key = np.full(space, -1, dtype=np.int64)
            idx_of_key[keys] = np.arange(N)

            def lookup(nk):
                return idx_of_key[nk]
        else:
            key_to_idx = {int(k): i for i, k in enumerate(keys)}

            def lookup(nk):
                return np.fromiter((key_to_idx.get(int(k), -1) for k in nk),
                                   dtype=np.int64, count=len(nk))

        ri_parts, ci_parts = [], []
        for p in range(L):
            cur = codes[:, p]
            base = keys - cur * powers[p]
            for a in range(A):
                mask = cur != a
                if not mask.any():
                    continue
                i = np.nonzero(mask)[0]
                nk = base[i] + a * powers[p]
                j = lookup(nk)
                valid = j >= 0
                i, j = i[valid], j[valid]
                keep = j > i                            # dedup undirected edges
                if keep.any():
                    ri_parts.append(i[keep]); ci_parts.append(j[keep])
        if not ri_parts:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(ri_parts), np.concatenate(ci_parts)

    def _check_beta(self, eig):
        """Warn (loudly) if the diffusion spectrum indicates beta is mistuned."""
        RED, BOLD, RESET = "\033[91m", "\033[1m", "\033[0m"
        beta = float(self.gb1.diffusion_beta)
        mags = np.abs(eig)
        m1 = float(mags[0]); m2 = float(mags[1]) if len(mags) > 1 else m1
        lo, hi = min(m1, m2), max(m1, m2)
        ratio = lo / hi if hi > 0 else 1.0
        collapsed = m1 < 1e-7                            # slow mode fell into the ~0 mode
        degenerate = ratio > 0.98                        # the two arms don't separate
        if collapsed or degenerate:
            what = ("OVER-ISOLATED PEAKS (beta too high)" if collapsed
                    else "TWO AXES NEARLY DEGENERATE")
            bar = "!" * 72
            msg = "\n".join(f"{RED}{BOLD}{ln}{RESET}" for ln in [
                f"WARNING: DIFFUSION SPECTRUM LOOKS OFF -- {what}.",
                f"SLOW-MODE EIGENVALUES = {m1:.2e}, {m2:.2e} (separation {ratio:.2f}).",
                f"==> RETUNE --diffusion-beta (try {beta/2:.3g} or {beta*2:.3g}) "
                "AND COMPARE THE SCATTER PLOTS.",
            ])
            print(f"\n{RED}{BOLD}{bar}{RESET}\n{msg}\n{RED}{BOLD}{bar}{RESET}\n", flush=True)
        else:
            print(f"[gb1] diffusion beta={beta:g} looks reasonable "
                  f"(slow modes {m1:.2e}, {m2:.2e}; separation {ratio:.2f}).")

    # ------------------------------------------------------------------ #
    # regions: dispatch on cluster_method  (both use landscape geometry only)
    # ------------------------------------------------------------------ #
    def _define_regions(self, emb, high):
        if self.gb1.cluster_method == "pcca":
            return self._regions_pcca(high)
        return self._regions_kmeans(emb, high)

    def _regions_kmeans(self, emb, high):
        """KMeans on the 2-D diffusion coordinates of the high-binding sequences."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        k = self.gb1.n_regions
        labels = np.full(emb.shape[0], -1, dtype=int)   # -1 = low-binding core
        Z = StandardScaler().fit_transform(emb[high])
        if Z.shape[0] < k:
            print(f"[gb1][warn] only {Z.shape[0]} high-binding seqs (< k={k}); "
                  "skipping region assignment.")
            return labels
        km = KMeans(n_clusters=k, n_init=10, random_state=self.cfg.seed).fit(Z)
        labels[high] = km.labels_
        self._report_silhouette(Z, km.labels_)
        return labels

    def _regions_pcca(self, high):
        """PCCA+ metastable sets of the evolutionary chain, restricted to the
        high-binding arms. Regions are the metastable basins -- the most faithful
        automatic analogue of the paper's construction.
        """
        k = self.gb1.n_regions
        N = self._psi.shape[0]
        labels = np.full(N, -1, dtype=int)
        if self._psi.shape[1] < k:
            print(f"[gb1][warn] only {self._psi.shape[1]} eigenvectors available "
                  f"(< k={k}); cannot run PCCA+. Falling back to kmeans is advised.")
            return labels
        chi = self._pcca_plus(k)                        # N x k memberships
        assign = np.argmax(chi, axis=1)
        labels[high] = assign[high]
        self._report_crispness(chi, high)
        present = sorted(set(labels[labels >= 0].tolist()))
        if len(present) < k:
            print(f"[gb1][warn] PCCA+ produced {len(present)} non-empty regions among "
                  f"the high-binding set (asked for {k}). Try adjusting --diffusion-beta, "
                  "or use --cluster-method kmeans.")
        return labels

    def _pcca_plus(self, m):
        """Spectral PCCA+ (Deuflhard & Weber inner-simplex algorithm).

        Uses the m dominant right eigenvectors of the Markov chain (the constant
        Perron mode + the m-1 slowest), stored in self._psi. Returns an N x m
        membership matrix whose rows sum to 1; argmax gives the crisp metastable
        assignment. This is the eigenvector-based core of PCCA+; we skip the
        optional metastability-optimization refinement (opt_soft), which mainly
        sharpens SOFT memberships and rarely changes the argmax hard assignment.

        We compute directly from the sparse slow-mode eigenvectors rather than a
        dense transition matrix, so this scales to the full 160k-genotype chain.
        """
        X = self._psi[:, :m].copy()
        X[:, 0] = 1.0                                   # enforce exact constant Perron mode
        n, mm = X.shape

        # Inner Simplex Algorithm: locate m vertices of the simplex spanned by X.
        index = np.zeros(mm, dtype=int)
        index[0] = int(np.argmax((X ** 2).sum(axis=1)))
        Ortho = X - X[index[0]]                         # translate first vertex to origin
        for j in range(1, mm):
            norms = (Ortho ** 2).sum(axis=1)
            index[j] = int(np.argmax(norms))
            v = Ortho[index[j]].copy()
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                break
            v = v / nv
            Ortho = Ortho - np.outer(Ortho.dot(v), v)   # project this direction out

        # Membership transform: chi = X @ inv(X[vertices]).  X[:,0]==1 => rows of chi
        # sum to 1 by construction; clip tiny negatives and renormalize.
        try:
            Ainv = np.linalg.inv(X[index, :])
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(X[index, :])
        chi = X.dot(Ainv)
        chi = np.clip(chi, 0.0, None)
        rs = chi.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return chi / rs

    def _report_crispness(self, chi, high):
        maxmemb = chi.max(axis=1)
        val = float(maxmemb[high].mean()) if high.any() else float("nan")
        print(f"[gb1] PCCA+ mean max-membership (high-binding) = {val:.3f} "
              "(1.0 = perfectly crisp metastable sets)")

    def _report_silhouette(self, Z, labels):
        from sklearn.metrics import silhouette_score
        n = Z.shape[0]
        if n > 3000:
            rng = np.random.default_rng(self.cfg.seed)
            s = rng.choice(n, size=3000, replace=False)
            Z, labels = Z[s], labels[s]
        try:
            sc = silhouette_score(Z, labels)
            print(f"[gb1] region silhouette (k={self.gb1.n_regions}) = {sc:.3f} "
                  "(higher = better-separated arms)")
        except Exception as e:  # noqa: BLE001
            print(f"[gb1] silhouette unavailable: {e}")

    # ------------------------------------------------------------------ #
    # check (5): landscape fidelity vs the paper
    # ------------------------------------------------------------------ #
    def _fidelity(self, seqs, y, labels, high, wt_meas):
        n_top = int(self.gb1.top_n)
        in_region = labels >= 0
        above_wt = y > wt_meas

        top_idx = np.argsort(-y)[:min(n_top, len(y))]
        top_captured = int(np.sum(in_region[top_idx]))
        n_above = int(np.sum(above_wt))
        captured_above = int(np.sum(in_region & above_wt))

        lines = [
            "[5] Landscape fidelity vs paper "
            "(paper: regions hold ~97.5% of above-WT seqs and all top-100 binders)",
            f"    above-WT sequences:           {n_above}",
            f"    ... assigned to a region:     {captured_above}  "
            f"({captured_above / max(1, n_above):.1%})",
            f"    top-{n_top} measured binders in a region: {top_captured}/{len(top_idx)}  "
            f"({top_captured / max(1, len(top_idx)):.1%})",
        ]
        for c in sorted(set(labels[labels >= 0].tolist())):
            share = int(np.sum(labels[top_idx] == c))
            lines.append(f"      region {c}: {share} of top {n_top}")
        return lines + [""]

    # ------------------------------------------------------------------ #
    # check (2): local vs global additivity + randomization test
    # ------------------------------------------------------------------ #
    def _additivity(self, seqs, y, pred, labels):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        target = self.gb1.additive_target
        yt = pred if target == "predicted" else y     # paper fit raw ('measured')

        def add_fit_indices(indices):
            sub_seqs = seqs[indices]
            sub_t = yt[indices]
            n = len(sub_t)
            if n < 5:
                return float("nan"), n, 0
            X = self._one_hot(sub_seqs)
            keep = X.std(axis=0) > 0
            Xk = X[:, keep]
            if Xk.shape[1] == 0:
                return float("nan"), n, 0
            lr = LinearRegression().fit(Xk, sub_t)
            return float(r2_score(sub_t, lr.predict(Xk))), n, int(Xk.shape[1])

        all_idx = np.arange(len(seqs))

        def perm_p(observed_r2, region_n):
            if not np.isfinite(observed_r2) or region_n < 5:
                return float("nan")
            rng = np.random.default_rng(self.cfg.seed)
            ge = 0
            for _ in range(self.gb1.n_perm):
                s = rng.choice(len(seqs), size=region_n, replace=False)
                r2s, _, _ = add_fit_indices(s)
                if np.isfinite(r2s) and r2s >= observed_r2:
                    ge += 1
            return (ge + 1) / (self.gb1.n_perm + 1)     # add-one smoothing

        lines = [f"[2] Local vs global additivity  (target={target}; "
                 "paper: 0.77-0.80 within region, 0.49 global)"]
        g_r2, g_n, g_par = add_fit_indices(all_idx)
        lines.append(f"    global additive R^2 = {g_r2:.3f}  (n={g_n}, params={g_par})")
        rows = [{"region": "global", "n": g_n, "n_params": g_par,
                 "additive_r2": g_r2, "perm_p": float("nan")}]
        for c in sorted(set(labels[labels >= 0].tolist())):
            m = np.nonzero(labels == c)[0]
            r2c, nc, parc = add_fit_indices(m)
            p = perm_p(r2c, nc)
            flag = "  [optimistic: n < 3*params]" if (parc and nc < 3 * parc) else ""
            lines.append(f"    region {c}: additive R^2 = {r2c:.3f}  "
                         f"(n={nc}, params={parc}, perm p={p:.3g}){flag}")
            rows.append({"region": c, "n": nc, "n_params": parc,
                         "additive_r2": r2c, "perm_p": p})
        pd.DataFrame(rows).to_csv(self.out_dir / "gb1_additive_fits.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        reg = [r for r in rows if r["region"] != "global"]
        ax.bar([f"region {r['region']}" for r in reg],
               [r["additive_r2"] for r in reg], color="#9bbb59")
        ax.axhline(g_r2, color="#c0504d", ls="--", label=f"global = {g_r2:.2f}")
        ax.axhspan(0.77, 0.80, color="#4f81bd", alpha=0.2, label="paper 0.77-0.80")
        ax.set_ylabel(f"additive R^2 (in-sample, target={target})")
        ax.set_xlabel("region")
        ax.set_title("Local additivity per region vs global")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out_dir / "gb1_additive_fits.png", dpi=130)
        plt.close(fig)
        lines.append(f"    (randomization test: {self.gb1.n_perm} subsets; "
                     "paper used 1000, reported p < 0.003)")
        return lines + [""]

    # ------------------------------------------------------------------ #
    # check (3a): per-region residue composition (sequence logos)
    # ------------------------------------------------------------------ #
    def _logos(self, seqs, labels):
        clusters = sorted(set(labels[labels >= 0].tolist()))
        idx = {c: i for i, c in enumerate(self.alphabet)}
        lines = ["[3a] Per-region residue composition  "
                 "(paper: sites 41,54 = positions 2,3 carry the differences)"]
        if not clusters:
            return lines + ["    (no regions)", ""]

        # precompute per-region frequency matrices (A x L)
        freqs = {}
        for c in clusters:
            m = labels == c
            fr = np.zeros((self.A, self.L))
            for s in seqs[m]:
                for p, ch in enumerate(s):
                    fr[idx[ch], p] += 1
            freqs[c] = (fr / max(1, m.sum()), int(m.sum()))

        # two rows: heatmaps (top) + paper-style stacked-letter logos (bottom).
        # Extra width on the right is reserved for a dedicated colorbar axis so the
        # bar sits to the RIGHT of all panels instead of stealing space from the
        # last one.
        ncol = len(clusters)
        fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol + 0.9, 7.2),
                                 squeeze=False,
                                 gridspec_kw={"height_ratios": [3, 2]})
        im = None
        for j, c in enumerate(clusters):
            fr, n = freqs[c]
            # ---- top: heatmap ----
            ax = axes[0][j]
            im = ax.imshow(fr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(self.L))
            ax.set_xticklabels([GB1_SITES[p] if p < len(GB1_SITES) else f"p{p}"
                                for p in range(self.L)], fontsize=8)
            ax.set_yticks(range(self.A)); ax.set_yticklabels(list(self.alphabet), fontsize=6)
            ax.set_xlabel("site")
            if j == 0:
                ax.set_ylabel("residue")
            modal = "".join(self.alphabet[int(np.argmax(fr[:, p]))] for p in range(self.L))
            ax.set_title(f"region {c}  (n={n})\nmodal={modal}", fontsize=9)
            for p in KEY_POSITIONS:
                if p < self.L:
                    ax.add_patch(plt.Rectangle((p - 0.5, -0.5), 1, self.A,
                                               fill=False, edgecolor="red", lw=1.2))
            # ---- bottom: stacked-letter sequence logo (letter height ~ frequency) ----
            self._draw_letter_logo(axes[1][j], fr)
            if j == 0:
                axes[1][j].set_ylabel("frequency")

            top = [(p, self.alphabet[int(np.argmax(fr[:, p]))], float(fr[:, p].max()))
                   for p in KEY_POSITIONS if p < self.L]
            lines.append(f"    region {c}: modal={modal}  "
                         + "  ".join(f"site{GB1_SITES[p]}={r}({f:.0%})" for p, r, f in top))

        fig.suptitle("Per-region residue composition (red = sites 41, 54)", fontsize=13)
        # Lay out ONLY the panel region (leave a right margin for the colorbar and
        # top headroom for the suptitle); then place the colorbar in that margin.
        fig.tight_layout(rect=[0, 0, 0.90, 0.94])
        cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.6])  # [left, bottom, w, h]
        fig.colorbar(im, cax=cbar_ax, label="residue frequency")
        fig.savefig(self.out_dir / "gb1_cluster_logos.png", dpi=130)
        plt.close(fig)
        lines.append("    (paper: region1 -> G at 41; region2 -> G at 54, L/F at 41; "
                     "region3 -> A at 54, C/A at 41)")
        return lines + [""]

    def _draw_letter_logo(self, ax, freq):
        """Paper-style sequence logo: at each site, stack residue letters with height
        proportional to their frequency (tallest on top), matching Fig. 5's logos.

        ``freq`` is an (A, L) column-stochastic matrix. Letters below a small floor
        are skipped so the panel stays legible; this is frequency-scaled (not
        information/bits), to match the paper's visual convention.
        """
        floor = 0.03
        for p in range(self.L):
            order = np.argsort(freq[:, p])          # ascending -> stack bottom to top
            y = 0.0
            for a in order:
                h = float(freq[a, p])
                if h < floor:
                    continue
                ch = self.alphabet[a]
                # size the glyph to the bar height via a scaled text object
                ax.text(p, y + h / 2.0, ch, ha="center", va="center",
                        fontsize=6 + 34 * h, fontweight="bold",
                        color=("crimson" if p in KEY_POSITIONS else "0.25"))
                y += h
        ax.set_xlim(-0.5, self.L - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(self.L))
        ax.set_xticklabels([GB1_SITES[p] if p < len(GB1_SITES) else f"p{p}"
                            for p in range(self.L)], fontsize=8)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("site")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # ------------------------------------------------------------------ #
    # SHAP explanation of the (independently-defined) regions
    # ------------------------------------------------------------------ #
    def _shap_on_regions(self, seqs, labels):
        """Dividend vectors for region-assigned sequences only.

        SHAP is computed AFTER the regions exist, so it explains rather than
        defines them. Returns (feats, feats_sd, singleton_cols, pair_col_2x3) where
        feats rows align with ``seqs`` (non-region rows left as NaN and never read).
        ``feats_sd`` holds the per-sequence GP posterior sd of each dividend when
        ``--signature-covariance`` is set, else None.
        """
        shap, pi = self.sfa.shap, self.sfa.pi
        positions = [(i,) for i in range(self.L)]
        inter = _enumerate_interactions(self.L, self.cfg.interaction_order)
        tuples = positions + inter
        singleton_cols = list(range(self.L))
        pair_col_2x3 = tuples.index(KEY_POSITIONS) if KEY_POSITIONS in tuples else None
        if pair_col_2x3 is None:
            print("[gb1][note] the (2,3)=41x54 interaction is not enumerated; "
                  "raise --interaction-order to >=2 to report it.")

        want_sd = self.gb1.signature_covariance
        target_idx = np.nonzero(labels >= 0)[0]
        feats = np.full((len(seqs), len(tuples)), np.nan, dtype=float)
        feats_sd = np.full((len(seqs), len(tuples)), np.nan, dtype=float) if want_sd else None
        N = len(target_idx)
        batch = max(1, self.gb1.shap_batch)
        t0 = time.time()
        for start in range(0, N, batch):
            for ii in target_idx[start:start + batch]:
                if want_sd:
                    # covariance="diag" returns the per-tuple posterior VARIANCES as a
                    # 1-D vector, which is all we store (as sd). The full path built the
                    # n x n grid and then kept only torch.diagonal(cov) -- O(n^2) work
                    # for an O(n) answer, once per region-assigned sequence. Numerically
                    # identical to the old sqrt(diagonal(full cov)) to ~1e-16
                    # (probe_R_diag.py), so the CSV columns are unchanged.
                    _, dv, dvar = shap.dividend_values(
                        str(seqs[ii]), pi, interactions=tuples,
                        calc_covar=True, covariance="diag")
                    feats[ii] = _to_np(dv)
                    feats_sd[ii] = _to_np(torch.sqrt(torch.clamp(dvar, min=0.0)))
                else:
                    _, dv = shap.dividend_values(str(seqs[ii]), pi, interactions=tuples)
                    feats[ii] = _to_np(dv)
            done = min(start + batch, N)
            rate = done / max(1e-9, time.time() - t0)
            print(f"[gb1]   region SHAP vectors {done}/{N}  "
                  f"({rate:.0f}/s, eta {(N - done) / max(1e-9, rate) / 60:.1f} min)"
                  + ("  [+covariance]" if want_sd else ""),
                  flush=True)
        print(f"[gb1] SHAP feature matrix on {N} region sequences "
              f"({self.L} singletons + {len(inter)} interactions)")
        return feats, feats_sd, singleton_cols, pair_col_2x3

    def _shap_signature(self, feats, singleton_cols, pair_col_2x3, labels, feats_sd=None):
        """check (3b): per-region main-effect signature + the 41x54 interaction.

        The bar is the region-mean main-effect dividend per site, with SEM error
        bars (std across the region's sequences / sqrt(n)). What confirms the
        paper's claim is not bar HEIGHT but between-region SEPARATION: a site whose
        per-region means differ by many combined-SEMs is a differentiating site.

        Because a single max-minus-min "spread" can hide a 2-vs-1 structure (one
        region far from two that sit together), we report the FULL pairwise
        separation z (every region pair, gap / combined SEM) for each site AND for
        the 41x54 interaction, and draw the interaction as its own per-region plot.
        The phrasing of the conclusion is driven by these z's, not assumed.

        ``feats_sd`` (optional, --signature-covariance) adds a model-confidence
        column (mean GP posterior sd); the SEM error bars are always drawn.
        """
        clusters = sorted(set(labels[labels >= 0].tolist()))
        lines = ["[3b] SHAP signature per region "
                 "(paper: additive effects differ 'particularly at sites 41 and 54')"]
        if not clusters:
            return lines + ["    (no regions)", ""]

        sig = feats[:, singleton_cols]
        sig_sd = feats_sd[:, singleton_cols] if feats_sd is not None else None

        def site_name(p):
            return GB1_SITES[p] if p < len(GB1_SITES) else str(p)

        # ---- per-region means + SEMs for main effects and the 41x54 interaction ----
        means_by_region, sems_by_region = {}, {}
        inter_mean, inter_sem = {}, {}
        rows = []
        for c in clusters:
            m = labels == c
            n_c = int(m.sum())
            mean_sig = np.nanmean(sig[m], axis=0)
            sem = np.nanstd(sig[m], axis=0) / max(1.0, np.sqrt(n_c))
            means_by_region[c], sems_by_region[c] = mean_sig, sem
            row = {"region": c}
            row.update({f"main_site{site_name(p)}": mean_sig[p] for p in range(self.L)})
            row.update({f"sem_site{site_name(p)}": sem[p] for p in range(self.L)})
            if sig_sd is not None:
                post = np.nanmean(sig_sd[m], axis=0)
                row.update({f"postsd_site{site_name(p)}": post[p] for p in range(self.L)})
            if pair_col_2x3 is not None:
                vals = feats[m, pair_col_2x3]
                inter_mean[c] = float(np.nanmean(vals))
                inter_sem[c] = float(np.nanstd(vals) / max(1.0, np.sqrt(n_c)))
                row["interaction_41x54"] = inter_mean[c]
                row["interaction_41x54_sem"] = inter_sem[c]
                if feats_sd is not None:
                    row["interaction_41x54_postsd"] = float(
                        np.nanmean(feats_sd[m, pair_col_2x3]))
            rows.append(row)

        def max_sep_z(vals, sems):
            """Largest pairwise |gap| / combined SEM over region pairs -> (z, pair)."""
            best_z, best_pair = 0.0, None
            for a in clusters:
                for b in clusters:
                    if a < b:
                        gap = abs(vals[a] - vals[b])
                        den = float(np.hypot(sems[a], sems[b]))
                        zz = gap / den if den > 0 else float("inf")
                        if zz > best_z:
                            best_z, best_pair = zz, (a, b)
            return best_z, best_pair

        def pairwise_lines(vals, sems, indent="      "):
            out = []
            for a in clusters:
                for b in clusters:
                    if a < b:
                        gap = abs(vals[a] - vals[b])
                        den = float(np.hypot(sems[a], sems[b]))
                        zz = gap / den if den > 0 else float("inf")
                        out.append(f"{indent}{a} vs {b}: gap={gap:.3f}  z={zz:.1f}")
            return out

        # per-site max separation z (drives the x-tick annotation)
        site_maxz = {p: max_sep_z({c: means_by_region[c][p] for c in clusters},
                                  {c: sems_by_region[c][p] for c in clusters})[0]
                     for p in range(self.L)}

        # ---- plot 1: main-effect signature, z annotated under each tick ----
        fig, ax = plt.subplots(figsize=(max(6, self.L * 1.6), 4))
        w = 0.8 / max(1, len(clusters))
        xpos = np.arange(self.L)
        for j, c in enumerate(clusters):
            ax.bar(xpos + j * w, means_by_region[c], width=w,
                   yerr=sems_by_region[c], capsize=3, label=f"region {c}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(xpos + w * (len(clusters) - 1) / 2)
        ax.set_xticklabels([f"{site_name(p)}\n(sep z={site_maxz[p]:.1f})"
                            for p in range(self.L)])
        ax.set_ylabel("mean main-effect dividend")
        ax.set_title("SHAP additive signature per region")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out_dir / "gb1_shap_signature.png", dpi=130)
        plt.close(fig)
        pd.DataFrame(rows).to_csv(self.out_dir / "gb1_shap_signature.csv", index=False)

        # ---- plot 2: per-region 41x54 interaction (the epistatic term) ----
        if pair_col_2x3 is not None:
            iz, ipair = max_sep_z(inter_mean, inter_sem)
            fig2, ax2 = plt.subplots(figsize=(max(5.5, len(clusters) * 1.8), 4.4))
            cmap = plt.get_cmap("tab10")
            ax2.bar([f"region {c}" for c in clusters],
                    [inter_mean[c] for c in clusters],
                    yerr=[inter_sem[c] for c in clusters], capsize=4,
                    color=[cmap(j) for j in range(len(clusters))])
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set_ylabel("mean 41x54 interaction dividend")
            ax2.set_title(f"41x54 epistatic term per region  (max sep z={iz:.1f})",
                          fontsize=10)
            fig2.tight_layout()
            fig2.savefig(self.out_dir / "gb1_interaction_41x54.png", dpi=130)
            plt.close(fig2)

        # ---- summary block ----
        for c in clusters:
            site_txt = "  ".join(f"site{site_name(p)}={means_by_region[c][p]:+.3f}"
                                 for p in range(self.L))
            lines.append(f"    region {c} main effects: {site_txt}")

        if pair_col_2x3 is not None:
            lines.append("    41x54 interaction per region: "
                         + "  ".join(f"r{c}={inter_mean[c]:+.3f}" for c in clusters))

            # Main-effect differentiation, full pairwise (exposes 2-vs-1 structure).
            lines.append("    MAIN-EFFECT between-region separation "
                         "(gap / combined SEM), per site:")
            for p in range(self.L):
                z, pr = max_sep_z({c: means_by_region[c][p] for c in clusters},
                                  {c: sems_by_region[c][p] for c in clusters})
                lines.append(f"    site{site_name(p)}: max z={z:.1f}"
                             + (f" (regions {pr[0]} vs {pr[1]})" if pr else ""))
                for pl in pairwise_lines({c: means_by_region[c][p] for c in clusters},
                                         {c: sems_by_region[c][p] for c in clusters}):
                    lines.append(pl)

            # Interaction differentiation, full pairwise (does 41x54 split 0 vs 1?).
            lines.append("    41x54 INTERACTION between-region separation "
                         "(gap / combined SEM):")
            for pl in pairwise_lines(inter_mean, inter_sem, indent="      "):
                lines.append(pl)

            lines.append("    (read the pairwise z's: a large max-z with one small "
                         "pair = a 2-vs-1 split, not a three-way separation)")
        return lines + [""]

    # ------------------------------------------------------------------ #
    # scatter + dumps
    # ------------------------------------------------------------------ #
    def _plot_scatter(self, seqs, emb, pred, labels, high):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        order = np.argsort(pred)
        sc = axes[0].scatter(emb[order, 0], emb[order, 1], c=pred[order],
                             s=4, cmap="RdYlBu_r", linewidths=0)
        fig.colorbar(sc, ax=axes[0], label="predicted fitness")
        axes[0].set_title("Diffusion embedding, all sequences by fitness")
        axes[0].set_xlabel("Diffusion axis 1"); axes[0].set_ylabel("Diffusion axis 2")

        axes[1].scatter(emb[~high, 0], emb[~high, 1], c="0.85", s=3, linewidths=0,
                        label="low-binding core")
        cmap = plt.get_cmap("tab10")
        for j, c in enumerate(sorted(set(labels[labels >= 0].tolist()))):
            m = labels == c
            axes[1].scatter(emb[m, 0], emb[m, 1], s=6, color=cmap(j),
                            linewidths=0, label=f"region {c}")
        axes[1].set_title(f"Regions from {self.gb1.cluster_method}")
        axes[1].set_xlabel("Diffusion axis 1"); axes[1].set_ylabel("Diffusion axis 2")
        axes[1].legend(fontsize=7, markerscale=1.5)
        fig.suptitle(f"GB1 diffusion landscape  kernel={self.cfg.kernel}  "
                     f"beta={self.gb1.diffusion_beta}  regions={self.gb1.cluster_method}",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(self.out_dir / "gb1_scatter.png", dpi=130)
        plt.close(fig)

    def _dump_embedding_csv(self, seqs, emb, pred, y, labels, high):
        pd.DataFrame({
            "seq": seqs, "diffusion_axis1": emb[:, 0], "diffusion_axis2": emb[:, 1],
            "predicted": pred, "measured": y, "region": labels, "high_binding": high,
        }).to_csv(self.out_dir / "gb1_embedding.csv", index=False)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _one_hot(self, seqs):
        idx = {c: i for i, c in enumerate(self.alphabet)}
        X = np.zeros((len(seqs), self.L * self.A))
        for r, s in enumerate(seqs):
            for p, ch in enumerate(s):
                X[r, p * self.A + idx[ch]] = 1.0
        return X

    def _require_deps(self):
        try:
            import sklearn  # noqa: F401
        except ImportError as e:  # noqa: BLE001
            raise ImportError("GB1_fitness needs scikit-learn "
                              "(pip install scikit-learn).") from e
        try:
            import scipy  # noqa: F401
        except ImportError as e:  # noqa: BLE001
            raise ImportError("the diffusion embedding needs scipy "
                              "(pip install scipy).") from e

    def _render_summary(self, lines, eig):
        head = [
            "GB1 fitness-landscape reproduction  (Zhou & McCandlish 2020, Fig. 5)",
            "=" * 68,
            f"data:            {self.cfg.data_path}",
            f"kernel / bg:     {self.cfg.kernel} / {self.cfg.background}",
            f"conditioned on:  {self.sfa.n_condition_used} seqs   "
            f"held-out R2 = {self.sfa.test_r2}",
            f"regions (k):     {self.gb1.n_regions}   method = {self.gb1.cluster_method}   "
            f"(defined by landscape GEOMETRY; SHAP explains, does not define)",
            f"diffusion beta:  {self.gb1.diffusion_beta}   "
            f"slow modes = {', '.join(f'{v:.2e}' for v in eig)}",
            f"additive target: {self.gb1.additive_target}   "
            f"high-binding: {self.gb1.high_binding}",
            "",
        ]
        return "\n".join(head + lines) + "\n"


# =========================================================================== #
# CLI
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, data_required=False)   # reuse analysis.py's fit/interpret flags
    # GB1-friendly defaults (data resolved below; out-dir defaults as requested)
    ap.set_defaults(out_dir="output/GB1_fitness_analysis",
                    wildtype=GB1_WT, background="uniform")
    g = ap.add_argument_group("GB1 landscape")
    g.add_argument("--n-regions", type=int, default=3,
                   help="number of high-binding regions (paper: 3)")
    g.add_argument("--cluster-method", default="pcca", choices=["pcca", "kmeans"],
                   help="region definition: 'pcca' (metastable sets, most faithful) "
                        "or 'kmeans' (on the 2-D diffusion coords of high binders)")
    g.add_argument("--max-seqs", type=int, default=None,
                   help="cap sequences used (default/negative: all). The diffusion "
                        "geometry is only faithful on the full 160k landscape.")
    g.add_argument("--shap-batch", type=int, default=2000,
                   help="SHAP-vector batch size (progress granularity)")
    g.add_argument("--high-binding", default="pred_wt", choices=["pred_wt", "wt"],
                   help="high-binding selector: predicted>WT (paper-like, smoothed) or "
                        "measured>WT")
    g.add_argument("--diffusion-beta", type=float, default=1.0,
                   help="diffusion 'temperature' (population-scaled selection); raise "
                        "for tighter arms, lower for more spread")
    g.add_argument("--additive-target", default="measured",
                   choices=["measured", "predicted"],
                   help="within-region additive fit target: raw y (paper) or GP-smoothed")
    g.add_argument("--top-n", type=int, default=100,
                   help="report how many of the top-N measured binders fall in a region")
    g.add_argument("--n-perm", type=int, default=200,
                   help="randomization-test subsets for the additive-R^2 p-value "
                        "(paper used 1000)")
    g.add_argument("--signature-covariance", action="store_true",
                   help="also compute the per-sequence GP posterior sd of the region "
                        "SHAP signature (adds model-confidence columns to the CSV). "
                        "The between-region SEM error bars are always drawn regardless.")
    args = ap.parse_args()

    # --- resolve paths so the script runs identically from root or from scripts/ ---
    root = find_project_root(args.project_root)
    args.project_root = str(root)                       # deterministic module bootstrap
    args.data = _resolve_data_path(args.data, root)
    args.model_dir = _resolve_dir(args.model_dir, root)
    args.out_dir = _resolve_dir(args.out_dir, root)
    print(f"[gb1] project root: {root}")
    print(f"[gb1] data:         {args.data}")
    print(f"[gb1] out-dir:      {args.out_dir}")

    max_seqs = None if (args.max_seqs is not None and args.max_seqs < 0) else args.max_seqs

    cfg = build_config_from_args(args)
    gb1 = GB1Config(
        n_regions=args.n_regions, cluster_method=args.cluster_method,
        max_seqs=max_seqs, shap_batch=args.shap_batch,
        high_binding=args.high_binding, diffusion_beta=args.diffusion_beta,
        additive_target=args.additive_target, top_n=args.top_n, n_perm=args.n_perm,
        signature_covariance=args.signature_covariance,
    )

    # Fit / load the GP + build the SHAP computer (no standard interpret/save pass).
    sfa = SequenceFunctionAnalysis(cfg).run_fit_only()
    GB1FitnessLandscape(sfa, gb1).run()


if __name__ == "__main__":
    main()