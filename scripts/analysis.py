#!/usr/bin/env python
"""
analysis.py
===========
Encapsulated sequence-function analysis: fit an EpiK Gaussian process to a
genotype-phenotype CSV, then interpret what the model learned using three
complementary SHAP-style interaction indices computed by the generalized kernel
trick (GKT):

    * SII          -- Shapley Interaction Index (Grabisch-Roubens). Literature-
                      comparable; smears a high-order effect onto its subsets.
    * Faith-Shap   -- truncated Faithful Shapley Interaction index at max order
                      ``order_max`` (Tsai, Yeh & Ravikumar 2023). Efficient,
                      order-aware; equals the dividends when the true order is
                      <= order_max (and exactly when order_max >= L).
    * Dividends    -- Harsanyi dividends == full-order (order_max == L) Faith-Shap.
                      The pure, non-smearing interaction; the clean epistasis view.

Each index is produced at two levels:
    * LOCAL  -- per target sequence: the signed contribution of each position and
                interaction to *that* sequence's predicted phenotype.
    * GLOBAL -- averaged over a target sample: dataset-level importance of each
                position / interaction (mean |value|), plus the dividend Sobol
                sensitivity (variance across targets).

The background distribution ``pi`` -- the reference every attribution is measured
against -- is configurable:
    * 'uniform'   : average background (each residue equally likely). Product
                    background => dividends carry the functional-ANOVA / Sobol
                    reading. Matches the synthetic validation suite.
    * 'wildtype'  : deviations from a wild-type sequence (effect of mutating away
                    from WT). Requires a wild-type sequence.
    * 'empirical' : dataset residue frequencies.

Design
------
Self-contained: reads the CSV directly and fits a fresh EpiK GP via the same
recipe the test-suite ``build_real_ktc`` uses (``encode_seqs`` ->
``split_training_test`` -> ``EpiK.fit`` -> ``KernelTrickComputer`` ->
``ShapKernelTrickComputer``). No dependency on ``dataset_loader.py``.

Usage
-----
    # from tests/GKT_tests/ or anywhere the project modules resolve:
    python analysis.py --data ../../data/gb1.csv --kernel jenga --gpu \\
        --background wildtype --wildtype VDGV --order-max 5 --out-dir output/gb1_analysis

    # or programmatically:
    cfg = AnalysisConfig(data_path="data/gb1.csv", kernel="jenga",
                         background="uniform", order_max=5)
    result = SequenceFunctionAnalysis(cfg).run()
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DT = torch.float64


# =========================================================================== #
# Path / epik bootstrap  (must run before importing epik or project modules)
# =========================================================================== #

def _bootstrap_paths(project_root: Optional[str] = None, epik_flavor: str = "epik") -> str:
    """Put the project root, modules dir, and epik on ``sys.path``; import epik.

    Mirrors the path setup in ``tests/GKT_tests/test_gkt_brute.py``. If
    ``project_root`` is None it is inferred as two levels up from this file
    (matching tests/GKT_tests/ -> project root); override when running the script
    from a different location.

    Returns the resolved project root path.
    """
    if project_root is None:
        # This file's default home is the project root or a sibling of modules/.
        # Try: (a) this file's dir contains modules/  -> that is the root;
        #      (b) else two levels up (tests/GKT_tests/ layout).
        here = Path(__file__).resolve().parent
        if (here / "modules").is_dir():
            root = here
        elif (here.parent / "modules").is_dir():
            root = here.parent
        else:
            root = here.parents[1] if len(here.parents) >= 2 else here
    else:
        root = Path(project_root).resolve()

    root_str = str(root)
    modules_dir = str(root / "modules")
    for p in (modules_dir, root_str):
        if p not in sys.path:
            sys.path.insert(0, p)

    import epik_loader  # noqa: E402  (resolves now that root is on the path)
    # test_gkt_brute points the loader at ROOT_DIR (it appends '/epik' internally).
    epik_loader.ensure_epik_loaded(epik_code_root=root_str)
    return root_str


# =========================================================================== #
# Config
# =========================================================================== #

@dataclass
class AnalysisConfig:
    """All knobs for one analysis run."""
    data_path: str
    # data columns
    seq_col: str = "seq"
    y_col: str = "y"
    yvar_col: Optional[str] = "y_var"          # None => ignore measurement noise
    alphabet: Optional[str] = None             # None => infer from the sequences
    # model
    kernel: str = "jenga"                      # 'jenga' | 'connectedness'
    ptrain: float = 0.8                        # fraction conditioned on (fit split)
    n_iter: int = 200
    learning_rate: float = 0.02
    seed: int = 0
    device: str = "cpu"                        # 'cpu' | 'cuda'
    # interpretation
    background: str = "uniform"                # 'uniform' | 'wildtype' | 'empirical'
    wildtype: Optional[str] = None             # required if background == 'wildtype'
    order_max: int = 5                         # Faith-Shap max interaction order (ell)
    interaction_order: int = 2                 # enumerate interactions up to this order
    n_global_targets: int = 300               # sample size for the global summaries
    n_top_binders: int = 3                     # default local targets: WT + top binders
    local_targets: Optional[List[str]] = None  # explicit local targets (override default)
    # output
    out_dir: str = "output/analysis"
    make_plots: bool = True
    project_root: Optional[str] = None
    epik_flavor: str = "epik"


# =========================================================================== #
# Result container
# =========================================================================== #

@dataclass
class AnalysisResult:
    config: AnalysisConfig
    alphabet: str
    seq_length: int
    n_sequences: int
    test_r2: Optional[float]
    wildtype: Optional[str]
    background_mode: str
    order_max_effective: int
    faith_is_dividend: bool
    local: dict = field(default_factory=dict)     # per-target index tables
    global_: dict = field(default_factory=dict)   # dataset-level summaries
    out_files: List[str] = field(default_factory=list)


# =========================================================================== #
# The analysis
# =========================================================================== #

class SequenceFunctionAnalysis:
    """Fit an EpiK GP to a sequence-function CSV and interpret it via GKT SHAP."""

    def __init__(self, config: AnalysisConfig):
        self.cfg = config
        self._root = _bootstrap_paths(config.project_root, config.epik_flavor)

        # Imports resolve only after the bootstrap above.
        from epik.epik.kernel import JengaKernel, ConnectednessKernel  # noqa: E402
        from epik.epik.model import EpiK                               # noqa: E402
        from epik.epik.utils import encode_seqs, split_training_test   # noqa: E402
        from modules.kernel_trick_computer_refactored import KernelTrickComputer  # noqa: E402
        from modules.shap_kernel_trick_computer import (               # noqa: E402
            ShapKernelTrickComputer, make_background,
        )

        self._JengaKernel = JengaKernel
        self._ConnectednessKernel = ConnectednessKernel
        self._EpiK = EpiK
        self._encode_seqs = encode_seqs
        self._split_training_test = split_training_test
        self._KernelTrickComputer = KernelTrickComputer
        self._ShapKernelTrickComputer = ShapKernelTrickComputer
        self._make_background = make_background

        # populated by run()
        self.seqs: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.y_var: Optional[np.ndarray] = None
        self.alphabet: Optional[str] = None
        self.L: Optional[int] = None
        self.A: Optional[int] = None
        self.model = None
        self.ktc = None
        self.shap = None
        self.pi: Optional[torch.Tensor] = None
        self.test_r2: Optional[float] = None

    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        cfg = self.cfg
        path = Path(cfg.data_path)
        if not path.exists():
            raise FileNotFoundError(f"data file not found: {path}")
        df = pd.read_csv(path)
        for col in (cfg.seq_col, cfg.y_col):
            if col not in df.columns:
                raise ValueError(f"column '{col}' not in {path.name} "
                                 f"(have: {list(df.columns)}).")
        seqs = df[cfg.seq_col].astype(str).to_numpy()
        y = df[cfg.y_col].to_numpy(dtype=float)
        if cfg.yvar_col and cfg.yvar_col in df.columns:
            y_var = df[cfg.yvar_col].to_numpy(dtype=float)
        else:
            y_var = np.zeros_like(y)
            if cfg.yvar_col:
                print(f"[note] '{cfg.yvar_col}' not found; using zero measurement noise.")

        lengths = {len(s) for s in seqs}
        if len(lengths) != 1:
            raise ValueError(f"sequences have varying lengths {sorted(lengths)}; "
                             "all rows must be the same length.")
        L = lengths.pop()
        alphabet = cfg.alphabet or "".join(sorted({c for s in seqs for c in s}))
        A = len(alphabet)

        self.seqs, self.y, self.y_var = seqs, y, y_var
        self.alphabet, self.L, self.A = alphabet, L, A
        print(f"[load] {len(seqs)} sequences, L={L}, alphabet='{alphabet}' (A={A})")

    # ------------------------------------------------------------------ #
    # 2. Fit the GP
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        X = self._encode_seqs(list(self.seqs), alphabet=self.alphabet)
        y_t = torch.as_tensor(self.y, dtype=torch.float32)
        yv_t = torch.as_tensor(self.y_var, dtype=torch.float32)

        # Train/test split for a held-out R2 (ptrain<1). ptrain>=1 conditions on all.
        if cfg.ptrain < 1.0:
            train_x, train_y, test_x, test_y, train_y_var = self._split_training_test(
                X, y_t, yv_t, ptrain=cfg.ptrain, dtype=torch.float32,
            )
        else:
            train_x, train_y, train_y_var = X, y_t, yv_t
            test_x = test_y = None

        kernel_cls = {"jenga": self._JengaKernel,
                      "connectedness": self._ConnectednessKernel}[cfg.kernel]
        model = self._EpiK(kernel_cls(n_alleles=self.A, seq_length=self.L),
                           track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        print(f"[fit] kernel={cfg.kernel}  n_iter={cfg.n_iter}  lr={cfg.learning_rate}  "
              f"ptrain={cfg.ptrain}")
        model.fit(n_iter=cfg.n_iter, learning_rate=cfg.learning_rate)

        # Prediction populates the caches the KernelTrickComputer reads.
        if test_x is not None:
            pred = model.predict(test_x, calc_variance=False)
            self.test_r2 = _r2(_to_np(pred), _to_np(test_y))
            print(f"[fit] held-out R2 = {self.test_r2:.4f}  (on {len(test_y)} test seqs)")
        else:
            # still need a predict() call to populate caches; predict the training set
            model.predict(train_x, calc_variance=False)
            self.test_r2 = None

        self.model = model
        self.ktc = self._KernelTrickComputer(model, self.alphabet, self.L)

        if cfg.device != "cpu":
            self._move_to_device(cfg.device)
        self.ktc.set_precision(use_double=True)
        self.shap = self._ShapKernelTrickComputer(ktc=self.ktc)

    def _move_to_device(self, device: str) -> None:
        """Move model + all cached tensors onto device BEFORE building the computer."""
        ktc = self.ktc
        if hasattr(ktc, "model") and hasattr(ktc.model, "to"):
            try:
                ktc.model = ktc.model.to(device)
            except Exception as e:  # noqa: BLE001
                print(f"  [note] could not move live model to {device}: {e}")
        for name, val in list(vars(ktc).items()):
            if torch.is_tensor(val):
                setattr(ktc, name, val.to(device))
        ktc.device = device

    # ------------------------------------------------------------------ #
    # 3. Background + targets
    # ------------------------------------------------------------------ #
    def _build_background(self) -> torch.Tensor:
        cfg = self.cfg
        dev = torch.device(cfg.device)
        if cfg.background == "empirical":
            counts = torch.zeros(self.L, self.A, dtype=DT)
            sym_to_idx = {c: i for i, c in enumerate(self.alphabet)}
            for s in self.seqs:
                for p, ch in enumerate(s):
                    counts[p, sym_to_idx[ch]] += 1.0
            return self._make_background("empirical", self.L, self.alphabet,
                                         empirical_counts=counts, device=dev, dtype=DT)
        if cfg.background == "wildtype":
            wt = cfg.wildtype or self._infer_wildtype()
            return self._make_background("wildtype", self.L, self.alphabet,
                                         wildtype=wt, device=dev, dtype=DT)
        return self._make_background("uniform", self.L, self.alphabet, device=dev, dtype=DT)

    def _infer_wildtype(self) -> str:
        """Default wild type = the sequence with the highest measured phenotype."""
        wt = str(self.seqs[int(np.argmax(self.y))])
        print(f"[bg] no wildtype given; using max-phenotype sequence '{wt}' as WT reference.")
        return wt

    def _resolve_local_targets(self) -> List[str]:
        cfg = self.cfg
        if cfg.local_targets:
            return list(cfg.local_targets)
        targets: List[str] = []
        if cfg.background == "wildtype" and cfg.wildtype:
            targets.append(cfg.wildtype)
        # top binders by measured phenotype (dedup, preserve order)
        order = np.argsort(-self.y)
        for i in order[: cfg.n_top_binders + len(targets)]:
            s = str(self.seqs[i])
            if s not in targets:
                targets.append(s)
            if len(targets) >= cfg.n_top_binders + (1 if cfg.wildtype else 0):
                break
        return targets

    def _global_target_sample(self) -> List[str]:
        cfg = self.cfg
        n = min(cfg.n_global_targets, len(self.seqs))
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(self.seqs), size=n, replace=False)
        return [str(self.seqs[i]) for i in idx]

    # ------------------------------------------------------------------ #
    # 4. Interpret
    # ------------------------------------------------------------------ #
    def interpret(self) -> AnalysisResult:
        cfg = self.cfg
        self.pi = self._build_background()
        ell = min(cfg.order_max, self.L)
        faith_is_dividend = ell >= self.L
        if faith_is_dividend:
            print(f"[faith] order_max={cfg.order_max} >= L={self.L}: "
                  "Faith-Shap collapses to the Harsanyi dividends (identical values).")

        # position + interaction tuples to score
        positions = [(i,) for i in range(self.L)]
        inter_tuples = _enumerate_interactions(self.L, cfg.interaction_order)

        result = AnalysisResult(
            config=cfg, alphabet=self.alphabet, seq_length=self.L,
            n_sequences=len(self.seqs), test_r2=self.test_r2,
            wildtype=(cfg.wildtype if cfg.background == "wildtype" else None),
            background_mode=cfg.background, order_max_effective=ell,
            faith_is_dividend=faith_is_dividend,
        )

        # ---- LOCAL: per-target tables ---- #
        local_targets = self._resolve_local_targets()
        print(f"[local] targets: {local_targets}")
        for tgt in local_targets:
            result.local[tgt] = self._local_tables(tgt, positions, inter_tuples, ell)

        # ---- GLOBAL: dataset-level summaries ---- #
        xs = self._global_target_sample()
        print(f"[global] averaging over {len(xs)} target sequences")
        result.global_ = self._global_tables(xs, positions, inter_tuples, ell)

        self._result = result
        return result

    def _local_tables(self, tgt, positions, inter_tuples, ell) -> dict:
        shap, pi = self.shap, self.pi
        # first-order Shapley (positions) for reference importance
        phi = shap.shap_values(tgt, pi, indices=list(range(self.L)))
        # SII on positions + interactions (raw SII scale)
        sii_T, sii = shap.shap_interaction_values(
            tgt, pi, interactions=positions + inter_tuples, normalize="sii")
        # Faith-Shap at ell
        fs_T, fs = shap.faith_shap_values(
            tgt, pi, order_max=ell, interactions=positions + inter_tuples)
        # Dividends (full-order Faith-Shap)
        dv_T, dv = shap.dividend_values(
            tgt, pi, interactions=positions + inter_tuples)
        return {
            "shapley_positions": (list(range(self.L)), _to_np(phi)),
            "sii": (sii_T, _to_np(sii)),
            "faith_shap": (fs_T, _to_np(fs)),
            "dividend": (dv_T, _to_np(dv)),
        }

    def _global_tables(self, xs, positions, inter_tuples, ell) -> dict:
        shap, pi = self.shap, self.pi
        out = {}
        # position importance (mean |phi_i|)
        idx, mean_abs_phi = shap.global_shap_importance(xs, pi, indices=list(range(self.L)))
        out["position_importance"] = (idx, _to_np(mean_abs_phi))
        # dividend global summary (mean|a|, signed, Sobol variance)
        dv_T, dv_sum = shap.global_dividend_summary(
            xs, pi, interactions=positions + inter_tuples, normalize_sobol=False)
        out["dividend_summary"] = (dv_T, {k: _to_np(v) for k, v in dv_sum.items()})
        # SII global importance (mean |SII|) over the target sample
        sii_T, SII = _batch_interaction_abs_mean(
            shap, xs, pi, positions + inter_tuples, normalize="sii")
        out["sii_summary"] = (sii_T, SII)
        # Faith-Shap global summary
        fs_T, fs_sum = shap.global_faith_shap_summary(
            xs, pi, order_max=ell, interactions=positions + inter_tuples)
        out["faith_shap_summary"] = (fs_T, {k: _to_np(v) for k, v in fs_sum.items()})
        return out

    # ------------------------------------------------------------------ #
    # 5. Output
    # ------------------------------------------------------------------ #
    def save(self, result: AnalysisResult) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # numeric dumps (CSV per table)
        self._dump_global_csv(result, out_dir)
        for tgt, tables in result.local.items():
            self._dump_local_csv(tgt, tables, out_dir)

        if cfg.make_plots:
            self._plot_global(result, out_dir)
            for tgt, tables in result.local.items():
                self._plot_local(tgt, tables, out_dir)

        # a short text summary
        summ = out_dir / "SUMMARY.txt"
        with open(summ, "w") as fh:
            fh.write(_render_summary(result))
        result.out_files.append(str(summ))
        print(f"[save] wrote {len(result.out_files)} files to {out_dir}")

    def _dump_global_csv(self, result, out_dir):
        g = result.global_
        # positions
        idx, mean_abs = g["position_importance"]
        pd.DataFrame({"position": idx, "mean_abs_shapley": mean_abs}
                     ).to_csv(out_dir / "global_position_importance.csv", index=False)
        result.out_files.append(str(out_dir / "global_position_importance.csv"))
        # interactions: SII, Faith-Shap, dividend + Sobol, aligned on the tuple list
        dv_T, dv = g["dividend_summary"]
        sii_T, sii = g["sii_summary"]
        fs_T, fs = g["faith_shap_summary"]
        rows = []
        for i, T in enumerate(dv_T):
            rows.append({
                "tuple": "x".join(str(p) for p in T),
                "order": len(T),
                "sii_mean_abs": sii[i],
                "faith_shap_mean_abs": fs["mean_abs"][i],
                "dividend_mean_abs": dv["mean_abs"][i],
                "dividend_sobol": dv["sobol_unnorm"][i],
                "dividend_mean_signed": dv["mean_signed"][i],
            })
        pd.DataFrame(rows).to_csv(out_dir / "global_interactions.csv", index=False)
        result.out_files.append(str(out_dir / "global_interactions.csv"))

    def _dump_local_csv(self, tgt, tables, out_dir):
        sii_T, sii = tables["sii"]
        fs_T, fs = tables["faith_shap"]
        dv_T, dv = tables["dividend"]
        rows = []
        for i, T in enumerate(sii_T):
            rows.append({
                "tuple": "x".join(str(p) for p in T),
                "order": len(T),
                "sii": sii[i],
                "faith_shap": fs[i],
                "dividend": dv[i],
            })
        fn = out_dir / f"local_{tgt}.csv"
        pd.DataFrame(rows).to_csv(fn, index=False)
        tables["_csv"] = str(fn)

    # ---- plots ---- #
    def _plot_global(self, result, out_dir):
        g = result.global_
        idx, mean_abs = g["position_importance"]
        dv_T, dv = g["dividend_summary"]
        sii_T, sii = g["sii_summary"]
        fs_T, fs = g["faith_shap_summary"]

        fig, axes = plt.subplots(3, 1, figsize=(max(8, len(dv_T) * 0.35), 11))
        # panel 1: position importance
        axes[0].bar([str(i) for i in idx], mean_abs, color="#c0504d")
        axes[0].set_title("Global position importance  mean|phi_i|")
        axes[0].set_xlabel("position"); axes[0].set_ylabel("mean |phi|")
        # panel 2: three interaction indices side by side
        labels = ["x".join(map(str, T)) for T in dv_T]
        xpos = np.arange(len(labels)); w = 0.27
        axes[1].bar(xpos - w, sii, width=w, label="SII |mean|", color="#8064a2")
        axes[1].bar(xpos, fs["mean_abs"], width=w,
                    label=f"Faith-Shap(ell={result.order_max_effective}) |mean|",
                    color="#4f81bd")
        axes[1].bar(xpos + w, dv["mean_abs"], width=w, label="dividend |mean|",
                    color="#9bbb59")
        axes[1].set_xticks(xpos); axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
        axes[1].set_title("Global interaction importance (three indices)")
        axes[1].legend(fontsize=8)
        # panel 3: dividend Sobol
        axes[2].bar(xpos, dv["sobol_unnorm"], color="#2c7fb8")
        axes[2].set_xticks(xpos); axes[2].set_xticklabels(labels, rotation=90, fontsize=7)
        axes[2].set_title("Global Sobol sensitivity  Var_x a(T)  (dividends)")
        fig.suptitle(f"Global analysis  bg={result.background_mode}  "
                     f"kernel={result.config.kernel}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fn = out_dir / "global_analysis.png"
        fig.savefig(fn, dpi=130); plt.close(fig)
        result.out_files.append(str(fn))

    def _plot_local(self, tgt, tables, out_dir):
        sii_T, sii = tables["sii"]
        _, fs = tables["faith_shap"]
        _, dv = tables["dividend"]
        labels = ["x".join(map(str, T)) for T in sii_T]
        xpos = np.arange(len(labels)); w = 0.27
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 5))
        ax.bar(xpos - w, sii, width=w, label="SII", color="#8064a2")
        ax.bar(xpos, fs, width=w, label="Faith-Shap", color="#4f81bd")
        ax.bar(xpos + w, dv, width=w, label="dividend", color="#9bbb59")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(f"Local attribution for '{tgt}'  (bg={self.cfg.background})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fn = out_dir / f"local_{tgt}.png"
        fig.savefig(fn, dpi=130); plt.close(fig)
        tables["_png"] = str(fn)

    # ------------------------------------------------------------------ #
    # orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> AnalysisResult:
        self.load()
        self.fit()
        result = self.interpret()
        self.save(result)
        return result


# =========================================================================== #
# small helpers
# =========================================================================== #

def _enumerate_interactions(L: int, order: int) -> List[Tuple[int, ...]]:
    tuples: List[Tuple[int, ...]] = []
    for k in range(2, order + 1):
        tuples.extend(itertools.combinations(range(L), k))
    return tuples


def _batch_interaction_abs_mean(shap, xs, pi, tuples, normalize="sii"):
    """mean_x |SII_T(x)| over targets (SII has no dedicated global method)."""
    rows = []
    T_list = None
    for x in xs:
        T_list, vals = shap.shap_interaction_values(
            x, pi, interactions=tuples, normalize=normalize)
        rows.append(_to_np(vals))
    arr = np.abs(np.stack(rows, axis=0)).mean(axis=0)
    return T_list, arr


def _to_np(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _r2(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _render_summary(result: AnalysisResult) -> str:
    c = result.config
    lines = [
        "Sequence-function analysis summary",
        "=" * 40,
        f"data:            {c.data_path}",
        f"sequences:       {result.n_sequences}   L={result.seq_length}   "
        f"alphabet='{result.alphabet}'",
        f"kernel:          {c.kernel}   ptrain={c.ptrain}   n_iter={c.n_iter}",
        f"held-out R2:     {result.test_r2 if result.test_r2 is not None else 'n/a (all-data fit)'}",
        f"background:      {result.background_mode}"
        + (f"  (WT={result.wildtype})" if result.wildtype else ""),
        f"Faith-Shap ell:  {result.order_max_effective}"
        + ("   [== dividends: ell >= L]" if result.faith_is_dividend else ""),
        f"local targets:   {list(result.local.keys())}",
        "",
        "Files: per-target local_*.csv/png, global_*.csv, global_analysis.png.",
    ]
    return "\n".join(lines) + "\n"


def resolve_device(want_gpu: bool) -> str:
    if not want_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    print("[warn] --gpu requested but no CUDA device visible; using CPU. "
          "Are you inside a GPU allocation (srun ... --gres=gpu:1)?")
    return "cpu"


# =========================================================================== #
# CLI
# =========================================================================== #

def build_config_from_args(args) -> AnalysisConfig:
    return AnalysisConfig(
        data_path=args.data,
        seq_col=args.seq_col, y_col=args.y_col, yvar_col=args.yvar_col,
        alphabet=args.alphabet,
        kernel=args.kernel, ptrain=args.ptrain, n_iter=args.n_iter,
        learning_rate=args.lr, seed=args.seed, device=resolve_device(args.gpu),
        background=args.background, wildtype=args.wildtype,
        order_max=args.order_max, interaction_order=args.interaction_order,
        n_global_targets=args.n_global_targets, n_top_binders=args.n_top_binders,
        local_targets=(args.targets.split(",") if args.targets else None),
        out_dir=args.out_dir, make_plots=(not args.no_plot),
        project_root=args.project_root, epik_flavor=args.epik_flavor,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to the sequence-function CSV")
    ap.add_argument("--seq-col", default="seq")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--yvar-col", default="y_var")
    ap.add_argument("--alphabet", default=None, help="override; default infers from data")
    ap.add_argument("--kernel", default="jenga", choices=["jenga", "connectedness"])
    ap.add_argument("--ptrain", type=float, default=0.8,
                    help="fraction to condition on; <1 gives a held-out R2, >=1 uses all")
    ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--background", default="uniform",
                    choices=["uniform", "wildtype", "empirical"])
    ap.add_argument("--wildtype", default=None,
                    help="WT sequence for background='wildtype' (default: max-phenotype seq)")
    ap.add_argument("--order-max", type=int, default=5,
                    help="Faith-Shap max interaction order (ell); >=L => dividends")
    ap.add_argument("--interaction-order", type=int, default=2,
                    help="enumerate interaction tuples up to this order")
    ap.add_argument("--n-global-targets", type=int, default=300)
    ap.add_argument("--n-top-binders", type=int, default=3)
    ap.add_argument("--targets", default=None,
                    help="comma-separated explicit local targets (overrides default)")
    ap.add_argument("--out-dir", default="output/analysis")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--project-root", default=None,
                    help="project root (default: inferred; contains modules/ and epik/)")
    ap.add_argument("--epik-flavor", default="epik", choices=["epik", "epik_dev"])
    args = ap.parse_args()

    cfg = build_config_from_args(args)
    result = SequenceFunctionAnalysis(cfg).run()
    print("\n" + _render_summary(result))
    return result


if __name__ == "__main__":
    main()