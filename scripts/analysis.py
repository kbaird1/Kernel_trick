#!/usr/bin/env python
"""
analysis.py
===========
General-purpose sequence-function interpretation.

Fit (or load) an EpiK Gaussian process to a genotype-phenotype CSV, then explain
what the model learned using three complementary SHAP-style interaction indices
computed by the generalized kernel trick (GKT):

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
                    reading.
    * 'wildtype'  : deviations from a wild-type sequence (effect of mutating away
                    from WT). Requires a wild-type sequence.
    * 'empirical' : dataset residue frequencies.

This script is deliberately generic: it knows nothing about any particular data
set. For the GB1 fitness-landscape reproduction of Zhou & McCandlish 2020, see
the separate ``GB1_fitness.py``, which drives this module as a library.

Faith-Shap on long sequences (order_cap)
----------------------------------------
Truncated Faith-Shap (``order_max`` == ell < L) is NOT the dividend vector chopped
at order ell: to stay efficient, each reported <=ell term must ABSORB the higher-
order epistasis sitting above it, so computing it exactly requires enumerating every
superset dividend up to order L (~2^(L-ell) of them). That is intractable for long
sequences even though the *dividends* of the same low-order tuples are cheap. The
``--order-cap`` flag bounds that redistribution tail at a chosen order: the result is
Faith-Shap of the ``order_cap``-order truncation of the model -- an honest, tractable
approximation. It only bites when ell < L; at ``order_max >= L`` Faith-Shap equals the
dividends and the tail is empty. SII and dividend outputs never use the tail and are
unaffected. See ``_preflight_faith`` (which hard-stops an intractable exact tail) and
``ShapKernelTrickComputer.faith_shap_values``.

Verbose / debugging (--verbose)
-------------------------------
By default the script prints its usual condensed ``[load]``/``[fit]``/``[global]``
lines. Passing ``--verbose`` adds, without changing any of the existing output or any
numbers: a per-phase wall-clock timer (load, fit, interpret, and each global
sub-computation), a heartbeat around each global block (which method is running, over
how many targets / tuples, and how long it took), per-target progress in the
``--global-covariance`` loop, and a one-time config echo (resolved conditioning size,
L/A, ell, resolved Faith-Shap cap, interaction order, tuple count, global target
count). All of it is gated on the flag, so a non-verbose run is byte-for-byte identical
to before.

Memory / OOM notes
------------------
An *exact* GP costs O(n_condition^2) memory (it forms the training kernel matrix
K_XX and its Cholesky) and O(n_condition^3) compute, and ``model.fit`` redoes that
work every iteration. That cost is driven ENTIRELY by how many points we condition
on (``set_data``) -- not by the size of the dataset. The same K_XX-sized cache
(``covar_cache``) is what the ShapKernelTrickComputer multiplies against later, so
an over-large conditioning set OOMs the interpretation step too.

Two safe recipes:

    * BOUNDED FIT (default). Condition on a bounded random subset of the data
      (``n_condition`` points) and run ``model.fit`` on just that subset. Cheap,
      works with any kernel.

    * LOAD PRE-FIT PARAMS. Provide ``params_path`` to a saved hyperparameter
      ``.pth``; the model is built with those hyperparameters and ``model.fit``
      is SKIPPED entirely. (Builds a GeneralProductKernel.)

Predictions are always batched (``predict_batch_size``) to avoid the
O(n_test x n_train) spike.

Usage
-----
    python analysis.py --data data/gb1.csv --kernel jenga --gpu \\
        --n-condition 2000 --background wildtype --wildtype VDGV \\
        --order-max 5 --out-dir output/gb1_analysis

    # long sequence, low-order Faith-Shap: cap the redistribution tail
    python analysis.py --data data/library_M/MABAa_ACE2.seqy.csv --kernel jenga \\
        --gpu --n-condition 5000 --interaction-order 2 --order-max 2 --order-cap 3 \\
        --background empirical --n-global-targets 300 --out-dir output/libraryM_ACE2

    # see exactly where the time goes on a big global pass
    python analysis.py --data data/gb1.csv --kernel jenga --gpu \\
        --n-condition 20000 --load-model --model-name gb1_jenga_nc20000 \\
        --order-max 4 --interaction-order 4 --verbose \\
        --out-dir output/gb1_analysis_uniform

    # load pre-optimized hyperparameters and skip model.fit():
    python analysis.py --data data/aav.csv --kernel jenga --gpu \\
        --params-path epik/aav_models/aav.full.1.GeneralProduct.model_params.pth \\
        --n-condition 5000 --out-dir output/aav_analysis

    # programmatically:
    cfg = AnalysisConfig(data_path="data/gb1.csv", kernel="jenga",
                         background="uniform", order_max=5, n_condition=2000)
    result = SequenceFunctionAnalysis(cfg).run()
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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

    If ``project_root`` is None it is inferred: prefer the directory that contains
    ``modules/``, walking up from this file. Override when running from elsewhere.

    Returns the resolved project root path.
    """
    if project_root is None:
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
    ptrain: float = 0.8                        # fraction conditioned on (only if n_condition is None)
    n_iter: int = 200
    learning_rate: float = 0.02
    seed: int = 0
    device: str = "cpu"                        # 'cpu' | 'cuda'
    # memory / conditioning controls  (see the "Memory / OOM notes" docstring)
    n_condition: Optional[int] = 2000          # HARD cap on GP conditioning points.
    params_path: Optional[str] = None          # pre-optimized hyperparameter .pth; skips fit
    predict_batch_size: int = 5000             # batch predictions
    n_r2_eval: int = 2000                      # cap on held-out points for the R2 estimate
    # model caching: opt-in load / save of trained model params (under model_dir)
    model_dir: str = "models"
    model_name: Optional[str] = None
    load_model: bool = False
    save_model: bool = False
    # interpretation
    background: str = "uniform"                # 'uniform' | 'wildtype' | 'empirical'
    wildtype: Optional[str] = None             # required if background == 'wildtype'
    order_max: int = 5                         # Faith-Shap max interaction order (ell)
    order_cap: Optional[int] = None            # Faith-Shap tail cap; None => L (exact) when
                                               # the tail is small enough, else required for
                                               # ell < L (see _preflight_faith).
    interaction_order: int = 2                 # enumerate interactions up to this order
    n_global_targets: Optional[int] = None     # None/negative => average over ALL
                                                # sequences; an int => random subsample
    n_top_binders: int = 3
    local_targets: Optional[List[str]] = None  # explicit local targets (override default)
    # posterior covariance of the attributions (from the GP posterior over functions)
    local_covariance: bool = True              # per-target sd + z on local tables (cheap)
    global_covariance: bool = False            # per-term mean posterior sd on the global
                                               # tables; REQUIRES a finite n_global_targets
                                               # (intractable over the full dataset)
    # output
    out_dir: str = "output/analysis"
    make_plots: bool = True
    verbose: bool = False                      # extra timing / progress / config echo;
                                               # purely additive, changes no numbers
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
    n_condition_used: Optional[int] = None
    local: dict = field(default_factory=dict)
    global_: dict = field(default_factory=dict)
    out_files: List[str] = field(default_factory=list)


# =========================================================================== #
# The analysis
# =========================================================================== #

class SequenceFunctionAnalysis:
    """Fit an EpiK GP to a sequence-function CSV and interpret it via GKT SHAP.

    After ``fit()`` the fitted GP, KernelTrickComputer, and ShapKernelTrickComputer
    are exposed as ``self.model`` / ``self.ktc`` / ``self.shap`` so downstream tools
    (e.g. GB1_fitness.py) can reuse them without re-fitting.
    """

    def __init__(self, config: AnalysisConfig):
        self.cfg = config
        self._root = _bootstrap_paths(config.project_root, config.epik_flavor)

        # Imports resolve only after the bootstrap above.
        from epik.epik.kernel import (                                  # noqa: E402
            JengaKernel, ConnectednessKernel, GeneralProductKernel,
        )
        from epik.epik.model import EpiK                               # noqa: E402
        from epik.epik.utils import encode_seqs, split_training_test   # noqa: E402
        from modules.kernel_trick_computer_refactored import KernelTrickComputer  # noqa: E402
        from modules.shap_kernel_trick_computer import (               # noqa: E402
            ShapKernelTrickComputer, make_background,
        )

        self._JengaKernel = JengaKernel
        self._ConnectednessKernel = ConnectednessKernel
        self._GeneralProductKernel = GeneralProductKernel
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
        self.n_condition_used: Optional[int] = None
        # resolved Faith-Shap tail cap (set by _preflight_faith; None => exact, cap = L)
        self._faith_cap: Optional[int] = None

    # ------------------------------------------------------------------ #
    # 0. Verbose logging helpers  (all output gated on cfg.verbose)
    # ------------------------------------------------------------------ #
    def _log(self, msg: str, level: str = "info") -> None:
        """Print ``msg`` respecting the verbosity level.

        ``level='info'`` always prints (these are the script's normal condensed
        lines). ``level='debug'`` prints ONLY when ``cfg.verbose`` is set, so a
        non-verbose run produces exactly the output it did before this helper
        existed. Debug lines are tagged ``[debug]`` to keep them visually distinct
        from the normal ``[load]``/``[fit]``/``[global]`` stream.
        """
        if level == "debug":
            if not self.cfg.verbose:
                return
            print(f"[debug] {msg}", flush=True)
        else:
            print(msg, flush=True)

    @contextmanager
    def _phase(self, name: str):
        """Time a named phase; announce start/duration only when ``--verbose``.

        Non-verbose: a no-op wrapper (prints nothing, so existing output is
        unchanged). Verbose: prints ``[time] <name>: start`` on entry and
        ``[time] <name>: <elapsed>`` on exit, choosing seconds vs minutes for
        readability. The body runs identically either way.
        """
        if not self.cfg.verbose:
            yield
            return
        print(f"[time] {name}: start", flush=True)
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            human = f"{dt:.1f}s" if dt < 90 else f"{dt / 60:.1f} min"
            print(f"[time] {name}: {human}", flush=True)

    def _echo_config(self, ell: int, n_tuples: int, n_global: int) -> None:
        """One-time verbose echo of the resolved run shape (numbers only, no math).

        Printed at the top of interpret() so a debugging run shows -- before the
        long global pass begins -- exactly what it is about to compute over: the
        conditioning size, sequence geometry, the effective Faith-Shap order and
        resolved tail cap, the interaction order, how many position/interaction
        tuples each attribution covers, and how many targets the global summaries
        average over (the last one is usually the thing that explains a slow run).
        """
        if not self.cfg.verbose:
            return
        cfg = self.cfg
        if ell >= self.L:
            cap_str = f"n/a (ell={ell} >= L={self.L}: Faith-Shap == dividends)"
        elif self._faith_cap is None:
            cap_str = f"exact (tail runs to L={self.L})"
        else:
            cap_str = f"{self._faith_cap} (truncated => approximation)"
        lines = [
            "[debug] --- resolved run configuration ---",
            f"[debug]   data:              {cfg.data_path}",
            f"[debug]   conditioned on:    {self.n_condition_used} sequences",
            f"[debug]   sequence geometry: L={self.L}  A={self.A}  "
            f"(code space A^L = {self.A ** self.L:,})",
            f"[debug]   background:        {cfg.background}"
            + (f"  (WT={cfg.wildtype})" if cfg.background == "wildtype" else ""),
            f"[debug]   Faith-Shap ell:    {ell}  (order_max={cfg.order_max})",
            f"[debug]   Faith-Shap cap:    {cap_str}",
            f"[debug]   interaction order: {cfg.interaction_order}",
            f"[debug]   tuples per attr:   {n_tuples}  "
            f"({self.L} singletons + {n_tuples - self.L} interactions)",
            f"[debug]   global targets:    {n_global:,}"
            + ("  (ALL sequences -- pass --n-global-targets N to subsample)"
               if n_global == len(self.seqs) else "  (--n-global-targets)"),
            f"[debug]   local covariance:  {'on' if cfg.local_covariance else 'off'}",
            f"[debug]   global covariance: {'on' if cfg.global_covariance else 'off'}",
            "[debug] ----------------------------------",
        ]
        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        cfg = self.cfg
        path = Path(cfg.data_path)
        if not path.exists():
            # tolerate being run from a different cwd (e.g. inside scripts/): retry
            # relative to the project root discovered during bootstrap.
            alt = Path(self._root) / cfg.data_path
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(
                    f"data file not found: {cfg.data_path}  (also tried {alt}).")
        self._log(f"reading CSV from {path}", level="debug")
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
        self._log(f"y range: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}; "
                  f"y_var mean={y_var.mean():.3g}", level="debug")

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

        train_x, train_y, train_y_var, test_x, test_y = self._make_conditioning_split(X, y_t, yv_t)
        self.n_condition_used = int(train_x.shape[0])

        if cfg.params_path:
            model = self._build_model_from_params(train_x, train_y, train_y_var)
        else:
            cache = self._model_cache_path()
            model = None
            if cfg.load_model and cache is not None and cache.exists():
                try:
                    model = self._load_cached_model(train_x, train_y, train_y_var, cache)
                except Exception as e:  # noqa: BLE001
                    print(f"[fit][warn] could not load {cache} ({e}); training instead.")
                    model = None
            if model is None:
                kernel_cls = {"jenga": self._JengaKernel,
                              "connectedness": self._ConnectednessKernel}[cfg.kernel]
                model = self._EpiK(kernel_cls(n_alleles=self.A, seq_length=self.L),
                                   track_progress=True)
                model.set_data(train_x, train_y, train_y_var)
                print(f"[fit] kernel={cfg.kernel}  n_iter={cfg.n_iter}  lr={cfg.learning_rate}  "
                      f"n_condition={train_x.shape[0]}")
                with self._phase("fit.optimize"):
                    model.fit(n_iter=cfg.n_iter, learning_rate=cfg.learning_rate)
                if cfg.save_model and cache is not None:
                    try:
                        self._save_cached_model(model, cache)
                    except Exception as e:  # noqa: BLE001
                        print(f"[fit][warn] could not save model to {cache} ({e}); "
                              "continuing without a saved copy.")

        # Prediction populates the caches the KernelTrickComputer reads.
        if test_x is not None and test_x.shape[0] > 0:
            with self._phase("fit.predict_eval"):
                pred = self._predict_batched(model, test_x)
            self.test_r2 = _r2(pred, _to_np(test_y))
            print(f"[fit] held-out R2 = {self.test_r2:.4f}  (on {len(test_y)} test seqs)")
        else:
            self._predict_batched(model, train_x[: min(train_x.shape[0], cfg.predict_batch_size)])
            self.test_r2 = None

        self.model = model
        self._log("building KernelTrickComputer (reads populated prediction caches)",
                  level="debug")
        self.ktc = self._KernelTrickComputer(model, self.alphabet, self.L)

        if cfg.device != "cpu":
            self._log(f"moving KernelTrickComputer tensors to {cfg.device}", level="debug")
            self._move_to_device(cfg.device)
        self.ktc.set_precision(use_double=True)
        self.shap = self._ShapKernelTrickComputer(ktc=self.ktc)

    # ------------------------------------------------------------------ #
    # 2a. Conditioning set / model construction / batched predict
    # ------------------------------------------------------------------ #
    def _make_conditioning_split(self, X, y_t, yv_t):
        """Pick the GP conditioning set + a capped held-out eval set.

        With ``n_condition`` set (the default), condition on a bounded random
        subset so exact-GP memory ~ O(n_condition^2) stays fixed no matter how
        large the dataset is. With ``n_condition=None`` fall back to the legacy
        ptrain-fraction split (and warn if that is large).
        """
        cfg = self.cfg
        N = X.shape[0]

        if cfg.n_condition is None:
            if cfg.ptrain < 1.0:
                tr_x, tr_y, te_x, te_y, tr_yv = self._split_training_test(
                    X, y_t, yv_t, ptrain=cfg.ptrain, dtype=torch.float32)
            else:
                tr_x, tr_y, tr_yv, te_x, te_y = X, y_t, yv_t, None, None
            if tr_x.shape[0] > 8000:
                print(f"[warn] conditioning on {tr_x.shape[0]} points (ptrain={cfg.ptrain}). "
                      "Exact-GP memory ~ O(n^2); this may OOM. Set --n-condition to cap it.")
            return tr_x, tr_y, tr_yv, te_x, te_y

        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(N)
        n_cond = min(cfg.n_condition, N)
        c = perm[:n_cond]
        tr_x, tr_y, tr_yv = X[c], y_t[c], yv_t[c]

        rest = perm[n_cond:]
        if len(rest) > 0:
            e = rest[: min(cfg.n_r2_eval, len(rest))]
            te_x, te_y = X[e], y_t[e]
        else:
            te_x, te_y = None, None
        print(f"[fit] conditioning on {n_cond}/{N} sequences (exact-GP mem ~ O(n_condition^2)); "
              f"eval on {0 if te_x is None else te_x.shape[0]} held-out seqs")
        return tr_x, tr_y, tr_yv, te_x, te_y

    def _build_model_from_params(self, train_x, train_y, train_y_var):
        """Load pre-optimized hyperparameters and skip model.fit()."""
        cfg = self.cfg
        dev = cfg.device
        params = torch.load(cfg.params_path, map_location=dev)
        try:
            kernel = self._GeneralProductKernel(
                n_alleles=self.A, seq_length=self.L,
                theta0=params["covar_module.theta"],
                log_var=params["covar_module.log_var"], device=dev)
        except KeyError as e:  # noqa: BLE001
            raise KeyError(
                f"{cfg.params_path} is missing expected key {e}. The pre-fit loader "
                "expects a GeneralProductKernel checkpoint with keys "
                "'covar_module.theta', 'covar_module.log_var', and "
                "'likelihood.second_noise_covar.raw_noise'."
            ) from e
        model = self._EpiK(kernel, train_noise=True, track_progress=True, device=dev)
        model.set_data(train_x, train_y, y_var=train_y_var)
        model.likelihood.second_noise_covar.noise = torch.exp(
            params["likelihood.second_noise_covar.raw_noise"])
        print(f"[fit] loaded hyperparameters from {cfg.params_path}; skipping model.fit(); "
              f"conditioning on {train_x.shape[0]} seqs")
        return model

    def _model_cache_path(self):
        cfg = self.cfg
        if not (cfg.save_model or cfg.load_model):
            return None
        name = cfg.model_name
        if name is None:
            name = (f"{Path(cfg.data_path).stem}_{cfg.kernel}"
                    f"_nc{self.n_condition_used}_seed{cfg.seed}_params.pth")
        elif not name.endswith((".pth", ".pt")):
            name = name + ".pth"
        return Path(cfg.model_dir) / name

    def _underlying_module(self, model):
        """Return the inner GPyTorch module that holds the parameters."""
        import torch.nn as nn
        for name in ("model", "gp", "gp_model", "_gp", "_model", "exact_gp", "module"):
            sub = getattr(model, name, None)
            if isinstance(sub, nn.Module) and hasattr(sub, "covar_module"):
                return sub
        for _, v in vars(model).items():
            if isinstance(v, nn.Module) and hasattr(v, "covar_module"):
                return v
        for _, v in vars(model).items():
            if isinstance(v, nn.Module):
                return v
        raise AttributeError(
            "could not locate the underlying GPyTorch module on EpiK "
            "(looked for an attribute holding a covar_module).")

    def _save_cached_model(self, model, path: Path) -> None:
        inner = self._underlying_module(model)
        keep = ("covar_module", "likelihood", "mean_module")
        sd = {k: v.detach().cpu() for k, v in inner.state_dict().items()
              if k.split(".")[0] in keep}
        if not sd:
            sd = {k: v.detach().cpu() for k, v in inner.state_dict().items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": sd,
            "kernel": self.cfg.kernel,
            "n_alleles": self.A,
            "seq_length": self.L,
            "alphabet": self.alphabet,
            "n_condition": self.n_condition_used,
            "seed": self.cfg.seed,
        }
        torch.save(ckpt, path)
        print(f"[fit] saved trained model to {path} "
              "(reuse it next time with --load-model --model-name).")

    def _load_cached_model(self, train_x, train_y, train_y_var, path: Path):
        ckpt = torch.load(path, map_location="cpu")
        for key, want in (("kernel", self.cfg.kernel), ("seq_length", self.L),
                          ("n_alleles", self.A), ("alphabet", self.alphabet)):
            if ckpt.get(key) != want:
                raise ValueError(
                    f"cached model {path} mismatch on '{key}' "
                    f"(saved {ckpt.get(key)!r} vs current {want!r}); "
                    "drop --load-model or choose a different --model-name to retrain."
                )
        kernel_cls = {"jenga": self._JengaKernel,
                      "connectedness": self._ConnectednessKernel}[self.cfg.kernel]
        model = self._EpiK(kernel_cls(n_alleles=self.A, seq_length=self.L),
                           track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        inner = self._underlying_module(model)
        missing, unexpected = inner.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[fit] [note] load_state_dict: {len(missing)} missing / "
                  f"{len(unexpected)} unexpected keys (usually harmless).")
        print(f"[fit] reused cached model {path}; skipping model.fit() "
              f"(n_condition={train_x.shape[0]}).")
        return model

    def _predict_batched(self, model, x):
        """Batched predict -> flat numpy array (keeps the cross-cov block small)."""
        bs = max(1, self.cfg.predict_batch_size)
        outs = []
        for i in range(0, x.shape[0], bs):
            pdf = model.predict(x[i:i + bs], calc_variance=False)
            outs.append(np.asarray(pdf.values, dtype=float).flatten())
        return np.concatenate(outs) if outs else np.array([], dtype=float)

    def _move_to_device(self, device: str) -> None:
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
            wt = self._validate_wildtype(cfg.wildtype)
            return self._make_background("wildtype", self.L, self.alphabet,
                                         wildtype=wt, device=dev, dtype=DT)
        return self._make_background("uniform", self.L, self.alphabet, device=dev, dtype=DT)

    def _validate_wildtype(self, wt: Optional[str]) -> str:
        """Require an explicit, well-formed WT when background == 'wildtype'.

        There is deliberately NO default and NO inference: the wildtype background
        measures effects relative to a specific reference, so a silently-guessed
        reference would quietly change the meaning of every attribution. The WT must
        be supplied and must be consistent with the data (drawn from the alphabet and
        the same length as the sequences), otherwise we stop rather than produce
        misleading numbers.
        """
        if wt is None or str(wt) == "":
            print("[error] --background wildtype was chosen but no --wildtype sequence "
                  "was given.\n        The wildtype background has no default; pass the "
                  "reference sequence explicitly, e.g. --wildtype VDGV.")
            sys.exit(1)
        wt = str(wt)
        if len(wt) != self.L:
            print(f"[error] --wildtype '{wt}' has length {len(wt)}, but the data "
                  f"sequences have length {self.L}. The wildtype must match the "
                  "sequence length.")
            sys.exit(1)
        bad = sorted({ch for ch in wt if ch not in self.alphabet})
        if bad:
            print(f"[error] --wildtype '{wt}' contains character(s) {bad} that are not "
                  f"in the data alphabet '{self.alphabet}'. The wildtype must use only "
                  "characters that appear in the data.")
            sys.exit(1)
        return wt

    def _resolve_local_targets(self) -> List[str]:
        cfg = self.cfg
        if cfg.local_targets:
            return list(cfg.local_targets)
        targets: List[str] = []
        if cfg.background == "wildtype" and cfg.wildtype:
            targets.append(cfg.wildtype)
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
        N = len(self.seqs)
        n = cfg.n_global_targets
        if n is None or n < 0 or n >= N:
            # default: average the global summaries over the ENTIRE dataset
            return [str(s) for s in self.seqs]
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(N, size=n, replace=False)
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
        elif self._faith_cap is not None and self._faith_cap < self.L:
            print(f"[faith] truncated Faith-Shap (ell={ell} < L={self.L}) with "
                  f"--order-cap {self._faith_cap}: the redistribution tail is capped, so "
                  "Faith-Shap values are for the order-cap truncation of the model "
                  "(an approximation). SII and dividend values are unaffected.")

        positions = [(i,) for i in range(self.L)]
        inter_tuples = _enumerate_interactions(self.L, cfg.interaction_order)
        n_tuples = len(positions) + len(inter_tuples)

        # verbose: one-time echo of what this run is about to compute over
        xs_preview = self._global_target_sample()
        self._echo_config(ell, n_tuples, len(xs_preview))

        result = AnalysisResult(
            config=cfg, alphabet=self.alphabet, seq_length=self.L,
            n_sequences=len(self.seqs), test_r2=self.test_r2,
            wildtype=(cfg.wildtype if cfg.background == "wildtype" else None),
            background_mode=cfg.background, order_max_effective=ell,
            faith_is_dividend=faith_is_dividend,
            n_condition_used=self.n_condition_used,
        )

        # ---- LOCAL: per-target tables ---- #
        local_targets = self._resolve_local_targets()
        print(f"[local] targets: {local_targets}")
        with self._phase("local (all targets)"):
            for ti, tgt in enumerate(local_targets, 1):
                self._log(f"local target {ti}/{len(local_targets)}: '{tgt}'",
                          level="debug")
                with self._phase(f"local[{tgt}]"):
                    result.local[tgt] = self._local_tables(
                        tgt, positions, inter_tuples, ell)

        # ---- GLOBAL: dataset-level summaries ---- #
        xs = xs_preview
        if len(xs) == len(self.seqs):
            print(f"[global] averaging over ALL {len(xs)} sequences "
                  "(a SHAP pass over the full dataset; pass --n-global-targets N "
                  "to subsample for a faster run)")
        else:
            print(f"[global] averaging over a random {len(xs)} of {len(self.seqs)} "
                  "sequences (--n-global-targets)")
        with self._phase("global (all summaries)"):
            result.global_ = self._global_tables(xs, positions, inter_tuples, ell)

        self._result = result
        return result

    def _local_tables(self, tgt, positions, inter_tuples, ell) -> dict:
        shap, pi = self.shap, self.pi
        all_T = positions + inter_tuples
        want_cov = self.cfg.local_covariance

        def diag_sd(cov):
            # posterior sd of each term = sqrt of the covariance diagonal (clamped
            # to guard tiny negative round-off before the sqrt).
            d = torch.diagonal(cov)
            return _to_np(torch.sqrt(torch.clamp(d, min=0.0)))

        if want_cov:
            phi, _ = shap.shap_values(tgt, pi, indices=list(range(self.L)), calc_covar=True)
            sii_T, sii, sii_cov = shap.shap_interaction_values(
                tgt, pi, interactions=all_T, normalize="sii", calc_covar=True)
            fs_T, fs, fs_cov = shap.faith_shap_values(
                tgt, pi, order_max=ell, interactions=all_T, calc_covar=True,
                order_cap=self._faith_cap)
            dv_T, dv, dv_cov = shap.dividend_values(
                tgt, pi, interactions=all_T, calc_covar=True)
            sii_sd, fs_sd, dv_sd = diag_sd(sii_cov), diag_sd(fs_cov), diag_sd(dv_cov)
        else:
            phi = shap.shap_values(tgt, pi, indices=list(range(self.L)))
            sii_T, sii = shap.shap_interaction_values(
                tgt, pi, interactions=all_T, normalize="sii")
            fs_T, fs = shap.faith_shap_values(
                tgt, pi, order_max=ell, interactions=all_T,
                order_cap=self._faith_cap)
            dv_T, dv = shap.dividend_values(tgt, pi, interactions=all_T)
            sii_sd = fs_sd = dv_sd = None

        # each table is (tuples, value, sd)  -- sd is None when covariance is off
        return {
            "shapley_positions": (list(range(self.L)), _to_np(phi), None),
            "sii": (sii_T, _to_np(sii), sii_sd),
            "faith_shap": (fs_T, _to_np(fs), fs_sd),
            "dividend": (dv_T, _to_np(dv), dv_sd),
        }

    def _global_tables(self, xs, positions, inter_tuples, ell) -> dict:
        """Dataset-level summaries via the VECTORIZED (batched-over-targets) SHAP path.

        The three mean summaries -- first-order importance, dividends, SII -- are
        computed with the batched ``*_batch_vectorized`` methods, which share one
        pi-channel build across a whole target chunk instead of rebuilding it per
        target. This is the launch-bound fix: a naive per-target loop leaves the GPU
        ~25% utilized (many tiny kernels), while the batched build collapses that to a
        handful of large ops. The batched methods are verified equal to the per-target
        paths (probe_batch_indices.py, ~1e-12; test_gkt_brute.py still green), so the
        NUMBERS are unchanged -- only the speed differs.

        Faith-Shap: at ``ell >= L`` (the tail is empty) Faith-Shap == the dividends,
        so its summary is read straight off the already-computed batched dividend
        matrix -- exact and free. Only the genuinely truncated case (``ell < L``) still
        routes through the per-target ``global_faith_shap_summary`` (no vectorized
        tail path yet); that case is gated tractable by ``_preflight_faith``.
        """
        shap, pi = self.shap, self.pi
        out = {}
        all_T = positions + inter_tuples
        n_x, n_t = len(xs), len(all_T)
        faith_is_dividend = ell >= self.L

        # ---- first-order position importance: mean_x |phi_i| ----
        self._log(f"-> shap_values_batch_vectorized ({n_x:,} targets, {self.L} positions)",
                  level="debug")
        with self._phase("global.position_importance"):
            phis = shap.shap_values_batch_vectorized(
                xs, pi, indices=list(range(self.L)))            # (n_x, L)
            mean_abs_phi = phis.abs().mean(dim=0)
        out["position_importance"] = (list(range(self.L)), _to_np(mean_abs_phi))

        # ---- dividends: batched matrix A (n_x, n_tuples), then reduce ----
        self._log(f"-> dividend_values_batch_vectorized ({n_x:,} targets, {n_t} tuples)",
                  level="debug")
        with self._phase("global.dividend_summary"):
            dv_T, A = shap.dividend_values_batch_vectorized(xs, pi, interactions=all_T)
            dv_sum = {
                "mean_abs": A.abs().mean(dim=0),
                "mean_signed": A.mean(dim=0),
                "sobol_unnorm": A.var(dim=0, unbiased=False),
            }
        out["dividend_summary"] = (dv_T, {k: _to_np(v) for k, v in dv_sum.items()})

        # ---- SII: batched matrix, then mean_x |SII| ----
        self._log(f"-> interaction_values_batch_vectorized SII ({n_x:,} targets, {n_t} tuples)",
                  level="debug")
        with self._phase("global.sii_summary"):
            sii_T, SII = shap.interaction_values_batch_vectorized(
                xs, pi, interactions=all_T, normalize="sii")    # (n_x, n_tuples)
            SII = SII.abs().mean(dim=0)
        out["sii_summary"] = (sii_T, _to_np(SII))

        # ---- Faith-Shap ----
        self._log(f"-> faith_shap summary ({n_x:,} targets, {n_t} tuples, "
                  f"ell={ell}, cap={self._faith_cap if self._faith_cap is not None else 'L'}, "
                  f"{'== dividends (vectorized)' if faith_is_dividend else 'truncated (per-target)'})",
                  level="debug")
        with self._phase("global.faith_shap_summary"):
            if faith_is_dividend:
                # ell >= L: Faith-Shap IS the dividends (empty redistribution tail),
                # so read the summary off the batched dividend matrix A -- exact, free.
                # (var == sobol_unnorm: both are var(dim=0, unbiased=False).)
                fs_T = dv_T
                fs_sum = {
                    "mean_abs": A.abs().mean(dim=0),
                    "mean_signed": A.mean(dim=0),
                    "var": A.var(dim=0, unbiased=False),
                }
                fs_sum = {k: _to_np(v) for k, v in fs_sum.items()}
            else:
                # truncated Faith-Shap (ell < L): vectorized batch path. It computes
                # every needed dividend (reported tuples + supersets up to order_cap)
                # once across the target chunk, then applies the target-independent
                # Eq. 16 redistribution map. Verified == the per-target loop and to the
                # order_max==L -> dividends anchor (probe_faith_batch.py). The tail is
                # bounded tractable by _preflight_faith via order_cap.
                fs_T, Phi = shap.faith_shap_values_batch_vectorized(
                    xs, pi, order_max=ell, interactions=all_T,
                    order_cap=self._faith_cap)
                fs_sum = {
                    "mean_abs": _to_np(Phi.abs().mean(dim=0)),
                    "mean_signed": _to_np(Phi.mean(dim=0)),
                    "var": _to_np(Phi.var(dim=0, unbiased=False)),
                }
        out["faith_shap_summary"] = (fs_T, fs_sum)

        if self.cfg.global_covariance:
            self._log(f"-> global posterior covariance ({n_x:,} targets)", level="debug")
            with self._phase("global.covariance"):
                out["cov_sd"] = self._global_cov_sd(xs, positions, inter_tuples, ell)
        return out

    def _global_cov_sd(self, xs, positions, inter_tuples, ell) -> dict:
        """Mean posterior sd per term, averaged over the (capped) global target sample.

        Loops the per-target covariance path and averages sqrt(diag(cov)) across
        targets. Gated by ``--global-covariance`` and an explicit ``--n-global-targets``
        cap (see ``_preflight_background``), because this is O(n_tuples^2) per target.
        The result is the dataset-level "typical posterior uncertainty" of each term --
        distinct from the Sobol variance, which is dispersion across sequences.
        """
        shap, pi = self.shap, self.pi
        all_T = positions + inter_tuples
        pos_sd = np.zeros(self.L)
        sii_sd = np.zeros(len(all_T))
        fs_sd = np.zeros(len(all_T))
        dv_sd = np.zeros(len(all_T))

        def diag_sd(cov):
            d = torch.diagonal(cov)
            return _to_np(torch.sqrt(torch.clamp(d, min=0.0)))

        n = len(xs)
        print(f"[global] computing posterior covariance over {n} targets "
              "(--global-covariance)")
        # verbose: tick every 25 targets (finer than the default every-100) with a
        # running rate + ETA, so this long per-target loop shows liveness.
        t0 = time.time()
        tick = 25 if self.cfg.verbose else 100
        for k, x in enumerate(xs, 1):
            _, pcov = shap.shap_values(x, pi, indices=list(range(self.L)), calc_covar=True)
            _, _, scov = shap.shap_interaction_values(
                x, pi, interactions=all_T, normalize="sii", calc_covar=True)
            _, _, fcov = shap.faith_shap_values(
                x, pi, order_max=ell, interactions=all_T, calc_covar=True,
                order_cap=self._faith_cap)
            _, _, dcov = shap.dividend_values(x, pi, interactions=all_T, calc_covar=True)
            pos_sd += diag_sd(pcov)
            sii_sd += diag_sd(scov)
            fs_sd += diag_sd(fcov)
            dv_sd += diag_sd(dcov)
            if k % tick == 0 or k == n:
                if self.cfg.verbose:
                    rate = k / max(1e-9, time.time() - t0)
                    eta = (n - k) / max(1e-9, rate)
                    print(f"[global]   covariance {k}/{n}  "
                          f"({rate:.1f}/s, eta {eta / 60:.1f} min)", flush=True)
                else:
                    print(f"[global]   covariance {k}/{n}", flush=True)
        return {
            "position_mean_sd": pos_sd / n,
            "sii_mean_sd": sii_sd / n,
            "faith_shap_mean_sd": fs_sd / n,
            "dividend_mean_sd": dv_sd / n,
        }

    # ------------------------------------------------------------------ #
    # 5. Output
    # ------------------------------------------------------------------ #
    def save(self, result: AnalysisResult) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._phase("save.csv"):
            self._dump_global_csv(result, out_dir)
            for tgt, tables in result.local.items():
                self._dump_local_csv(tgt, tables, out_dir)

        if cfg.make_plots:
            with self._phase("save.plots"):
                self._plot_global(result, out_dir)
                for tgt, tables in result.local.items():
                    self._plot_local(tgt, tables, out_dir)

        summ = out_dir / "SUMMARY.txt"
        with open(summ, "w") as fh:
            fh.write(_render_summary(result))
        result.out_files.append(str(summ))
        print(f"[save] wrote {len(result.out_files)} files to {out_dir}")

    def _dump_global_csv(self, result, out_dir):
        g = result.global_
        cov = g.get("cov_sd")
        idx, mean_abs = g["position_importance"]
        pos_cols = {"position": idx, "mean_abs_shapley": mean_abs}
        if cov is not None:
            pos_cols["mean_posterior_sd"] = cov["position_mean_sd"]
        pd.DataFrame(pos_cols).to_csv(
            out_dir / "global_position_importance.csv", index=False)
        result.out_files.append(str(out_dir / "global_position_importance.csv"))
        dv_T, dv = g["dividend_summary"]
        sii_T, sii = g["sii_summary"]
        fs_T, fs = g["faith_shap_summary"]
        rows = []
        for i, T in enumerate(dv_T):
            row = {
                "tuple": "x".join(str(p) for p in T),
                "order": len(T),
                "sii_mean_abs": sii[i],
                "faith_shap_mean_abs": fs["mean_abs"][i],
                "dividend_mean_abs": dv["mean_abs"][i],
                "dividend_sobol": dv["sobol_unnorm"][i],
                "dividend_mean_signed": dv["mean_signed"][i],
            }
            if cov is not None:
                # mean posterior sd of each term across the global sample (distinct
                # from dividend_sobol, which is dispersion across sequences).
                row["sii_mean_posterior_sd"] = cov["sii_mean_sd"][i]
                row["faith_shap_mean_posterior_sd"] = cov["faith_shap_mean_sd"][i]
                row["dividend_mean_posterior_sd"] = cov["dividend_mean_sd"][i]
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "global_interactions.csv", index=False)
        result.out_files.append(str(out_dir / "global_interactions.csv"))

    def _dump_local_csv(self, tgt, tables, out_dir):
        sii_T, sii, sii_sd = tables["sii"]
        fs_T, fs, fs_sd = tables["faith_shap"]
        dv_T, dv, dv_sd = tables["dividend"]
        have_cov = sii_sd is not None

        def z(val, sd):
            # attribution in units of its own posterior sd; guard sd == 0.
            return float(val / sd) if (sd is not None and sd > 0) else float("nan")

        rows = []
        for i, T in enumerate(sii_T):
            row = {
                "tuple": "x".join(str(p) for p in T),
                "order": len(T),
                "sii": sii[i],
                "faith_shap": fs[i],
                "dividend": dv[i],
            }
            if have_cov:
                row["sii_sd"] = sii_sd[i]
                row["faith_shap_sd"] = fs_sd[i]
                row["dividend_sd"] = dv_sd[i]
                row["sii_z"] = z(sii[i], sii_sd[i])
                row["faith_shap_z"] = z(fs[i], fs_sd[i])
                row["dividend_z"] = z(dv[i], dv_sd[i])
            rows.append(row)
        fn = out_dir / f"local_{tgt}.csv"
        pd.DataFrame(rows).to_csv(fn, index=False)
        tables["_csv"] = str(fn)

    # ---- plots ---- #
    def _plot_global(self, result, out_dir):
        g = result.global_
        cov = g.get("cov_sd")
        idx, mean_abs = g["position_importance"]
        dv_T, dv = g["dividend_summary"]
        sii_T, sii = g["sii_summary"]
        fs_T, fs = g["faith_shap_summary"]

        fig, axes = plt.subplots(3, 1, figsize=(max(8, len(dv_T) * 0.35), 11))
        pos_err = cov["position_mean_sd"] if cov is not None else None
        axes[0].bar([str(i) for i in idx], mean_abs, yerr=pos_err, capsize=3,
                    color="#c0504d")
        axes[0].set_title("Global position importance  mean|phi_i|")
        axes[0].set_xlabel("position"); axes[0].set_ylabel("mean |phi|")
        labels = ["x".join(map(str, T)) for T in dv_T]
        xpos = np.arange(len(labels)); w = 0.27
        sii_err = cov["sii_mean_sd"] if cov is not None else None
        fs_err = cov["faith_shap_mean_sd"] if cov is not None else None
        dv_err = cov["dividend_mean_sd"] if cov is not None else None
        axes[1].bar(xpos - w, sii, width=w, yerr=sii_err, capsize=2,
                    label="SII |mean|", color="#8064a2")
        axes[1].bar(xpos, fs["mean_abs"], width=w, yerr=fs_err, capsize=2,
                    label=f"Faith-Shap(ell={result.order_max_effective}) |mean|",
                    color="#4f81bd")
        axes[1].bar(xpos + w, dv["mean_abs"], width=w, yerr=dv_err, capsize=2,
                    label="dividend |mean|", color="#9bbb59")
        axes[1].set_xticks(xpos); axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
        it = ("  (error bars = mean posterior sd)" if cov is not None else "")
        axes[1].set_title("Global interaction importance (three indices)" + it)
        axes[1].set_xlabel("position / interaction tuple")
        axes[1].set_ylabel("mean |attribution|")
        axes[1].legend(fontsize=8)
        axes[2].bar(xpos, dv["sobol_unnorm"], color="#2c7fb8")
        axes[2].set_xticks(xpos); axes[2].set_xticklabels(labels, rotation=90, fontsize=7)
        axes[2].set_title("Global Sobol sensitivity  Var_x a(T)  (dividends)")
        axes[2].set_xlabel("position / interaction tuple")
        axes[2].set_ylabel("Sobol sensitivity  Var_x a(T)")
        cap = ("   [error bars: mean posterior sd across the global sample; "
               "calibration depends on the GP posterior]" if cov is not None else "")
        fig.suptitle(f"Global analysis  bg={result.background_mode}  "
                     f"kernel={result.config.kernel}{cap}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fn = out_dir / "global_analysis.png"
        fig.savefig(fn, dpi=130); plt.close(fig)
        result.out_files.append(str(fn))

    def _plot_local(self, tgt, tables, out_dir):
        sii_T, sii, sii_sd = tables["sii"]
        _, fs, fs_sd = tables["faith_shap"]
        _, dv, dv_sd = tables["dividend"]
        labels = ["x".join(map(str, T)) for T in sii_T]
        xpos = np.arange(len(labels)); w = 0.27
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 5))
        ax.bar(xpos - w, sii, width=w, yerr=sii_sd, capsize=2, label="SII", color="#8064a2")
        ax.bar(xpos, fs, width=w, yerr=fs_sd, capsize=2, label="Faith-Shap", color="#4f81bd")
        ax.bar(xpos + w, dv, width=w, yerr=dv_sd, capsize=2, label="dividend", color="#9bbb59")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        eb = ("  (error bars = posterior sd)" if sii_sd is not None else "")
        ax.set_title(f"Local attribution for '{tgt}'  (bg={self.cfg.background}){eb}")
        ax.set_xlabel("position / interaction tuple")
        ax.set_ylabel("signed attribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fn = out_dir / f"local_{tgt}.png"
        fig.savefig(fn, dpi=130); plt.close(fig)
        tables["_png"] = str(fn)

    # ------------------------------------------------------------------ #
    # orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> AnalysisResult:
        with self._phase("TOTAL"):
            with self._phase("load"):
                self.load()
            self._preflight_background()   # fail fast before the expensive fit
            self._preflight_faith()        # resolve/validate the Faith-Shap tail cap
            with self._phase("fit"):
                self.fit()
            with self._phase("interpret"):
                result = self.interpret()
            with self._phase("save"):
                self.save(result)
        return result

    def run_fit_only(self) -> "SequenceFunctionAnalysis":
        """Load + fit, without the standard interpret/save pass.

        Convenience for downstream drivers (e.g. GB1_fitness.py) that want the
        fitted GP and ShapKernelTrickComputer but will build their own background
        and attributions. Also sets ``self.pi`` from the configured background so
        callers can use it directly.

        Note: no ``_preflight_faith`` here -- this path never calls Faith-Shap
        (GB1_fitness.py uses only ``dividend_values``), so the tail cap is irrelevant
        and ``self._faith_cap`` stays at its ``__init__`` default of None.
        """
        self.load()
        self._preflight_background()   # fail fast before the expensive fit
        self.fit()
        self.pi = self._build_background()
        return self

    def _preflight_background(self) -> None:
        """Validate background prerequisites right after load(), before fitting.

        For background == 'wildtype' this enforces that a valid --wildtype was
        supplied (see _validate_wildtype). Also hard-stops the intractable
        --global-covariance-without-a-cap combination. Runs early so bad
        configurations stop in seconds rather than after training the GP.
        """
        if self.cfg.background == "wildtype":
            self._validate_wildtype(self.cfg.wildtype)

        # Global covariance loops the O(n^2)-per-target covariance path over every
        # global target. That is fine for a capped sample (e.g. --n-global-targets
        # 300) but intractable over the full dataset, so require an explicit cap.
        if self.cfg.global_covariance:
            n = self.cfg.n_global_targets
            N = len(self.seqs)
            is_all = (n is None or n < 0 or n >= N)
            if is_all:
                print("[error] --global-covariance needs a finite --n-global-targets "
                      f"(e.g. 300). Computing the posterior covariance over all {N} "
                      "sequences is intractable (the covariance path is O(n_tuples^2) "
                      "per target). Re-run with, say, --n-global-targets 300, or drop "
                      "--global-covariance to keep the mean-only global summaries.")
                sys.exit(1)
            if n > 5000:
                print(f"[warn] --global-covariance with --n-global-targets {n} will be "
                      "slow (covariance is computed per target). Consider a smaller cap "
                      "if this is just a check run.")

    def _preflight_faith(self) -> None:
        """Resolve + validate the Faith-Shap tail cap before fitting.

        Truncated Faith-Shap (ell = min(order_max, L) < L) is NOT the dividends
        truncated at ell: each reported <=ell term absorbs the higher-order epistasis
        above it, so computing it EXACTLY enumerates every superset dividend up to
        order L (~2^(L-ell) tuples). That is intractable for long sequences even though
        the dividends of the same low-order tuples are cheap. ``--order-cap`` bounds
        that redistribution tail; the result is Faith-Shap of the order-cap truncation
        of the model.

        This resolves ``self._faith_cap`` (the cap forwarded to every Faith-Shap call
        in interpret()) and hard-stops the intractable "ell < L, no cap, tail runs to
        L" case -- mirroring how _preflight_background refuses the intractable
        --global-covariance combination. Exact runs (small L, or ell >= L where
        Faith-Shap == dividends and the tail is empty) are left untouched: _faith_cap
        stays None, meaning cap = L inside the shap module.
        """
        from math import comb
        L = self.L
        ell = min(self.cfg.order_max, L)
        self._faith_cap = None
        if ell >= L:
            self._log(f"Faith-Shap: ell={ell} >= L={L}; tail empty (== dividends), "
                      "no cap needed", level="debug")
            return                          # == dividends; empty tail, cap irrelevant

        # Rough upper bound on the number of distinct superset dividends the tail
        # would enumerate (sum of C(L, k) over the tail orders). A heuristic ceiling,
        # not an exact count of the per-tuple work, but enough to catch the blowup.
        THRESH = 200_000
        cap = self.cfg.order_cap
        if cap is not None:
            cap = int(cap)
            if cap < ell or cap > L:
                print(f"[error] --order-cap must be in [order_max, L] = [{ell}, {L}]; "
                      f"got {cap}. (order_cap bounds the Faith-Shap redistribution tail; "
                      "it cannot be below order_max or above the sequence length.)")
                sys.exit(1)
            n_tail = sum(comb(L, k) for k in range(ell + 1, cap + 1))
            self._log(f"Faith-Shap: ell={ell} < L={L}, --order-cap={cap}; tail "
                      f"enumerates ~{n_tail:,} superset dividends", level="debug")
            if n_tail > THRESH:
                print(f"[warn] --order-cap {cap} enumerates ~{n_tail:,} superset "
                      f"dividends for Faith-Shap (ell={ell}, L={L}); this will be slow. "
                      "Lower --order-cap for a faster pass.")
            self._faith_cap = cap
            return

        # cap not given on a truncated run: the exact tail runs to L.
        n_exact = sum(comb(L, k) for k in range(ell + 1, L + 1))
        if n_exact <= THRESH:
            self._log(f"Faith-Shap: ell={ell} < L={L}, no --order-cap; exact tail "
                      f"~{n_exact:,} superset dividends (under threshold, kept exact)",
                      level="debug")
            self._faith_cap = None          # cheap enough; keep it exact (cap = L)
            return
        print(f"[error] truncated Faith-Shap (order_max={ell} < L={L}) with no "
              f"--order-cap would enumerate ~{n_exact:,} higher-order superset "
              "dividends (the exact tail runs to order L). This is intractable.\n"
              f"        Re-run with an explicit --order-cap (e.g. --order-cap {ell + 1} "
              f"to keep one order of redistribution), or --order-max {L} for exact "
              "dividends, which need no tail. SII and dividend outputs do not use the "
              "tail and are always available.")
        sys.exit(1)


# =========================================================================== #
# small helpers  (shared with downstream drivers)
# =========================================================================== #

def _enumerate_interactions(L: int, order: int) -> List[Tuple[int, ...]]:
    """All interaction tuples of size 2..order (singletons handled separately)."""
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
    fit_mode = ("pre-fit params (no model.fit)" if c.params_path
                else f"fit on subset  n_iter={c.n_iter}")
    if result.faith_is_dividend:
        faith_line = (f"Faith-Shap ell:  {result.order_max_effective}"
                      "   [== dividends: ell >= L]")
    elif c.order_cap is not None:
        faith_line = (f"Faith-Shap ell:  {result.order_max_effective}"
                      f"   [truncated; order_cap={c.order_cap} => approximation]")
    else:
        faith_line = (f"Faith-Shap ell:  {result.order_max_effective}"
                      "   [truncated; exact tail to L]")
    lines = [
        "Sequence-function analysis summary",
        "=" * 40,
        f"data:            {c.data_path}",
        f"sequences:       {result.n_sequences}   L={result.seq_length}   "
        f"alphabet='{result.alphabet}'",
        f"kernel:          {c.kernel}   {fit_mode}",
        f"conditioned on:  {result.n_condition_used} sequences   "
        f"(exact-GP mem ~ O(n_condition^2))",
        f"held-out R2:     {result.test_r2 if result.test_r2 is not None else 'n/a (all-data fit)'}",
        f"background:      {result.background_mode}"
        + (f"  (WT={result.wildtype})" if result.wildtype else ""),
        faith_line,
        f"local targets:   {list(result.local.keys())}",
        f"covariance:      local={'on' if c.local_covariance else 'off'}  "
        f"global={'on' if c.global_covariance else 'off'}"
        + ("   [error bars/sd = posterior sd; calibration depends on the GP posterior]"
           if (c.local_covariance or c.global_covariance) else ""),
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
    n_condition = (None if (args.n_condition is not None and args.n_condition < 0)
                   else args.n_condition)
    return AnalysisConfig(
        data_path=args.data,
        seq_col=args.seq_col, y_col=args.y_col, yvar_col=args.yvar_col,
        alphabet=args.alphabet,
        kernel=args.kernel, ptrain=args.ptrain, n_iter=args.n_iter,
        learning_rate=args.lr, seed=args.seed, device=resolve_device(args.gpu),
        n_condition=n_condition, params_path=args.params_path,
        predict_batch_size=args.predict_batch_size, n_r2_eval=args.n_r2_eval,
        model_dir=args.model_dir, model_name=args.model_name,
        load_model=args.load_model, save_model=args.save_model,
        background=args.background, wildtype=args.wildtype,
        order_max=args.order_max, order_cap=args.order_cap,
        interaction_order=args.interaction_order,
        n_global_targets=args.n_global_targets, n_top_binders=args.n_top_binders,
        local_targets=(args.targets.split(",") if args.targets else None),
        local_covariance=(not args.no_local_covariance),
        global_covariance=args.global_covariance,
        out_dir=args.out_dir, make_plots=(not args.no_plot),
        verbose=args.verbose,
        project_root=args.project_root, epik_flavor=args.epik_flavor,
    )


def add_common_args(ap: argparse.ArgumentParser, data_required: bool = True) -> None:
    """Register the shared data/model/interpretation args.

    Factored out so GB1_fitness.py can reuse exactly the same fit/interpret flags
    without duplicating them. Pass ``data_required=False`` when the caller supplies
    its own default (e.g. GB1_fitness.py defaults --data to <root>/data/gb1.csv).
    """
    ap.add_argument("--data", required=data_required, default=None,
                    help="path to the sequence-function CSV")
    ap.add_argument("--seq-col", default="seq")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--yvar-col", default="y_var")
    ap.add_argument("--alphabet", default=None, help="override; default infers from data")
    ap.add_argument("--kernel", default="jenga", choices=["jenga", "connectedness"])
    ap.add_argument("--ptrain", type=float, default=0.8,
                    help="fraction to condition on; only used when --n-condition is negative")
    ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    # memory / conditioning
    ap.add_argument("--n-condition", type=int, default=2000,
                    help="HARD cap on GP conditioning points (exact-GP mem ~ O(n^2)). "
                         "Pass a negative value to disable the cap and use --ptrain.")
    ap.add_argument("--params-path", default=None,
                    help="pre-optimized hyperparameter .pth (GeneralProduct checkpoint); "
                         "if set, model.fit() is SKIPPED")
    ap.add_argument("--predict-batch-size", type=int, default=5000)
    ap.add_argument("--n-r2-eval", type=int, default=2000)
    # model caching
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--load-model", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    # interpretation
    ap.add_argument("--background", default="uniform",
                    choices=["uniform", "wildtype", "empirical"])
    ap.add_argument("--wildtype", default=None,
                    help="WT sequence for background='wildtype' (required for that "
                         "background; validated against the data alphabet/length)")
    ap.add_argument("--order-max", type=int, default=5,
                    help="Faith-Shap max interaction order (ell); >=L => dividends")
    ap.add_argument("--order-cap", type=int, default=None,
                    help="highest superset order Faith-Shap folds into its <=order_max "
                         "terms (the redistribution tail). Default: L (exact) when the "
                         "tail is small enough, otherwise REQUIRED for order_max < L. On "
                         "long sequences set a low value (e.g. order_max+1) for a "
                         "tractable, truncated approximation. Only affects Faith-Shap "
                         "when order_max < L; SII and dividends never use the tail.")
    ap.add_argument("--interaction-order", type=int, default=2,
                    help="enumerate interaction tuples up to this order")
    ap.add_argument("--n-global-targets", type=int, default=None,
                    help="how many sequences the GLOBAL summaries average over. "
                         "Omitted (default) => the ENTIRE dataset; set an int (e.g. "
                         "300) to average over that many randomly chosen sequences "
                         "for a faster pass. A negative value also means all.")
    ap.add_argument("--no-local-covariance", action="store_true",
                    help="skip the posterior covariance of the LOCAL attributions "
                         "(by default it is computed: adds sd + z columns and error "
                         "bars). Cheap for a few targets; disable only for speed.")
    ap.add_argument("--global-covariance", action="store_true",
                    help="also compute the posterior covariance for the GLOBAL "
                         "summaries (per-term mean posterior sd). REQUIRES a finite "
                         "--n-global-targets (e.g. 300); it is intractable over the "
                         "full dataset and will hard-stop without a cap.")
    ap.add_argument("--n-top-binders", type=int, default=3)
    ap.add_argument("--targets", default=None,
                    help="comma-separated explicit local targets (overrides default)")
    # output / bootstrap
    ap.add_argument("--out-dir", default="output/analysis")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="extra timing / progress / config diagnostics. Adds per-phase "
                         "wall-clock timers, a heartbeat around each global block "
                         "(which method, over how many targets/tuples, how long), "
                         "per-target progress in the --global-covariance loop, and a "
                         "one-time echo of the resolved run shape. Purely additive: "
                         "changes no outputs or numbers, only what is printed.")
    ap.add_argument("--project-root", default=None,
                    help="project root (default: inferred; contains modules/ and epik/)")
    ap.add_argument("--epik-flavor", default="epik", choices=["epik", "epik_dev"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = build_config_from_args(args)
    result = SequenceFunctionAnalysis(cfg).run()
    print("\n" + _render_summary(result))
    return result


if __name__ == "__main__":
    main()