#!/usr/bin/env python3
"""
test_kernel_trick_computer.py
=============================
One standalone script that verifies ``KernelTrickComputer`` -- the consolidation of
what used to be a pytest suite (``conftest.py`` + ``pytest.ini`` +
``test_correctness.py`` + ``test_equivalence.py``), the shared helpers
(``kt_test_utils.py``), and the separate ``benchmark_compare.py`` timing tool. It is
meant to be run occasionally, like the other ``test_gkt_*`` scripts, rather than under
pytest.

It does three things, all against ONE small fitted synthetic model:

  1. CORRECTNESS -- the refactored tool vs *independent* ground truth: brute-force
     enumeration over the full sequence space, the exact projection matrix P, and
     empirical covariance from posterior sampling, plus one targeted guard per
     refactor optimisation.
  2. EQUIVALENCE -- original vs refactored across the whole public API (means,
     covariances, expectations, metric matrices). Requires the ORIGINAL
     implementation, which now lives in ``modules/old/kernel_trick_computer.py``. If
     it cannot be found, the equivalence section is skipped (correctness still runs).
  3. BENCHMARK (``--benchmark``) -- times every representative operation on both
     implementations, reports the refactored speedup and any output difference, and
     writes timing/parity graphs + a CSV/Markdown summary.

All graphics go to ``<tests>/output/kernel_trick_computer_test/`` by default
(override with ``--output-dir``).

Usage
-----
    python test_kernel_trick_computer.py                 # correctness + equivalence
    python test_kernel_trick_computer.py --keops         # also the slow KeOps check
    python test_kernel_trick_computer.py --no-graphs      # skip PNG emission
    python test_kernel_trick_computer.py --benchmark      # timing/parity comparison
    python test_kernel_trick_computer.py --seq-length 5 --alphabet ABC --device cpu
    python test_kernel_trick_computer.py --real-data      # optional; needs KTC_REAL_* env

Tolerances default to atol=rtol=1e-4, ve_rtol=1e-2 and can be overridden with
--atol/--rtol/--ve-rtol or the KTC_ATOL/KTC_RTOL/KTC_VE_RTOL env vars.

Requires the same stack as the notebooks: torch, epik, gpytorch, linear_operator,
pykeops (and, for the equivalence/benchmark sections, the original implementation in
modules/old/). Missing heavy dependencies are reported cleanly rather than crashing.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# =========================================================================== #
# Path bootstrap  (running a script does not load conftest, so do it ourselves)
# =========================================================================== #

def _bootstrap_paths() -> Path:
    """Put the repo root (dir containing epik/ and modules/) and modules/ on sys.path.

    Without this, ``import epik`` fails with "No module named 'epik'" when the script
    is run directly. Returns the resolved repo root.
    """
    here = Path(__file__).resolve().parent          # .../tests
    root = None
    for cand in [here, *here.parents]:
        if (cand / "epik").is_dir() and (cand / "modules").is_dir():
            root = cand
            break
    if root is None:
        root = here.parent
    for p in (str(root / "modules"), str(root), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("KTC_MODULES_DIR", str(root / "modules"))
    return root


_ROOT = _bootstrap_paths()


# =========================================================================== #
# Implementation loading  (original now lives in modules/old/)
# =========================================================================== #

ORIG_FILENAME = "kernel_trick_computer.py"
REFAC_FILENAME = "kernel_trick_computer_refactored.py"


def _candidate_module_dirs() -> List[Path]:
    """Directories to search for the implementation files, in priority order.

    Includes ``modules/old`` (and a couple of variants) because the ORIGINAL
    implementation was moved there; the refactored file stays in ``modules/``.
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    dirs: List[Path] = []
    env = os.environ.get("KTC_MODULES_DIR")
    if env:
        dirs += [Path(env), Path(env) / "old"]
    dirs += [
        here, here.parent,
        here.parent / "modules", here.parent / "modules" / "old",
        here.parent / "old",
        cwd, cwd / "modules", cwd / "modules" / "old", cwd / "old",
        cwd.parent, cwd.parent / "modules", cwd.parent / "modules" / "old",
    ]
    seen, unique = set(), []
    for d in dirs:
        rd = d.resolve()
        if rd not in seen:
            seen.add(rd)
            unique.append(rd)
    return unique


def _find_file(filename: str, explicit_env: Optional[str]) -> Optional[Path]:
    """Locate a source file; return None if not found (callers decide if that's fatal)."""
    if explicit_env and os.environ.get(explicit_env):
        p = Path(os.environ[explicit_env]).resolve()
        return p if p.exists() else None
    for d in _candidate_module_dirs():
        p = d / filename
        if p.exists():
            return p.resolve()
    return None


def _load_module_from_path(mod_name: str, path: Path):
    """Import a .py file under a private module name, isolated from the canonical name."""
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_implementations(require_original: bool = False):
    """Return ``(original_module_or_None, refactored_module)``.

    The refactored file is required; the original is optional (it lives in
    ``modules/old/`` after the move). Set ``require_original=True`` to make a missing
    original an error instead of a None return.
    """
    refac_path = _find_file(REFAC_FILENAME, "KTC_REFAC_PATH")
    if refac_path is None:
        searched = "\n  ".join(str(d) for d in _candidate_module_dirs())
        raise FileNotFoundError(
            f"Could not locate '{REFAC_FILENAME}'. Set KTC_MODULES_DIR or "
            f"KTC_REFAC_PATH. Searched:\n  {searched}"
        )
    refac = _load_module_from_path("ktc_refac_impl", refac_path)

    orig_path = _find_file(ORIG_FILENAME, "KTC_ORIG_PATH")
    if orig_path is None:
        if require_original:
            raise FileNotFoundError(
                f"Could not locate the ORIGINAL '{ORIG_FILENAME}'. It was moved to "
                f"modules/old/; set KTC_ORIG_PATH to point at it."
            )
        return None, refac
    orig = _load_module_from_path("ktc_orig_impl", orig_path)
    return orig, refac


def _patch_epik_loader() -> None:
    """Neutralise ``epik_loader.ensure_epik_loaded`` once epik is importable.

    The implementations call it inside ``_encode_subseqs``; in this layout epik is
    already on the path, and the loader's source-tree hunt would otherwise raise
    FileNotFoundError. Mirrors the old conftest shim.
    """
    try:
        import epik  # noqa: F401
    except Exception:
        return
    try:
        import epik_loader
        epik_loader.ensure_epik_loaded = lambda *a, **k: None
    except Exception:
        pass


def try_import_projection_matrix():
    """Return the user's ``ProjectionMatrix`` class, or None if not importable."""
    for d in _candidate_module_dirs():
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        from projection_mtx import ProjectionMatrix  # type: ignore
        return ProjectionMatrix
    except Exception:
        return None


# =========================================================================== #
# Fixture + model construction
# =========================================================================== #

@dataclass
class Fixture:
    """Everything a check needs about one fitted model + its full sequence space."""
    model: Any
    alphabet: str
    seq_length: int
    n_alleles: int
    X: torch.Tensor           # encoded *full* sequence space, (N, L*A)
    all_seqs: List[str]
    preds: torch.Tensor       # posterior mean over the full space, (N,)
    device: torch.device
    label: str = "synthetic"
    enumerable: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _accepts_device(cls) -> bool:
    try:
        import inspect
        return "device" in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _predict_full(model, X: torch.Tensor, batch_size: int = 5000) -> torch.Tensor:
    """Posterior mean over X, batched; also populates the GP caches the KTC reads."""
    chunks = []
    for i in range(0, X.shape[0], batch_size):
        df = model.predict(X[i:i + batch_size], calc_variance=False)
        vals = df.values if hasattr(df, "values") else np.asarray(df)
        chunks.append(torch.as_tensor(vals.flatten(), dtype=torch.float32))
    return torch.cat(chunks, dim=0)


def build_synthetic_model(seq_length: int = 5, alphabet: str = "ABC", seed: int = 0,
                          n_iter: Optional[int] = None, learning_rate: float = 0.1,
                          device: Optional[str] = None) -> Fixture:
    """Small, fully-enumerable synthetic GP fixture (JengaKernel, random linear truth).

    Because the whole space is enumerated, brute-force / sampling / projection
    references are all valid. The correctness identities are exact statements about the
    estimator (both sides use the same model), so a short fit is fine -- ``n_iter`` only
    affects speed.
    """
    from epik.epik.utils import encode_seqs, split_training_test
    from epik.epik.model import EpiK
    from epik.epik.kernel import JengaKernel

    if n_iter is None:
        n_iter = int(os.environ.get("KTC_FIT_ITERS", "25"))
    device = torch.device(device or os.environ.get("KTC_DEVICE", "cpu"))
    _seed_everything(seed)

    A = len(alphabet)
    all_seqs = ["".join(p) for p in itertools.product(alphabet, repeat=seq_length)]
    X = encode_seqs(all_seqs, alphabet=alphabet)
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)
    X = X.to(torch.float32)

    effects = torch.rand(seq_length * A)
    y = X @ effects
    y_var = torch.rand(len(all_seqs))

    train_x, train_y, test_x, test_y, train_y_var = split_training_test(
        X, y, y_var, ptrain=0.5, dtype=torch.float32)

    kernel = JengaKernel(n_alleles=A, seq_length=seq_length)
    model = EpiK(kernel, track_progress=False, device=str(device)) if _accepts_device(EpiK) \
        else EpiK(kernel, track_progress=False)
    model.set_data(train_x, train_y, train_y_var)
    if n_iter > 0:
        model.fit(n_iter=n_iter, learning_rate=learning_rate)

    preds = _predict_full(model, X)
    return Fixture(model=model, alphabet=alphabet, seq_length=seq_length, n_alleles=A,
                   X=X.to(device), all_seqs=all_seqs, preds=preds.to(device),
                   device=device, label="synthetic", enumerable=True)


def build_real_data_model(device: Optional[str] = None, seed: int = 0) -> Fixture:
    """Optional real-data hook, driven by KTC_REAL_* env vars (mirrors the notebooks).

    KTC_REAL_CSV / KTC_REAL_ALPHABET / KTC_REAL_L / KTC_REAL_KERNEL / KTC_REAL_PARAMS /
    KTC_REAL_PTRAIN / KTC_REAL_ENUMERATE. ``enumerable`` is True only when the full
    A**L space was enumerated (required for the brute-force identity).
    """
    import pandas as pd
    from epik.epik.utils import encode_seqs, split_training_test
    from epik.epik.model import EpiK
    from epik.epik import kernel as kernel_mod

    csv_path = os.environ.get("KTC_REAL_CSV")
    if not csv_path:
        raise FileNotFoundError("No real-data CSV provided (set KTC_REAL_CSV).")
    alphabet = os.environ.get("KTC_REAL_ALPHABET")
    if not alphabet:
        raise ValueError("Real-data alphabet required (set KTC_REAL_ALPHABET).")
    seq_length = int(os.environ.get("KTC_REAL_L", "0"))
    if seq_length <= 0:
        raise ValueError("Real-data sequence length required (set KTC_REAL_L).")
    kernel_name = os.environ.get("KTC_REAL_KERNEL", "ConnectednessKernel")
    params_path = os.environ.get("KTC_REAL_PARAMS")
    ptrain = float(os.environ.get("KTC_REAL_PTRAIN", "0.02"))
    enumerate_full_space = os.environ.get("KTC_REAL_ENUMERATE") == "1"
    device = torch.device(device or os.environ.get("KTC_DEVICE", "cpu"))
    _seed_everything(seed)
    A = len(alphabet)

    data = pd.read_csv(csv_path, index_col=0)
    X = encode_seqs(data.index.values, alphabet=alphabet)
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)
    X = X.to(torch.float32)
    y = data["y"].values
    y_var = data["y_var"].values if "y_var" in data.columns else np.ones_like(y)

    train_x, train_y, test_x, test_y, train_y_var = split_training_test(
        X, y, y_var, ptrain=ptrain, dtype=torch.float32)

    KernelCls = getattr(kernel_mod, kernel_name)
    if params_path:
        params = torch.load(params_path, map_location=str(device))
        kernel = KernelCls(n_alleles=A, seq_length=seq_length,
                           theta0=params["covar_module.theta"],
                           log_var=params["covar_module.log_var"])
        model = EpiK(kernel, train_noise=True, track_progress=False)
        model.set_data(train_x, train_y, y_var=train_y_var)
        if "likelihood.second_noise_covar.raw_noise" in params:
            model.likelihood.second_noise_covar.noise = torch.exp(
                params["likelihood.second_noise_covar.raw_noise"])
    else:
        kernel = KernelCls(n_alleles=A, seq_length=seq_length)
        model = EpiK(kernel, track_progress=False)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100, learning_rate=0.1)

    if enumerate_full_space:
        all_seqs = ["".join(p) for p in itertools.product(alphabet, repeat=seq_length)]
        Xfull = encode_seqs(all_seqs, alphabet=alphabet)
        if not torch.is_tensor(Xfull):
            Xfull = torch.as_tensor(Xfull)
        Xfull = Xfull.to(torch.float32)
        preds = _predict_full(model, Xfull)
        enumerable = True
    else:
        all_seqs = list(data.index.values)
        Xfull = X
        preds = _predict_full(model, Xfull)
        enumerable = False

    return Fixture(model=model, alphabet=alphabet, seq_length=seq_length, n_alleles=A,
                   X=Xfull.to(device), all_seqs=all_seqs, preds=preds.to(device),
                   device=device, label=f"real:{Path(csv_path).stem}", enumerable=enumerable)


# =========================================================================== #
# Use-case generation
# =========================================================================== #

def generate_use_case(seq_length: int, alphabet: str, num_subseqs: int = 200,
                      rng: Optional[random.Random] = None,
                      gen: Optional[torch.Generator] = None
                      ) -> Tuple[torch.Tensor, float, List[str]]:
    """Seeded port of the notebooks' generate_use_case.

    Returns (pi [L,A] row-stochastic, eta in (0.01,0.99), subseqs [num_subseqs strings]).
    """
    rng = rng or random.Random()
    A = len(alphabet)
    pi = torch.rand(seq_length, A, generator=gen) if gen is not None \
        else torch.rand(seq_length, A)
    pi = pi / pi.sum(dim=1, keepdim=True)
    eta = rng.uniform(0.01, 0.99)
    subseqs = ["".join("-" if rng.random() < 0.5 else rng.choice(alphabet)
                       for _ in range(seq_length))
               for _ in range(num_subseqs)]
    return pi.to(torch.float32), eta, subseqs


def random_pis(seq_length: int, n_alleles: int, batch: int = 4, seed: int = 0,
               device: Optional[torch.device] = None) -> torch.Tensor:
    """A batch of random row-stochastic pis, shape (batch, L, A)."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(batch, seq_length, n_alleles, generator=g)
    p = (p / p.sum(dim=-1, keepdim=True)).to(torch.float32)
    return p.to(device) if device is not None else p


# =========================================================================== #
# Brute-force references (from variance_explained.ipynb)
# =========================================================================== #

def compute_seq_probs(X: torch.Tensor, pis: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Probability of each enumerated sequence under per-position pis. Returns (N,)."""
    L, A = pis.shape
    pis = pis.to(X.device)
    probs_pos = torch.sum(X.reshape(-1, L, A) * pis, dim=-1).clamp(min=eps)   # (N, L)
    log_seq = torch.sum(torch.log(probs_pos), dim=-1)
    log_seq = log_seq - torch.logsumexp(log_seq, dim=0)
    out = torch.exp(log_seq)
    total = out.sum().item()
    assert np.isfinite(total) and abs(total - 1.0) < 1e-6, f"seq probs do not sum to 1: {total}"
    return out


def predict_from_truncated(seqs: torch.Tensor, encoded_subseqs: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
    """Truncated/linear-model predictions over enumerated sequences. Returns (N,)."""
    dtype = seqs.dtype
    compat = (encoded_subseqs[..., :-1] + encoded_subseqs[..., -1:]).to(dtype)  # (r, L, A)
    Phi = torch.einsum("rla,tla->rtl", compat, seqs.to(dtype))                  # (r, N, L)
    indicator = torch.prod(Phi, dim=-1)                                        # (r, N)
    return weights.to(dtype) @ indicator                                       # (N,)


def naive_ve_mse(preds: torch.Tensor, w_preds: torch.Tensor,
                 seq_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brute-force (variance_explained, mse) under seq_probs (matches the notebook)."""
    diff = w_preds - preds
    num = torch.sum(diff ** 2 * seq_probs) - torch.sum(diff * seq_probs) ** 2
    den = torch.sum(seq_probs * preds ** 2) - torch.sum(seq_probs * preds) ** 2
    ve = 1.0 - num / den
    return ve, num


# =========================================================================== #
# Projection / sampling references (from verify_kernel_trick.ipynb)
# =========================================================================== #

def get_test_train_K(model, test_x, train_x) -> torch.Tensor:
    n_test = test_x.shape[0]
    full = model.gp.forward(torch.cat([test_x, train_x])).covariance_matrix
    return full[:n_test, n_test:]


def get_test_test_K(model, test_x) -> torch.Tensor:
    return model.gp.forward(test_x).covariance_matrix


def get_S(model) -> torch.Tensor:
    return model.gp.prediction_strategy.covar_cache


def mean_via_projection(po, preds, pis, eta, subseqs) -> torch.Tensor:
    """Reference gauge-fixed weights: P @ preds (the exact projection)."""
    P = po.compute_P(pis, eta, subseqs)
    return P.to(preds.dtype) @ preds


def cov_via_thm6(model, po, pis, eta, subseqs, all_x) -> torch.Tensor:
    """Reference covariance via 'thm 6': P K P^T - (P K_tt S)(P K_tt S)^T."""
    P = po.compute_P(pis, eta, subseqs)
    Ktt = get_test_train_K(model, all_x, model.X)
    K = get_test_test_K(model, all_x)
    S = get_S(model)
    P = P.to(K.dtype)
    PKS = P @ Ktt @ S
    return P @ K @ P.T - PKS @ PKS.T


def approx_cov_by_sampling(model, po, pis, eta, subseqs, all_x,
                           num_samples: int = 20000, seed: int = 0) -> torch.Tensor:
    """Empirical covariance of the projected posterior samples."""
    torch.manual_seed(seed)
    P = po.compute_P(pis, eta, subseqs)
    samples = model.get_posterior(all_x, calc_variance=True).sample(
        sample_shape=torch.Size([num_samples]))
    P = P.to(samples.dtype)
    p_samples = (P @ samples.T).T
    mean = p_samples.mean(dim=0)
    centered = p_samples - mean
    return centered.T @ centered / (p_samples.size(0) - 1)


# =========================================================================== #
# Adapters: a uniform interface over the two differing public APIs
# =========================================================================== #

class _BaseAdapter:
    impl_name = "base"

    def __init__(self, ktc):
        self.ktc = ktc

    def e_map(self, pi):                    return self.ktc.E_MAP(pi)
    def e_map2(self, pi, keops=False):      return self.ktc.E_MAP_2(pi, keops=keops)
    def e_trunc(self, pi, w, subseqs):      return self.ktc.E_TRUNC(pi, w, subseqs)
    def e_trunc2(self, pi, w, subseqs):     return self.ktc.E_TRUNC_2(pi, w, subseqs)
    def e_map_trunc(self, pi, w, subseqs):  return self.ktc.E_MAP_TRUNC(pi, w, subseqs)
    def variance_explained(self, pi, w, subseqs, **kw): return self.ktc.variance_explained(pi, w, subseqs, **kw)
    def mse(self, pi, w, subseqs, **kw):    return self.ktc.MSE(pi, w, subseqs, **kw)
    def encode_subseqs(self, subseqs):      return self.ktc._encode_subseqs(subseqs)

    @property
    def linear_subseqs(self):  return self.ktc.linear_subseqs
    @property
    def constant_subseq(self): return self.ktc.constant_subseq

    def weights(self, pis, eta, subtype, subseqs=None, calc_covar=False):
        raise NotImplementedError

    def across(self, pis, eta, subtype, subseqs=None, calc_covar=False):
        raise NotImplementedError

    def ve_matrix(self, pis):  raise NotImplementedError
    def mse_matrix(self, pis): raise NotImplementedError
    def ve_diag(self, pis):    raise NotImplementedError


class OrigAdapter(_BaseAdapter):
    impl_name = "original"

    def weights(self, pis, eta, subtype, subseqs=None, calc_covar=False):
        k = self.ktc
        if subtype == "linear":
            return k.gauge_fixed_posterior_linear(pis, eta, calc_covar=calc_covar)
        if subtype == "constant":
            return k.gauge_fixed_posterior_constant(pis, eta, calc_covar=calc_covar)
        if subtype == "both":
            return k.gauge_fixed_posterior_constant_linear(pis, eta, calc_covar=calc_covar)
        if subtype == "custom":
            return k.gauge_fixed_posterior(pis, eta, subseqs, calc_covar=calc_covar)
        raise ValueError(subtype)

    def across(self, pis, eta, subtype, subseqs=None, calc_covar=False):
        k = self.ktc
        if subtype == "linear":
            return k.gauge_fixed_posterior_various_pis_linear(pis, eta, calc_covar=calc_covar)
        if subtype == "constant":
            return k.gauge_fixed_posterior_various_pis_constant(pis, eta, calc_covar=calc_covar)
        if subtype == "both":
            return k.gauge_fixed_posterior_various_pis_constant_linear(pis, eta, calc_covar=calc_covar)
        if subtype == "custom":
            return k.gauge_fixed_posterior_various_pis(pis, eta, subseqs, calc_covar=calc_covar)
        raise ValueError(subtype)

    def ve_matrix(self, pis):  return self.ktc.compute_ve_matrix(pis)
    def mse_matrix(self, pis): return self.ktc.compute_MSE_matrix(pis)
    def ve_diag(self, pis):    return self.ktc.compute_ve_matrix_diag(pis)


class RefacAdapter(_BaseAdapter):
    impl_name = "refactored"

    def weights(self, pis, eta, subtype, subseqs=None, calc_covar=False):
        return self.ktc.gauge_fixed_posterior(
            pis, eta, subseqs=subseqs or (), subseq_type=subtype,
            calc_covar=calc_covar, across_pis=False)

    def across(self, pis, eta, subtype, subseqs=None, calc_covar=False):
        return self.ktc.gauge_fixed_posterior(
            pis, eta, subseqs=subseqs or (), subseq_type=subtype,
            calc_covar=calc_covar, across_pis=True)

    def ve_matrix(self, pis):  return self.ktc.compute_metrics_matrix(pis, metric="ve")
    def mse_matrix(self, pis): return self.ktc.compute_metrics_matrix(pis, metric="mse")
    def ve_diag(self, pis):    return self.ktc.compute_metrics_matrix(pis, metric="ve", diagonal_only=True)


def make_computers(fx: Fixture, orig_module, refac_module):
    """Instantiate refactored (and original, if available) on the same fixture."""
    refac = RefacAdapter(refac_module.KernelTrickComputer(fx.model, fx.alphabet, fx.seq_length))
    orig = None
    if orig_module is not None:
        orig = OrigAdapter(orig_module.KernelTrickComputer(fx.model, fx.alphabet, fx.seq_length))
    return orig, refac


# =========================================================================== #
# Numeric + plotting helpers
# =========================================================================== #

def to_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float64).numpy().ravel()
    return np.asarray(x, dtype=np.float64).ravel()


def pearson_r(a, b) -> float:
    a, b = to_np(a), to_np(b)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if n < 2 or np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def max_abs_diff(a, b) -> float:
    a, b = to_np(a), to_np(b)
    n = min(a.size, b.size)
    return float(np.max(np.abs(a[:n] - b[:n]))) if n else float("nan")


def _close(a, b, atol, rtol) -> bool:
    return torch.allclose(a.to(torch.float64), b.to(torch.float64), atol=atol, rtol=rtol)


def _ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pairwise_scatter(named_arrays, out_path, title="Method agreement") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from itertools import combinations

    names = [n for n, _ in named_arrays]
    arrs = [to_np(a) for _, a in named_arrays]
    pairs = list(combinations(range(len(names)), 2))
    ncol = max(1, len(pairs))
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    for ax_idx, (i, j) in enumerate(pairs):
        ax = axes[0][ax_idx]
        n = min(arrs[i].size, arrs[j].size)
        x, y = arrs[i][:n], arrs[j][:n]
        ax.scatter(x, y, alpha=0.5, s=12)
        lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(names[i]); ax.set_ylabel(names[j])
        ax.set_title(f"{names[i]} vs {names[j]}\nr = {pearson_r(x, y):.5f}")
    fig.suptitle(title)
    fig.tight_layout()
    out = _ensure_dir(Path(out_path).parent) / Path(out_path).name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def save_parity_scatter(x, y, out_path, xlabel="original", ylabel="refactored",
                        title="Parity") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = to_np(x), to_np(y)
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(a, b, alpha=0.5, s=12)
    if n:
        lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nr = {pearson_r(a, b):.6f}   max|Δ| = {max_abs_diff(a, b):.2e}")
    fig.tight_layout()
    out = _ensure_dir(Path(out_path).parent) / Path(out_path).name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


# =========================================================================== #
# Check reporting
# =========================================================================== #

class Reporter:
    """Collects (name, ok, detail) rows and skip notes; prints a compact report."""

    def __init__(self):
        self.rows: List[Tuple[str, bool, str]] = []
        self.notes: List[Tuple[str, str]] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        self.rows.append((name, bool(ok), detail))

    def run(self, name: str, fn):
        """Run ``fn`` (returns bool or (bool, detail)); record a FAIL on exception."""
        try:
            res = fn()
            ok, detail = res if isinstance(res, tuple) else (res, "")
            self.check(name, ok, detail)
        except Exception as e:  # noqa: BLE001 -- one bad check must not abort the rest
            self.check(name, False, f"EXCEPTION {type(e).__name__}: {e}")

    def skip(self, name: str, why: str):
        self.notes.append((name, why))

    @property
    def all_pass(self) -> bool:
        return all(ok for _, ok, _ in self.rows)

    def print_report(self, header: str):
        print("\n" + "=" * 74)
        print(header)
        print("=" * 74)
        for name, ok, detail in self.rows:
            flag = "PASS" if ok else "FAIL"
            extra = f"   ({detail})" if detail else ""
            print(f"  [{flag}] {name}{extra}")
        for name, why in self.notes:
            print(f"  [SKIP] {name}   ({why})")
        n_pass = sum(1 for _, ok, _ in self.rows if ok)
        print("-" * 74)
        print(f"  {n_pass}/{len(self.rows)} checks passed, {len(self.notes)} skipped")


# =========================================================================== #
# Correctness:  refactored tool vs independent ground truth
# =========================================================================== #

def _weights_and_wpreds(fx, refac, num_subseqs=20):
    L, A = fx.seq_length, fx.n_alleles
    pi, eta, subseqs = generate_use_case(L, fx.alphabet, num_subseqs=num_subseqs)
    w = refac.weights(pi[None, ...], eta, "custom", subseqs=subseqs).flatten()
    seqs = fx.X.reshape(-1, L, A)
    enc = refac.encode_subseqs(subseqs)
    w_preds = predict_from_truncated(seqs, enc, w)
    return pi, eta, subseqs, w, w_preds


def run_correctness(rep, refac, fx, projection_cls, tol, out_dir, save_graphs,
                    n_samples, do_keops):
    L, A = fx.seq_length, fx.n_alleles
    atol, rtol, ve_rtol = tol["atol"], tol["rtol"], tol["ve_rtol"]
    preds, X = fx.preds, fx.X
    pi_u = torch.full((L, A), 1.0 / A, device=fx.device)

    # ---- expectations vs brute force ------------------------------------- #
    def e_map_check():
        ok_u = _close(refac.e_map(pi_u), preds.mean(), atol, rtol)
        pi, _, _ = generate_use_case(L, fx.alphabet, num_subseqs=1)
        probs = compute_seq_probs(X, pi)
        ok_r = _close(refac.e_map(pi), torch.sum(preds * probs), atol, rtol)
        return ok_u and ok_r, "uniform + random pi"
    rep.run("E_MAP == brute-force expectation", e_map_check)

    def e_map2_check():
        ok_u = _close(refac.e_map2(pi_u), (preds ** 2).mean(), atol, rtol)
        pi, _, _ = generate_use_case(L, fx.alphabet, num_subseqs=1)
        probs = compute_seq_probs(X, pi)
        ok_r = _close(refac.e_map2(pi), torch.sum(preds ** 2 * probs), atol, rtol)
        return ok_u and ok_r, "uniform + random pi"
    rep.run("E_MAP_2 == brute-force second moment", e_map2_check)

    if do_keops:
        rep.run("E_MAP_2 KeOps == dense",
                lambda: (_close(refac.e_map2(pi_u, keops=False),
                                refac.e_map2(pi_u, keops=True), atol, rtol), "keops path"))
    else:
        rep.skip("E_MAP_2 KeOps == dense", "pass --keops to run the slow KeOps check")

    def e_trunc_check():
        pi, eta, subseqs, w, w_preds = _weights_and_wpreds(fx, refac)
        ok_u = _close(refac.e_trunc(pi_u, w, subseqs), w_preds.mean(), atol, rtol)
        pr, _, _ = generate_use_case(L, fx.alphabet, num_subseqs=1)
        probs = compute_seq_probs(X, pr)
        ok_r = _close(refac.e_trunc(pr, w, subseqs), torch.sum(w_preds * probs), atol, rtol)
        return ok_u and ok_r, "uniform + random pi"
    rep.run("E_TRUNC == truncated-model expectation", e_trunc_check)

    def e_trunc2_check():
        pi, eta, subseqs, w, w_preds = _weights_and_wpreds(fx, refac)
        return _close(refac.e_trunc2(pi_u, w, subseqs), (w_preds ** 2).mean(), atol, rtol), "uniform pi"
    rep.run("E_TRUNC_2 == truncated second moment", e_trunc2_check)

    def e_map_trunc_check():
        pi, eta, subseqs, w, w_preds = _weights_and_wpreds(fx, refac)
        return _close(refac.e_map_trunc(pi_u, w, subseqs), (w_preds * preds).mean(), atol, rtol), "uniform pi"
    rep.run("E_MAP_TRUNC == cross expectation", e_map_trunc_check)

    def ve_mse_check():
        pi, eta, subseqs, w, w_preds = _weights_and_wpreds(fx, refac, num_subseqs=30)
        p = preds.to(torch.float64); wp = w_preds.to(torch.float64)
        pr, _, _ = generate_use_case(L, fx.alphabet, num_subseqs=1)
        probs = compute_seq_probs(X, pr).to(torch.float64)
        ve_naive, mse_naive = naive_ve_mse(p, wp, probs)
        ve = refac.variance_explained(pr, w, subseqs, use_double=True)
        mse = refac.mse(pr, w, subseqs, use_double=True)
        ok_mse = torch.allclose(mse.to(torch.float64), mse_naive, rtol=rtol, atol=atol)
        ok_ve = torch.allclose(ve.to(torch.float64), ve_naive, rtol=ve_rtol, atol=1e-3)
        return ok_mse and ok_ve, f"mse ok={ok_mse}, ve ok={ok_ve}"
    rep.run("variance_explained / MSE == naive enumeration", ve_mse_check)

    # ---- gauge-fixed posterior vs projection / sampling ------------------ #
    if projection_cls is None:
        rep.skip("mean == exact projection P @ preds", "projection_mtx.ProjectionMatrix not importable")
        rep.skip("covariance three-way agreement", "projection_mtx.ProjectionMatrix not importable")
    else:
        po = projection_cls(fx.alphabet, fx.seq_length)

        def mean_proj_check():
            worst = 0.0
            for _ in range(3):
                pi, eta, subseqs = generate_use_case(L, fx.alphabet, num_subseqs=4)
                w = refac.weights(pi[None, ...], eta, "custom", subseqs=subseqs).flatten()
                ref = mean_via_projection(po, preds, pi, eta, subseqs)
                worst = max(worst, max_abs_diff(w, ref.to(w.dtype)))
            return worst < atol, f"max|Δ|={worst:.2e}"
        rep.run("mean == exact projection P @ preds", mean_proj_check)

        def cov_three_way_check():
            kt_all, thm6_all, samp_all = [], [], []
            for trial in range(3):
                pi, eta, subseqs = generate_use_case(L, fx.alphabet, num_subseqs=2)
                _, cov = refac.weights(pi[None, ...], eta, "custom", subseqs=subseqs, calc_covar=True)
                cov = cov.squeeze(0)
                t6 = cov_via_thm6(fx.model, po, pi, eta, subseqs, X)
                samp = approx_cov_by_sampling(fx.model, po, pi, eta, subseqs, X,
                                              num_samples=n_samples, seed=trial)
                kt_all.append(cov); thm6_all.append(t6); samp_all.append(samp)
            kt = torch.cat([c.flatten() for c in kt_all])
            t6 = torch.cat([c.flatten() for c in thm6_all])
            sp = torch.cat([c.flatten() for c in samp_all])
            if save_graphs:
                save_pairwise_scatter(
                    [("sampling", sp), ("thm 6", t6), ("kernel trick", kt)],
                    out_dir / "correctness_covariance_three_way.png",
                    title=f"Covariance agreement ({fx.label})")
            ok_analytic = torch.allclose(kt.to(torch.float64), t6.to(torch.float64), atol=1e-3, rtol=1e-3)
            r_samp = pearson_r(kt, sp)
            return ok_analytic and (r_samp > 0.98), f"kt≈thm6={ok_analytic}, r(kt,sampling)={r_samp:.4f}"
        rep.run("covariance three-way agreement", cov_three_way_check)

    # ---- structural invariants ------------------------------------------- #
    def cov_sym_psd_check():
        pis = torch.full((1, L, A), 1.0 / A, device=fx.device)
        _, cov = refac.weights(pis, 1.0, "linear", calc_covar=True)
        cov = cov.squeeze(0).to(torch.float64)
        ok_sym = torch.allclose(cov, cov.T, atol=1e-6)
        sym = 0.5 * (cov + cov.T)
        eigmin = torch.linalg.eigvalsh(sym).min().item()
        tol_psd = -max(1e-6, 1e-4 * float(cov.abs().max()))
        return ok_sym and (eigmin >= tol_psd), f"sym={ok_sym}, min_eig={eigmin:.2e}"
    rep.run("covariance symmetric + PSD", cov_sym_psd_check)

    def batched_eq_check():
        pis = random_pis(L, A, batch=4, seed=1, device=fx.device)
        batched = refac.weights(pis, 1.0, "linear")
        worst = 0.0
        for i in range(pis.shape[0]):
            single = refac.weights(pis[i:i + 1], 1.0, "linear")
            worst = max(worst, max_abs_diff(batched[i], single.flatten()))
        return worst < 1e-5, f"max|Δ|={worst:.2e}"
    rep.run("batched mean == per-pi mean", batched_eq_check)

    # ---- targeted guards, one per refactor optimisation ------------------ #
    ktc = refac.ktc

    def change1_encode_cache():
        subs = ktc.linear_subseqs[:3]
        a = ktc._encode_subseqs(subs)
        b = ktc._encode_subseqs(subs)
        return a is b, "same cached tensor object on repeat call"
    rep.run("Change 1: _encode_subseqs memoised", change1_encode_cache)

    def change1_4_kx_cache():
        if not hasattr(ktc, "_kx_cache"):
            return False, "refactored KTC has no _kx_cache"
        ktc._kx_cache.clear()
        pi = torch.full((L, A), 1.0 / A, device=fx.device)
        _ = ktc.E_MAP(pi)
        populated = len(ktc._kx_cache) >= 1
        ktc.set_precision(use_double=True)
        cleared = len(ktc._kx_cache) == 0
        ktc.set_precision(use_double=False)
        return populated and cleared, f"populated={populated}, cleared_by_set_precision={cleared}"
    rep.run("Change 1/4: KX cache populated then cleared", change1_4_kx_cache)

    def change2_block_diag():
        pis = random_pis(L, A, batch=3, seed=2, device=fx.device)
        r = L * A
        _, block_cov = refac.across(pis, 1.0, "linear", calc_covar=True)
        worst = 0.0
        for i in range(pis.shape[0]):
            sl = slice(i * r, (i + 1) * r)
            _, single = refac.weights(pis[i:i + 1], 1.0, "linear", calc_covar=True)
            worst = max(worst, max_abs_diff(block_cov[sl, sl], single.squeeze(0)))
        return worst < 1e-5, f"max|Δ|={worst:.2e}"
    rep.run("Change 2: across-pis diagonal block == single-pi cov", change2_block_diag)

    def change3_across_mean():
        pis = random_pis(L, A, batch=4, seed=3, device=fx.device)
        across_mean = refac.across(pis, 1.0, "linear", calc_covar=False)
        per_pi = torch.cat([refac.weights(pis[i:i + 1], 1.0, "linear").flatten()
                            for i in range(pis.shape[0])])
        return max_abs_diff(across_mean, per_pi) < 1e-5, "batched == per-pi assembly"
    rep.run("Change 3: batched across-pis mean == per-pi", change3_across_mean)


# =========================================================================== #
# Equivalence:  original vs refactored, across the public API
# =========================================================================== #

def run_equivalence(rep, orig, refac, fx, tol, out_dir, save_graphs):
    L, A = fx.seq_length, fx.n_alleles
    atol, rtol = tol["atol"], tol["rtol"]
    pis = random_pis(L, A, batch=6, seed=11, device=fx.device)
    _, _, custom_subseqs = generate_use_case(L, fx.alphabet, num_subseqs=12,
                                             rng=random.Random(7))

    # weights (per-pi)
    for subtype in ("linear", "constant", "both"):
        rep.run(f"weights mean [{subtype}]",
                lambda st=subtype: (_close(orig.weights(pis, 1.0, st),
                                           refac.weights(pis, 1.0, st), atol, rtol), ""))
    rep.run("weights mean [custom]",
            lambda: (_close(orig.weights(pis, 1.0, "custom", subseqs=custom_subseqs),
                            refac.weights(pis, 1.0, "custom", subseqs=custom_subseqs),
                            atol, rtol), ""))

    for subtype in ("linear", "both"):
        def cov_check(st=subtype):
            om, oc = orig.weights(pis, 1.0, st, calc_covar=True)
            rm, rc = refac.weights(pis, 1.0, st, calc_covar=True)
            return (_close(om, rm, atol, rtol) and _close(oc, rc, atol, rtol)), "mean+cov"
        rep.run(f"weights mean+cov [{subtype}]", cov_check)

    # across-pis
    for subtype in ("linear", "both"):
        rep.run(f"across mean [{subtype}]",
                lambda st=subtype: (_close(orig.across(pis, 1.0, st),
                                           refac.across(pis, 1.0, st), atol, rtol), ""))
    for subtype in ("linear", "both"):
        def across_cov_check(st=subtype):
            om, oc = orig.across(pis, 1.0, st, calc_covar=True)
            rm, rc = refac.across(pis, 1.0, st, calc_covar=True)
            return (_close(om, rm, atol, rtol) and _close(oc, rc, atol, rtol)), "mean+block cov"
        rep.run(f"across mean+cov [{subtype}]", across_cov_check)

    # expectations + metrics at uniform pi
    def expectations_check():
        pi = torch.full((L, A), 1.0 / A, device=fx.device)
        w = refac.weights(pi[None, ...], 1.0, "custom", subseqs=custom_subseqs).flatten()
        checks = [
            _close(orig.e_map(pi), refac.e_map(pi), atol, rtol),
            _close(orig.e_map2(pi), refac.e_map2(pi), atol, rtol),
            _close(orig.e_trunc(pi, w, custom_subseqs), refac.e_trunc(pi, w, custom_subseqs), atol, rtol),
            _close(orig.e_trunc2(pi, w, custom_subseqs), refac.e_trunc2(pi, w, custom_subseqs), atol, rtol),
            _close(orig.e_map_trunc(pi, w, custom_subseqs), refac.e_map_trunc(pi, w, custom_subseqs), atol, rtol),
            _close(orig.variance_explained(pi, w, custom_subseqs), refac.variance_explained(pi, w, custom_subseqs), atol, rtol),
            _close(orig.mse(pi, w, custom_subseqs), refac.mse(pi, w, custom_subseqs), atol, rtol),
        ]
        return all(checks), f"{sum(checks)}/{len(checks)} expectation identities"
    rep.run("expectations + metrics agree", expectations_check)

    # metrics matrices -- the known compute_metrics_matrix ordering guard
    small_pis = random_pis(L, A, batch=4, seed=21, device=fx.device)
    rep.run("ve_matrix == compute_metrics_matrix(ve)",
            lambda: (_close(orig.ve_matrix(small_pis), refac.ve_matrix(small_pis), 1e-3, 1e-3),
                     "needs the subseq-ordering fix if this FAILs"))
    rep.run("mse_matrix == compute_metrics_matrix(mse)",
            lambda: (_close(orig.mse_matrix(small_pis), refac.mse_matrix(small_pis), 1e-3, 1e-3),
                     "needs the subseq-ordering fix if this FAILs"))
    rep.run("ve_diag == compute_metrics_matrix(ve, diagonal)",
            lambda: (_close(orig.ve_diag(small_pis), refac.ve_diag(small_pis), 1e-3, 1e-3),
                     "needs the subseq-ordering fix if this FAILs"))

    # parity graphs
    if save_graphs:
        try:
            save_parity_scatter(orig.weights(pis, 1.0, "both"), refac.weights(pis, 1.0, "both"),
                                out_dir / "equiv_mean_both.png",
                                title="Gauge-fixed mean (subseq_type=both)")
            _, o_cov = orig.weights(pis, 1.0, "linear", calc_covar=True)
            _, r_cov = refac.weights(pis, 1.0, "linear", calc_covar=True)
            save_parity_scatter(o_cov, r_cov, out_dir / "equiv_cov_linear.png",
                                title="Per-pi covariance (linear)")
            _, o_ac = orig.across(pis, 1.0, "linear", calc_covar=True)
            _, r_ac = refac.across(pis, 1.0, "linear", calc_covar=True)
            save_parity_scatter(o_ac, r_ac, out_dir / "equiv_cov_across_linear.png",
                                title="Across-pis covariance (linear)")
        except Exception as e:  # noqa: BLE001
            rep.skip("equivalence parity graphs", f"emit failed: {e}")


# =========================================================================== #
# Benchmark:  time every operation on both implementations
# =========================================================================== #

def _sync(device):
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _timeit(fn, repeats, warmup, device):
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    _sync(device)
    samples = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), out


def _to_parity(result):
    return result[-1] if isinstance(result, tuple) else result


def _make_scenarios(pis_b, pis_k):
    return [
        ("weights_both_mean", lambda a: a.weights(pis_b, 1.0, "both"),
         "baseline gauge-fixed mean"),
        ("weights_linear_cov", lambda a: a.weights(pis_b, 1.0, "linear", calc_covar=True),
         "per-pi covariance"),
        ("across_linear_mean", lambda a: a.across(pis_b, 1.0, "linear"),
         "Change 3 (batched across-pis mean)"),
        ("across_linear_cov", lambda a: a.across(pis_b, 1.0, "linear", calc_covar=True),
         "Change 2+3 (hoisted Y/M; batched mean)"),
        ("ve_matrix", lambda a: a.ve_matrix(pis_k), "Change 1 (encode + KX memoisation)"),
        ("mse_matrix", lambda a: a.mse_matrix(pis_k), "Change 1 (encode + KX memoisation)"),
    ]


def run_benchmark(orig, refac, fx, out_dir, repeats, warmup, batch, k, save_graphs):
    pis_b = random_pis(fx.seq_length, fx.n_alleles, batch=batch, seed=101, device=fx.device)
    pis_k = random_pis(fx.seq_length, fx.n_alleles, batch=k, seed=202, device=fx.device)
    scenarios = _make_scenarios(pis_b, pis_k)

    rows = []
    print(f"\nRunning {len(scenarios)} benchmark scenarios "
          f"(repeats={repeats}, warmup={warmup}, device={fx.device}) ...\n")
    for name, fn, note in scenarios:
        try:
            t_o, out_o = _timeit(lambda: fn(orig), repeats, warmup, fx.device)
            t_r, out_r = _timeit(lambda: fn(refac), repeats, warmup, fx.device)
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] ERROR: {e}")
            rows.append(dict(scenario=name, note=note, orig_ms=float("nan"),
                             refac_ms=float("nan"), speedup=float("nan"),
                             max_abs_diff=float("nan"), pearson_r=float("nan")))
            continue
        po, pr = _to_parity(out_o), _to_parity(out_r)
        mad = max_abs_diff(po, pr)
        r = pearson_r(po, pr)
        speedup = (t_o / t_r) if t_r > 0 else float("inf")
        rows.append(dict(scenario=name, note=note, orig_ms=t_o * 1e3, refac_ms=t_r * 1e3,
                         speedup=speedup, max_abs_diff=mad, pearson_r=r))
        print(f"  {name:22s}  orig={t_o*1e3:8.2f} ms  refac={t_r*1e3:8.2f} ms  "
              f"speedup={speedup:5.2f}x  max|Δ|={mad:.2e}  r={r:.5f}")
        if save_graphs:
            save_parity_scatter(po, pr, out_dir / f"bench_parity_{name}.png",
                                title=f"{name}\n({note})")

    _write_benchmark_summaries(rows, out_dir)
    if save_graphs:
        _plot_benchmark_timing(rows, out_dir)

    bad = [row for row in rows if row["scenario"] in ("ve_matrix", "mse_matrix")
           and (row["max_abs_diff"] != row["max_abs_diff"] or row["max_abs_diff"] > 1e-3)]
    if bad:
        print("\nNOTE: metrics matrices differ between implementations -- the known "
              "compute_metrics_matrix subseq-ordering issue. Apply the one-line fix "
              "(subseqs = self.constant_subseq + self.linear_subseqs) and re-run.")
    print(f"\nWrote benchmark graphs + summaries to: {out_dir}")
    return rows


def _write_benchmark_summaries(rows, out_dir: Path):
    import csv
    fields = ["scenario", "note", "orig_ms", "refac_ms", "speedup", "max_abs_diff", "pearson_r"]
    with open(out_dir / "benchmark_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with open(out_dir / "benchmark_summary.md", "w") as f:
        f.write("# KernelTrickComputer benchmark: original vs refactored\n\n")
        f.write("| scenario | exercises | orig (ms) | refac (ms) | speedup | max|Δ| | r |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['scenario']} | {row['note']} | {row['orig_ms']:.2f} | "
                    f"{row['refac_ms']:.2f} | {row['speedup']:.2f}x | "
                    f"{row['max_abs_diff']:.2e} | {row['pearson_r']:.5f} |\n")
        f.write("\n*Speedup = orig / refac (higher is better). A large max|Δ| flags a "
                "functional difference.*\n")


def _plot_benchmark_timing(rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["orig_ms"] == row["orig_ms"]]
    if not valid:
        return
    names = [row["scenario"] for row in valid]
    o = [row["orig_ms"] for row in valid]
    r = [row["refac_ms"] for row in valid]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4.6))
    ax.bar(x - width / 2, o, width, label="original")
    ax.bar(x + width / 2, r, width, label="refactored")
    ax.set_ylabel("median time (ms)"); ax.set_yscale("log")
    ax.set_title("Original vs refactored — per-operation timing")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    for xi, row in zip(x, valid):
        sp = row["speedup"]
        if sp == sp:
            ax.annotate(f"{sp:.1f}x", (xi + width / 2, row["refac_ms"]),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_timing.png", dpi=120)
    plt.close(fig)


# =========================================================================== #
# Main
# =========================================================================== #

def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "output" / "kernel_trick_computer_test"


def main(argv=None) -> bool:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None,
                    help="where graphs/summaries are written "
                         "(default: <tests>/output/kernel_trick_computer_test)")
    ap.add_argument("--seq-length", type=int, default=5)
    ap.add_argument("--alphabet", default="ABC")
    ap.add_argument("--device", default=None, help="cpu (default) or cuda")
    ap.add_argument("--fit-iters", type=int, default=None,
                    help="GP fit iterations (default 25 / $KTC_FIT_ITERS; identities are "
                         "independent of fit quality)")
    ap.add_argument("--samples", type=int, default=20000,
                    help="posterior samples for the sampling covariance reference")
    ap.add_argument("--keops", action="store_true",
                    help="also run the slow KeOps-vs-dense E_MAP_2 check")
    ap.add_argument("--no-graphs", action="store_true", help="disable PNG emission")
    ap.add_argument("--real-data", action="store_true",
                    help="also run the E_MAP identity on the KTC_REAL_* dataset")
    ap.add_argument("--atol", type=float, default=float(os.environ.get("KTC_ATOL", "1e-4")))
    ap.add_argument("--rtol", type=float, default=float(os.environ.get("KTC_RTOL", "1e-4")))
    ap.add_argument("--ve-rtol", type=float, default=float(os.environ.get("KTC_VE_RTOL", "1e-2")))
    # benchmark mode
    ap.add_argument("--benchmark", action="store_true",
                    help="run the timing/parity comparison instead of the check battery")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="batch size B for pis")
    ap.add_argument("--k", type=int, default=6, help="number of pis for metrics matrices")
    args = ap.parse_args(argv)

    out_dir = _ensure_dir(Path(args.output_dir).resolve() if args.output_dir
                          else _default_output_dir())
    save_graphs = not args.no_graphs
    tol = dict(atol=args.atol, rtol=args.rtol, ve_rtol=args.ve_rtol)

    print("=" * 74)
    print("KERNEL TRICK COMPUTER — consolidated test")
    print("=" * 74)
    print(f"torch {torch.__version__}   cuda available: {torch.cuda.is_available()}")
    print(f"output dir: {out_dir}")

    # ---- load implementations (original is optional; lives in modules/old/) ---- #
    try:
        orig_mod, refac_mod = load_implementations(require_original=False)
    except Exception as e:
        print(f"\nERROR: could not import the refactored implementation / its stack: {e}")
        return False
    _patch_epik_loader()
    has_orig = orig_mod is not None
    print(f"original implementation: {'found' if has_orig else 'NOT FOUND (equivalence + benchmark will be skipped; put it in modules/old/ or set KTC_ORIG_PATH)'}")

    # ---- build the model + computers ------------------------------------- #
    try:
        if args.fit_iters is not None:
            os.environ["KTC_FIT_ITERS"] = str(args.fit_iters)
        fx = build_synthetic_model(seq_length=args.seq_length, alphabet=args.alphabet,
                                   device=args.device)
    except Exception as e:
        print(f"\nERROR: could not build synthetic model (epik unavailable?): {e}")
        return False
    orig, refac = make_computers(fx, orig_mod, refac_mod)
    print(f"model: {fx.label}  L={fx.seq_length}  A={fx.n_alleles}  "
          f"|space|={len(fx.all_seqs)}  device={fx.device}")

    # ---- benchmark mode -------------------------------------------------- #
    if args.benchmark:
        if not has_orig:
            print("\nBenchmark needs the ORIGINAL implementation (modules/old/"
                  "kernel_trick_computer.py) to compare against. Nothing to do.")
            return False
        run_benchmark(orig, refac, fx, out_dir, args.repeats, args.warmup,
                      args.batch, args.k, save_graphs)
        print("\n==> BENCHMARK COMPLETE")
        return True

    # ---- correctness + equivalence check battery ------------------------- #
    projection_cls = try_import_projection_matrix()
    rep = Reporter()

    run_correctness(rep, refac, fx, projection_cls, tol, out_dir, save_graphs,
                    args.samples, args.keops)

    if has_orig:
        run_equivalence(rep, orig, refac, fx, tol, out_dir, save_graphs)
    else:
        rep.skip("equivalence (original vs refactored)",
                 "original not found in modules/old/ (set KTC_ORIG_PATH)")

    if args.real_data:
        def real_data_check():
            fxr = build_real_data_model(device=args.device)
            if not fxr.enumerable:
                return True, "skipped: real fixture not full-space enumerable (set KTC_REAL_ENUMERATE=1)"
            ktc = refac_mod.KernelTrickComputer(fxr.model, fxr.alphabet, fxr.seq_length)
            pi_u = torch.full((fxr.seq_length, fxr.n_alleles), 1.0 / fxr.n_alleles, device=fxr.device)
            return _close(ktc.E_MAP(pi_u), fxr.preds.mean(), 1e-3, 1e-3), fxr.label
        rep.run("real-data E_MAP identity", real_data_check)

    rep.print_report("RESULTS")
    if save_graphs:
        print(f"\ngraphs written under: {out_dir}")
    print("\n==>", "ALL CHECKS PASS" if rep.all_pass else "SOME CHECKS FAILED")
    return rep.all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)