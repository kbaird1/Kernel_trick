"""
Shared foundation for the kernel-trick test suite.

This module is intentionally dependency-light at import time: it imports ``torch``
(required by everything downstream) but defers ``epik`` / ``pykeops`` / the
implementation modules to the functions that actually need them, so that a missing
heavy dependency turns into a clean pytest *skip* rather than a collection error.

Contents
--------
* Implementation loading        : load the original + refactored modules side by side.
* Fixture construction          : synthetic (no data files) + optional real-data hook.
* Brute-force references         : compute_seq_probs / predict_from_truncated / naive VE+MSE
                                   (lifted from variance_explained.ipynb).
* Projection / sampling refs     : mean via P, covariance via "thm 6", empirical covariance
                                   via posterior sampling (lifted from verify_kernel_trick.ipynb).
                                   These require the user's ``projection_mtx.ProjectionMatrix``.
* Adapters                       : a single uniform interface over the original and refactored
                                   public APIs (their entry-point names differ).
* Plotting helpers               : pairwise correlation scatter + parity scatter -> PNG.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Implementation loading
# ---------------------------------------------------------------------------

ORIG_FILENAME = "kernel_trick_computer.py"
REFAC_FILENAME = "kernel_trick_computer_refactored.py"


def _candidate_module_dirs() -> List[Path]:
    """Directories to search for the two implementation files, in priority order."""
    here = Path(__file__).resolve().parent
    dirs: List[Path] = []
    env = os.environ.get("KTC_MODULES_DIR")
    if env:
        dirs.append(Path(env))
    cwd = Path.cwd()
    dirs += [
        here,
        here.parent,
        here.parent / "modules",
        cwd,
        cwd / "modules",
        cwd.parent,
        cwd.parent / "modules",
    ]
    # de-duplicate while preserving order
    seen, unique = set(), []
    for d in dirs:
        rd = d.resolve()
        if rd not in seen:
            seen.add(rd)
            unique.append(rd)
    return unique


def _find_file(filename: str, explicit_env: Optional[str]) -> Path:
    if explicit_env and os.environ.get(explicit_env):
        p = Path(os.environ[explicit_env]).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{explicit_env}={p} does not exist")
        return p
    for d in _candidate_module_dirs():
        p = d / filename
        if p.exists():
            return p.resolve()
    searched = "\n  ".join(str(d) for d in _candidate_module_dirs())
    raise FileNotFoundError(
        f"Could not locate '{filename}'. Set KTC_MODULES_DIR (or "
        f"KTC_ORIG_PATH / KTC_REFAC_PATH) to point at it. Searched:\n  {searched}"
    )


def _load_module_from_path(mod_name: str, path: Path):
    """Import a .py file under a private module name, isolated from the canonical name."""
    # Ensure the modules dir is importable so intra-package imports (if any) resolve.
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


def load_implementations():
    """Return (original_module, refactored_module).

    Raises ImportError/FileNotFoundError if the files or their heavy dependencies
    (pykeops, gpytorch, linear_operator) are unavailable; callers should treat that
    as a skip condition.
    """
    orig_path = _find_file(ORIG_FILENAME, "KTC_ORIG_PATH")
    refac_path = _find_file(REFAC_FILENAME, "KTC_REFAC_PATH")
    orig = _load_module_from_path("ktc_orig_impl", orig_path)
    refac = _load_module_from_path("ktc_refac_impl", refac_path)
    return orig, refac


def try_import_projection_matrix():
    """Return the user's ProjectionMatrix class, or None if it is not importable.

    The notebooks import ``from projection_mtx import ProjectionMatrix`` out of a
    ``modules/`` directory; we mirror that and degrade gracefully when absent.
    """
    for d in _candidate_module_dirs():
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        from projection_mtx import ProjectionMatrix  # type: ignore
        return ProjectionMatrix
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fixture container
# ---------------------------------------------------------------------------

@dataclass
class Fixture:
    """Everything a test needs about one fitted model + its full sequence space."""
    model: Any
    alphabet: str
    seq_length: int
    n_alleles: int
    X: torch.Tensor          # encoded *full* sequence space, shape (N, L*A)
    all_seqs: List[str]      # the N sequences as strings
    preds: torch.Tensor      # posterior mean over the full space, shape (N,)
    device: torch.device
    label: str = "synthetic"
    enumerable: bool = True   # whether X spans the entire sequence space (brute force valid)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_synthetic_model(
    seq_length: int = 5,
    alphabet: str = "ABC",
    seed: int = 0,
    n_iter: Optional[int] = None,
    learning_rate: float = 0.1,
    device: Optional[str] = None,
) -> Fixture:
    """Construct a small, fully-enumerable synthetic GP fixture (no data files needed).

    Mirrors the synthetic cell in the verify notebooks: a JengaKernel over the entire
    sequence space with random linear ground-truth effects. Because the full space is
    enumerated, brute-force / sampling / projection references are all valid.

    Correctness identities tested against this fixture are *exact identities about the
    estimator* (both sides use the same model), so they hold regardless of fit quality;
    ``n_iter`` only affects speed, not whether the identities should pass.
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

    splits = split_training_test(X, y, y_var, ptrain=0.5, dtype=torch.float32)
    train_x, train_y, test_x, test_y, train_y_var = splits

    kernel = JengaKernel(n_alleles=A, seq_length=seq_length)
    model = EpiK(kernel, track_progress=False, device=str(device)) if _accepts_device(EpiK) \
        else EpiK(kernel, track_progress=False)
    model.set_data(train_x, train_y, train_y_var)
    if n_iter > 0:
        model.fit(n_iter=n_iter, learning_rate=learning_rate)

    preds = _predict_full(model, X)

    return Fixture(
        model=model,
        alphabet=alphabet,
        seq_length=seq_length,
        n_alleles=A,
        X=X.to(device),
        all_seqs=all_seqs,
        preds=preds.to(device),
        device=device,
        label="synthetic",
        enumerable=True,
    )


def build_real_data_model(
    csv_path: Optional[str] = None,
    alphabet: Optional[str] = None,
    seq_length: Optional[int] = None,
    kernel_name: str = "ConnectednessKernel",
    ptrain: float = 0.02,
    n_iter: int = 100,
    params_path: Optional[str] = None,
    enumerate_full_space: bool = False,
    device: Optional[str] = None,
    seed: int = 0,
) -> Fixture:
    """Optional real-data hook (mirrors the CSV/checkpoint cells in the notebooks).

    Driven by environment variables so it can be enabled without editing code:
      KTC_REAL_CSV       path to a CSV with an index of sequences and a 'y' (+'y_var') column
      KTC_REAL_ALPHABET  e.g. 'ACGU' or 'ARNDCEQGHILKMFPSTWYV'
      KTC_REAL_L         sequence length (int)
      KTC_REAL_KERNEL    one of ConnectednessKernel|JengaKernel|GeneralProductKernel
      KTC_REAL_PARAMS    optional path to a saved GeneralProduct model_params .pth
      KTC_REAL_PTRAIN    training fraction (float)
      KTC_REAL_ENUMERATE '1' to enumerate the full A**L space (only feasible for small A**L)

    Returns a Fixture. ``enumerable`` is True only when the full space was enumerated;
    brute-force/sampling/projection references require ``enumerable=True``.
    """
    import pandas as pd
    from epik.epik.utils import encode_seqs, split_training_test
    from epik.epik.model import EpiK
    from epik.epik import kernel as kernel_mod

    csv_path = csv_path or os.environ.get("KTC_REAL_CSV")
    if not csv_path:
        raise FileNotFoundError("No real-data CSV provided (set KTC_REAL_CSV).")
    alphabet = alphabet or os.environ.get("KTC_REAL_ALPHABET")
    if not alphabet:
        raise ValueError("Real-data alphabet required (set KTC_REAL_ALPHABET).")
    seq_length = int(seq_length or os.environ.get("KTC_REAL_L", "0"))
    if seq_length <= 0:
        raise ValueError("Real-data sequence length required (set KTC_REAL_L).")
    kernel_name = os.environ.get("KTC_REAL_KERNEL", kernel_name)
    params_path = params_path or os.environ.get("KTC_REAL_PARAMS")
    ptrain = float(os.environ.get("KTC_REAL_PTRAIN", str(ptrain)))
    enumerate_full_space = enumerate_full_space or os.environ.get("KTC_REAL_ENUMERATE") == "1"
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

    splits = split_training_test(X, y, y_var, ptrain=ptrain, dtype=torch.float32)
    train_x, train_y, test_x, test_y, train_y_var = splits

    KernelCls = getattr(kernel_mod, kernel_name)
    if params_path:
        params = torch.load(params_path, map_location=str(device))
        kernel = KernelCls(
            n_alleles=A, seq_length=seq_length,
            theta0=params["covar_module.theta"],
            log_var=params["covar_module.log_var"],
        )
        model = EpiK(kernel, train_noise=True, track_progress=False)
        model.set_data(train_x, train_y, y_var=train_y_var)
        if "likelihood.second_noise_covar.raw_noise" in params:
            model.likelihood.second_noise_covar.noise = torch.exp(
                params["likelihood.second_noise_covar.raw_noise"]
            )
    else:
        kernel = KernelCls(n_alleles=A, seq_length=seq_length)
        model = EpiK(kernel, track_progress=False)
        model.set_data(train_x, train_y, train_y_var)
        if n_iter > 0:
            model.fit(n_iter=n_iter, learning_rate=0.1)

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

    return Fixture(
        model=model,
        alphabet=alphabet,
        seq_length=seq_length,
        n_alleles=A,
        X=Xfull.to(device),
        all_seqs=all_seqs,
        preds=preds.to(device),
        device=device,
        label=f"real:{Path(csv_path).stem}",
        enumerable=enumerable,
    )


def _accepts_device(cls) -> bool:
    try:
        import inspect
        return "device" in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _predict_full(model, X: torch.Tensor, batch_size: int = 5000) -> torch.Tensor:
    """Posterior mean over X, batched to avoid blowing up memory. Populates GP caches."""
    chunks = []
    for i in range(0, X.shape[0], batch_size):
        df = model.predict(X[i:i + batch_size], calc_variance=False)
        vals = df.values if hasattr(df, "values") else np.asarray(df)
        chunks.append(torch.as_tensor(vals.flatten(), dtype=torch.float32))
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Use-case generation
# ---------------------------------------------------------------------------

def generate_use_case(seq_length: int, alphabet: str, num_subseqs: int = 200,
                      rng: Optional[random.Random] = None,
                      gen: Optional[torch.Generator] = None
                      ) -> Tuple[torch.Tensor, float, List[str]]:
    """Seeded port of the notebooks' ``generate_use_case``.

    Returns (pi [L,A] row-stochastic, eta in (0.01,0.99), subseqs [num_subseqs strings]).
    """
    rng = rng or random.Random()
    A = len(alphabet)
    if gen is not None:
        pi = torch.rand(seq_length, A, generator=gen)
    else:
        pi = torch.rand(seq_length, A)
    pi = pi / pi.sum(dim=1, keepdim=True)
    eta = rng.uniform(0.01, 0.99)
    subseqs = [
        "".join("-" if rng.random() < 0.5 else rng.choice(alphabet) for _ in range(seq_length))
        for _ in range(num_subseqs)
    ]
    return pi.to(torch.float32), eta, subseqs


def uniform_pis(seq_length: int, n_alleles: int, batch: int = 1,
                device: Optional[torch.device] = None) -> torch.Tensor:
    """A batch of uniform row-stochastic pis, shape (batch, L, A)."""
    p = torch.full((batch, seq_length, n_alleles), 1.0 / n_alleles, dtype=torch.float32)
    return p.to(device) if device is not None else p


def random_pis(seq_length: int, n_alleles: int, batch: int = 4, seed: int = 0,
               device: Optional[torch.device] = None) -> torch.Tensor:
    """A batch of random row-stochastic pis, shape (batch, L, A)."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(batch, seq_length, n_alleles, generator=g)
    p = p / p.sum(dim=-1, keepdim=True)
    p = p.to(torch.float32)
    return p.to(device) if device is not None else p


# ---------------------------------------------------------------------------
# Brute-force references (from variance_explained.ipynb)
# ---------------------------------------------------------------------------

def compute_seq_probs(X: torch.Tensor, pis: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Probability of each enumerated sequence under the per-position distribution pis.

    X: (N, L*A) one-hot encoded full space. pis: (L, A). Returns (N,) summing to 1.
    """
    L, A = pis.shape
    pis = pis.to(X.device)
    probs_pos = torch.sum(X.reshape(-1, L, A) * pis, dim=-1)   # (N, L)
    probs_pos = probs_pos.clamp(min=eps)
    log_seq = torch.sum(torch.log(probs_pos), dim=-1)          # (N,)
    log_seq = log_seq - torch.logsumexp(log_seq, dim=0)
    out = torch.exp(log_seq)
    total = out.sum().item()
    assert np.isfinite(total) and abs(total - 1.0) < 1e-6, f"seq probs do not sum to 1: {total}"
    return out


def predict_from_truncated(seqs: torch.Tensor, encoded_subseqs: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
    """Truncated/linear-model predictions over enumerated sequences.

    seqs: (N, L, A). encoded_subseqs: (r, L, A+1). weights: (r,). Returns (N,).
    """
    dtype = seqs.dtype
    compat = (encoded_subseqs[..., :-1] + encoded_subseqs[..., -1:]).to(dtype)  # (r, L, A)
    Phi = torch.einsum("rla,tla->rtl", compat, seqs.to(dtype))                  # (r, N, L)
    indicator = torch.prod(Phi, dim=-1)                                        # (r, N)
    return weights.to(dtype) @ indicator                                       # (N,)


def naive_ve_mse(preds: torch.Tensor, w_preds: torch.Tensor,
                 seq_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brute-force (variance_explained, mse) under distribution seq_probs.

    Matches the naive expressions used in variance_explained.ipynb:
        num = E[(w-p)^2] - E[w-p]^2  (== analytic MSE)
        den = E[p^2] - E[p]^2
        ve  = 1 - num/den
    """
    diff = w_preds - preds
    num = torch.sum(diff ** 2 * seq_probs) - torch.sum(diff * seq_probs) ** 2
    den = torch.sum(seq_probs * preds ** 2) - torch.sum(seq_probs * preds) ** 2
    ve = 1.0 - num / den
    return ve, num


# ---------------------------------------------------------------------------
# Projection / sampling references (from verify_kernel_trick.ipynb)
# ---------------------------------------------------------------------------

def get_test_train_K(model, test_x: torch.Tensor, train_x: torch.Tensor) -> torch.Tensor:
    n_train, n_test = train_x.shape[0], test_x.shape[0]
    full = model.gp.forward(torch.cat([test_x, train_x])).covariance_matrix
    return full[:n_test, n_test:]


def get_test_test_K(model, test_x: torch.Tensor) -> torch.Tensor:
    return model.gp.forward(test_x).covariance_matrix


def get_S(model) -> torch.Tensor:
    return model.gp.prediction_strategy.covar_cache


def mean_via_projection(po, preds: torch.Tensor, pis: torch.Tensor,
                        eta: float, subseqs: Sequence[str]) -> torch.Tensor:
    """Reference gauge-fixed weights: P @ preds (the exact projection)."""
    P = po.compute_P(pis, eta, subseqs)
    return P.to(preds.dtype) @ preds


def cov_via_thm6(model, po, pis: torch.Tensor, eta: float,
                 subseqs: Sequence[str], all_x: torch.Tensor) -> torch.Tensor:
    """Reference gauge-fixed covariance via 'thm 6': P K P^T - (P K_tt S)(P K_tt S)^T."""
    P = po.compute_P(pis, eta, subseqs)
    Ktt = get_test_train_K(model, all_x, model.X)
    K = get_test_test_K(model, all_x)
    S = get_S(model)
    P = P.to(K.dtype)
    PKS = P @ Ktt @ S
    return P @ K @ P.T - PKS @ PKS.T


def approx_cov_by_sampling(model, po, pis: torch.Tensor, eta: float,
                           subseqs: Sequence[str], all_x: torch.Tensor,
                           num_samples: int = 20000, seed: int = 0) -> torch.Tensor:
    """Empirical covariance of the projected posterior samples."""
    torch.manual_seed(seed)
    P = po.compute_P(pis, eta, subseqs)
    samples = model.get_posterior(all_x, calc_variance=True).sample(
        sample_shape=torch.Size([num_samples])
    )
    P = P.to(samples.dtype)
    p_samples = (P @ samples.T).T               # (num_samples, r)
    mean = p_samples.mean(dim=0)
    centered = p_samples - mean
    return centered.T @ centered / (p_samples.size(0) - 1)


# ---------------------------------------------------------------------------
# Adapters: a uniform interface over the two differing public APIs
# ---------------------------------------------------------------------------

# Subseq selectors understood by both adapters.
SUBTYPES = ("linear", "constant", "both", "custom")


class _BaseAdapter:
    impl_name = "base"

    def __init__(self, ktc):
        self.ktc = ktc

    # expectations / metrics share the same method names across implementations
    def e_map(self, pi):                       return self.ktc.E_MAP(pi)
    def e_map2(self, pi, keops=False):         return self.ktc.E_MAP_2(pi, keops=keops)
    def e_trunc(self, pi, w, subseqs):         return self.ktc.E_TRUNC(pi, w, subseqs)
    def e_trunc2(self, pi, w, subseqs):        return self.ktc.E_TRUNC_2(pi, w, subseqs)
    def e_map_trunc(self, pi, w, subseqs):     return self.ktc.E_MAP_TRUNC(pi, w, subseqs)
    def variance_explained(self, pi, w, subseqs, **kw): return self.ktc.variance_explained(pi, w, subseqs, **kw)
    def mse(self, pi, w, subseqs, **kw):       return self.ktc.MSE(pi, w, subseqs, **kw)
    def encode_subseqs(self, subseqs):         return self.ktc._encode_subseqs(subseqs)

    @property
    def linear_subseqs(self):   return self.ktc.linear_subseqs
    @property
    def constant_subseq(self):  return self.ktc.constant_subseq

    # the following are implementation-specific; subclasses override
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
            calc_covar=calc_covar, across_pis=False,
        )

    def across(self, pis, eta, subtype, subseqs=None, calc_covar=False):
        return self.ktc.gauge_fixed_posterior(
            pis, eta, subseqs=subseqs or (), subseq_type=subtype,
            calc_covar=calc_covar, across_pis=True,
        )

    def ve_matrix(self, pis):  return self.ktc.compute_metrics_matrix(pis, metric="ve")
    def mse_matrix(self, pis): return self.ktc.compute_metrics_matrix(pis, metric="mse")
    def ve_diag(self, pis):    return self.ktc.compute_metrics_matrix(pis, metric="ve", diagonal_only=True)


def make_computers(fx: Fixture, orig_module, refac_module
                   ) -> Tuple[OrigAdapter, RefacAdapter]:
    """Instantiate both KernelTrickComputer variants on the same fixture."""
    orig = orig_module.KernelTrickComputer(fx.model, fx.alphabet, fx.seq_length)
    refac = refac_module.KernelTrickComputer(fx.model, fx.alphabet, fx.seq_length)
    return OrigAdapter(orig), RefacAdapter(refac)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plotting (matplotlib; no seaborn requirement)
# ---------------------------------------------------------------------------

def _ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pairwise_scatter(named_arrays: Sequence[Tuple[str, Any]], out_path,
                          title: str = "Method agreement") -> str:
    """Pairwise scatter plots across N methods (mirrors the notebooks' 3-way panels).

    named_arrays: list of (name, array-like). Produces one scatter per unique pair,
    annotated with Pearson r, plus a y=x reference line.
    """
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
                        title: str = "Parity") -> str:
    """Single parity scatter (refactored vs original) annotated with r and max|diff|."""
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
