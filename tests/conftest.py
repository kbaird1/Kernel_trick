"""
pytest configuration and shared fixtures for the kernel-trick test suite.

Heavy dependencies (torch, epik, pykeops, gpytorch, linear_operator) and the two
implementation modules are resolved lazily here so that an environment missing any
of them yields a clean, explained *skip* rather than a collection crash.

Command-line options
--------------------
  --output-dir DIR   where graphs are written (default: ./output_graphs)
  --real-data        also build the optional real-data fixture (needs KTC_REAL_* env vars)
  --samples N        posterior samples for the sampling-based covariance reference
  --save-graphs      emit correlation/parity PNGs from the pytest tests (on by default)
  --no-save-graphs   disable graph emission from pytest

Tolerances can be overridden with env vars KTC_ATOL / KTC_RTOL / KTC_VE_RTOL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# ---- sys.path bootstrap ----------------------------------------------------
# pytest does not put the project root on sys.path, so `import epik` (a package
# living at the repo root) fails under `pytest` even though it works in a script.
# Walk up from this file to find the repo root (the dir containing epik/ and
# modules/) and put both the root and modules/ on sys.path before anything else
# imports. This makes `import epik`, `import epik_loader`, `import projection_mtx`,
# and `import kt_test_utils` all resolve regardless of the working directory.
def _bootstrap_paths():
    here = Path(__file__).resolve().parent          # .../tests
    for cand in [here, *here.parents]:
        if (cand / "epik").is_dir() and (cand / "modules").is_dir():
            root = cand
            break
    else:
        root = here.parent                          # best-effort fallback
    for p in (str(root / "modules"), str(root), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("KTC_MODULES_DIR", str(root / "modules"))


_bootstrap_paths()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "keops: test exercises the KeOps/LazyTensor path (slow first call; "
                   "deselect with -m 'not keops')")


def pytest_addoption(parser):
    parser.addoption("--output-dir", action="store", default="output_graphs",
                     help="Directory for emitted graphs.")
    parser.addoption("--real-data", action="store_true", default=False,
                     help="Also build the real-data fixture (requires KTC_REAL_* env).")
    parser.addoption("--samples", action="store", type=int, default=20000,
                     help="Posterior samples for the sampling covariance reference.")
    parser.addoption("--save-graphs", action="store_true", default=True,
                     help="Emit correlation/parity graphs from pytest tests.")
    parser.addoption("--no-save-graphs", action="store_true", default=False,
                     help="Disable graph emission from pytest tests.")


# ---- epik_loader shim ------------------------------------------------------
# In this repo, `epik` is a package living at the project root and is already
# importable (verified before the suite runs). The implementations, however, call
# epik_loader.ensure_epik_loaded() inside _encode_subseqs, and that function hunts
# for the epik *source tree* at a path derived one level above the project -- which
# does not exist here, raising FileNotFoundError. Since epik is already available,
# we neutralise ensure_epik_loaded() into a no-op (this is what the verify notebook
# did via monkeypatch). If epik is NOT importable, we leave the loader untouched so
# its original error still surfaces.

def _patch_epik_loader():
    try:
        import epik  # noqa: F401
    except Exception:
        return "epik package not importable"
    try:
        import epik_loader
        epik_loader.ensure_epik_loaded = lambda *a, **k: None
    except Exception:
        # No epik_loader module in this layout; nothing to patch.
        pass
    return None


# ---- module-level availability gate (skips the whole suite cleanly) --------

def _require_stack():
    """Import torch + both implementations, or return a skip reason string."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        return None, f"torch not importable: {e}"
    try:
        import kt_test_utils as U
        orig, refac = U.load_implementations()
        _patch_epik_loader()   # neutralise epik_loader path-hunting once epik is present
        return (U, orig, refac), None
    except Exception as e:
        return None, f"implementations not importable: {e}"


_STACK, _SKIP_REASON = _require_stack()


@pytest.fixture(scope="session")
def utils():
    if _STACK is None:
        pytest.skip(_SKIP_REASON)
    return _STACK[0]


@pytest.fixture(scope="session")
def implementations():
    if _STACK is None:
        pytest.skip(_SKIP_REASON)
    return _STACK[1], _STACK[2]


@pytest.fixture(scope="session")
def synth(utils):
    """The default synthetic fixture (fully enumerable, no data files)."""
    try:
        return utils.build_synthetic_model()
    except Exception as e:
        pytest.skip(f"could not build synthetic model (epik unavailable?): {e}")


@pytest.fixture(scope="session")
def real(request, utils):
    """Optional real-data fixture; skipped unless --real-data and KTC_REAL_* are set."""
    if not request.config.getoption("--real-data"):
        pytest.skip("real-data tests not requested (use --real-data)")
    try:
        return utils.build_real_data_model()
    except Exception as e:
        pytest.skip(f"real-data fixture unavailable: {e}")


@pytest.fixture(scope="session")
def computers(synth, utils, implementations):
    """(OrigAdapter, RefacAdapter) built on the synthetic fixture."""
    orig, refac = implementations
    return utils.make_computers(synth, orig, refac)


@pytest.fixture(scope="session")
def projection(utils):
    """The user's ProjectionMatrix class, or skip if unavailable."""
    PM = utils.try_import_projection_matrix()
    if PM is None:
        pytest.skip("projection_mtx.ProjectionMatrix not importable; "
                    "projection/sampling references unavailable")
    return PM


@pytest.fixture(scope="session")
def output_dir(request):
    d = Path(request.config.getoption("--output-dir")).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def save_graphs(request):
    if request.config.getoption("--no-save-graphs"):
        return False
    return bool(request.config.getoption("--save-graphs"))


@pytest.fixture(scope="session")
def n_samples(request):
    return int(request.config.getoption("--samples"))


@pytest.fixture(scope="session")
def tol():
    return dict(
        atol=float(os.environ.get("KTC_ATOL", "1e-4")),
        rtol=float(os.environ.get("KTC_RTOL", "1e-4")),
        ve_rtol=float(os.environ.get("KTC_VE_RTOL", "1e-2")),
    )