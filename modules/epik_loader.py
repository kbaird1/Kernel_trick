from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Literal, Optional, Union


def _repo_root() -> Path:
    # This file lives at: <repo>/kernel_trick_2/modules/epik_loader.py
    return Path(__file__).resolve().parents[2]


EpikFlavor = Literal["epik", "epik_dev"]
_ACTIVE_EPIK_ROOT: Optional[Path] = None


def _normalize_epik_flavor(epik_flavor: str) -> EpikFlavor:
    flavor = epik_flavor.strip().lower()
    if flavor not in {"epik", "epik_dev"}:
        raise ValueError(
            f"Unsupported epik_flavor '{epik_flavor}'. "
            "Expected one of: 'epik', 'epik_dev'."
        )
    return flavor  # type: ignore[return-value]


def get_epik_repo_root(
    epik_flavor: str = "epik",
    epik_code_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Return the repo root that contains the `epik/` python package directory.

    There are only two supported loading modes:
    - epik: uses <repo>/epik
    - epik_dev: uses <repo>/epik-dev
    """
    flavor = _normalize_epik_flavor(epik_flavor)
    if epik_code_root is not None:
        return Path(epik_code_root)

    if flavor == "epik":
        return _repo_root() / "epik"
    return _repo_root() / "epik-dev"


def ensure_epik_loaded(
    epik_flavor: str = "epik",
    epik_code_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Make sure Python imports `epik` from the requested checkout.

    - Inserts `<epik_repo_root>` at the front of `sys.path`
    - If `epik` was already imported from a different checkout in this process,
      clears `epik*` modules so the next import uses the correct one.
    """
    global _ACTIVE_EPIK_ROOT

    if epik_code_root is None and _ACTIVE_EPIK_ROOT is not None:
        epik_repo_root = _ACTIVE_EPIK_ROOT.resolve()
    else:
        epik_repo_root = get_epik_repo_root(epik_flavor, epik_code_root).resolve()
    inner_pkg_dir = epik_repo_root / "epik"
    if not inner_pkg_dir.exists():
        raise FileNotFoundError(
            f"Could not find epik Python package at '{inner_pkg_dir}'. "
            "Use epik_flavor='epik' or 'epik_dev', or provide epik_code_root."
        )

    epik_repo_root_str = str(epik_repo_root)
    if epik_repo_root_str not in sys.path:
        sys.path.insert(0, epik_repo_root_str)

    # If epik was already imported from another checkout, drop the cached modules.
    try:
        import epik as epik_pkg  # type: ignore

        loaded_root = Path(epik_pkg.__file__).resolve().parents[1]
        if loaded_root != epik_repo_root:
            for name in list(sys.modules):
                if name == "epik" or name.startswith("epik."):
                    del sys.modules[name]
            importlib.invalidate_caches()
    except ModuleNotFoundError:
        # Not imported yet; fine.
        pass

    _ACTIVE_EPIK_ROOT = epik_repo_root
    return epik_repo_root

