#dataset_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    A: int
    L: int
    alphabet: str
    offset: int
    data_csv: Path
    model_params: Path
    alphabet_csv: Optional[Path] = None
    kernel_type: str = "GeneralProduct"
    epik_flavor: str = "epik_dev"
    train_noise: bool = True
    train_mean: bool = False
    y_column: str = "y"
    index_col: int = 0
    phenotype_offset: float = 0.0
    use_keops: bool = False


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    seqs: np.ndarray
    train_seqs: np.ndarray
    X: torch.Tensor
    y: np.ndarray
    y_var: np.ndarray
    kernel: Any
    model: Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_dataset_spec(dataset: str) -> DatasetSpec:
    ds = dataset.strip().upper()
    root = _repo_root()

    if ds == "AAV":
        # aav2.GeneralProduct.params.pth was trained with the epik-dev default
        # "protein" alphabet (see epik-dev/epik/settings.py). The per-site
        # theta_p tensors are indexed in this exact "RKEDQNHSTAVILMPGYFWC"
        # order; every site's one-hot column ordering MUST match this string
        # or predictions are silently permuted per site. Empirically verified
        # against the old (epik, sorted-alphabet) params in
        # compare_AAV_kernels.ipynb: this ordering minimises the per-site
        # kernel diff.
        alphabet = "RKEDQNHSTAVILMPGYFWC"
        return DatasetSpec(
            name="AAV",
            A=20,
            L=28,
            alphabet=alphabet,
            offset=561,
            data_csv=root / "fitness_landscape_data" / "aav.csv",
            model_params=(
                root
                / "fitness_landscape_data"
                / "aav_models"
                / "aav2.GeneralProduct.params.pth"
            ),
            kernel_type="GeneralProduct",
            epik_flavor="epik_dev",
            y_column="y",
            index_col=0,
            phenotype_offset=5.77,
            use_keops=False,
        )

    if ds == "GRB2":
        alphabet = "01"
        return DatasetSpec(
            name="GRB2",
            A=2,
            L=34,
            alphabet=alphabet,
            offset=0,
            data_csv=root / "fitness_landscape_data" / "grb2.csv",
            model_params=(
                root
                / "fitness_landscape_data"
                / "grb2_models"
                / "grb2.Connectedness.58.params.pth"
            ),
            alphabet_csv=root / "fitness_landscape_data" / "grb2.alphabet.csv",
            kernel_type="Connectedness",
            epik_flavor="epik_dev",
            train_mean=True,
            y_column="y",
            index_col=0,
            phenotype_offset=0.0,
            use_keops=True
        )

    raise ValueError(f"Unknown dataset '{dataset}'. Add it in get_dataset_spec().")


def _read_alphabet_list(csv_path: Path) -> List[List[str]]:
    alpha_df = pd.read_csv(csv_path, index_col=0)
    return [[str(r["0"]), str(r["1"])] for _, r in alpha_df.iterrows()]


def _build_alphabet_list(spec: DatasetSpec) -> List[List[str]]:
    """Build the per-site alphabet list for `spec`.

    If `spec.alphabet_csv` is provided, use it (per-site alphabets). Otherwise,
    replicate `spec.alphabet` across all `spec.L` sites.
    """
    if spec.alphabet_csv is not None and spec.alphabet_csv.exists():
        alphabet_list = _read_alphabet_list(spec.alphabet_csv)
        if len(alphabet_list) != spec.L:
            raise ValueError(
                f"alphabet_csv has {len(alphabet_list)} sites but spec.L={spec.L}"
            )
        return alphabet_list
    return [list(spec.alphabet) for _ in range(spec.L)]


def _encode_per_site(seqs: np.ndarray, alphabet_list: List[List[str]]) -> np.ndarray:
    """One-hot encode sequences using per-site alphabets.

    The column order at each site follows the order of characters in
    ``alphabet_list[i]``: site ``i`` occupies columns
    ``[sum_{k<i} |alphabet_list[k]|, sum_{k<=i} |alphabet_list[k]|)``, and
    within that block character ``alphabet_list[i][j]`` maps to the ``j``-th
    column.
    """
    strs = [str(s) for s in seqs]
    L = len(alphabet_list)
    n_features = sum(len(a) for a in alphabet_list)
    out = np.zeros((len(strs), n_features), dtype=np.float32)
    offsets = np.cumsum([0] + [len(a) for a in alphabet_list[:-1]])
    lookup = [{ch: j for j, ch in enumerate(chars)} for chars in alphabet_list]
    for r, s in enumerate(strs):
        if len(s) != L:
            raise ValueError(
                f"Sequence {r!r} has length {len(s)} but expected {L} "
                "(matching alphabet_list)"
            )
        for i, ch in enumerate(s):
            try:
                col = lookup[i][ch]
            except KeyError as exc:
                raise ValueError(
                    f"Character {ch!r} at position {i} of sequence {r} "
                    f"is not in alphabet {alphabet_list[i]}"
                ) from exc
            out[r, offsets[i] + col] = 1.0
    return out


def load_dataset(
    dataset: str,
    *,
    device: str = "cuda",
    ptrain: float = 1.0,
    seed: int = 0,
    epik_flavor: Optional[str] = None,
) -> LoadedDataset:
    """
    Load a dataset and return a ready-to-use EpiK model.

    Uses the epik-dev loading convention::

        model.set_data(X, y, y_var)
        model.load(path)

    Parameters
    ----------
    ptrain : float
        Fraction of data to condition the GP on (1.0 = all data).
    seed : int
        Random seed for the train/test split when ptrain < 1.
    """
    spec = get_dataset_spec(dataset)

    import epik_loader

    flavor = epik_flavor or spec.epik_flavor
    epik_loader.ensure_epik_loaded(epik_flavor=flavor)

    from epik.epik.kernel import ConnectednessKernel, GeneralProductKernel
    from epik.epik.model import EpiK

    data = pd.read_csv(spec.data_csv, index_col=spec.index_col)
    seqs = data.index.to_numpy()
    y = data[spec.y_column].to_numpy(dtype=float) + float(spec.phenotype_offset)
    y_var = np.zeros_like(y)

    # Build per-site alphabets, then one-hot encode sequences according to them.
    # Column order at each site follows the order of characters in spec.alphabet
    # (or spec.alphabet_csv when provided), so X is consistent with what the
    # kernel/model expects after they are constructed from the same alphabets.
    alphabet_list = _build_alphabet_list(spec)
    print(
        f"Loading dataset with alphabet (site 0): {alphabet_list[0]}; "
        f"sites: {len(alphabet_list)}"
    )
    X = torch.as_tensor(
        _encode_per_site(seqs, alphabet_list),
        dtype=torch.float32,
        device=device,
    )

    # Build kernel + model using the same alphabet_list used to encode X above.
    kernel_classes = {
        "GeneralProduct": GeneralProductKernel,
        "Connectedness": ConnectednessKernel,
    }
    if spec.kernel_type not in kernel_classes:
        raise ValueError(
            f"Unsupported kernel_type '{spec.kernel_type}'. "
            f"Supported: {sorted(kernel_classes)}"
        )
    kernel_cls = kernel_classes[spec.kernel_type]

    kernel = kernel_cls(alphabet_list=alphabet_list, use_keops=spec.use_keops)
    model = EpiK(
        kernel,
        alphabet_list=alphabet_list,
        train_noise=spec.train_noise,
        train_mean=spec.train_mean,
        device=device,
    )

    # Condition model on (optionally subsampled) training data
    if ptrain < 1.0:
        np.random.seed(seed)
        mask = np.random.uniform(size=len(seqs)) <= ptrain
        train_seqs = seqs[mask]
        model.set_data(train_seqs, y[mask], y_var[mask])
    else:
        train_seqs = seqs
        print("removing last sequence")
        model.set_data(seqs[:-1], y[:-1], y_var[:-1]) #REMOVE THIS JUST A TEST FOR NOW

    model.load(str(spec.model_params))

    # Populate prediction caches (needed by KernelTrickComputer)
    model.predict(seqs[:-1], calc_variance=False) #REMOVE THIS JUST A TEST FOR NOW

    return LoadedDataset(
        spec=spec,
        seqs=seqs,
        train_seqs=train_seqs,
        X=X,
        y=y,
        y_var=y_var,
        kernel=kernel,
        model=model,
    )
