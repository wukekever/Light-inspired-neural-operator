from __future__ import annotations

from datasets.common import (
    OperatorDataset,
    OperatorSplit,
    UnitGaussianNormalizer,
    build_operator_split,
    load_mat_file,
)
import torch


BurgersSplit = OperatorSplit
Burgers1DDataset = OperatorDataset


def build_burgers1d_splits(
    mat_path: str,
    input_key: str = "a",
    target_key: str = "u",
    target_size: int | None = None,
    n_train: int = 1600,
    n_val: int = 448,
    use_coord: bool = True,
    normalize_x: bool = True,
    normalize_y: bool = True,
    seed: int = 42,
) -> BurgersSplit:
    data = load_mat_file(mat_path)
    if input_key not in data or target_key not in data:
        raise KeyError(
            f"Missing keys in {mat_path!r}: need {input_key!r} and {target_key!r}, got {sorted(data.keys())}"
        )

    coeff = torch.from_numpy(data[input_key]).float()
    sol = torch.from_numpy(data[target_key]).float()

    return build_operator_split(
        coeff,
        sol,
        target_size=target_size,
        interp_mode="linear",
        n_train=n_train,
        n_val=n_val,
        use_coord=use_coord,
        normalize_x=normalize_x,
        normalize_y=normalize_y,
        seed=seed,
    )


__all__ = [
    "UnitGaussianNormalizer",
    "BurgersSplit",
    "Burgers1DDataset",
    "build_burgers1d_splits",
    "load_mat_file",
]
