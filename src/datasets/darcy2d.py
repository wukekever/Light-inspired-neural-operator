from __future__ import annotations

from typing import Optional, Tuple

import torch

from datasets.common import (
    OperatorDataset,
    OperatorSplit,
    UnitGaussianNormalizer,
    build_operator_split,
    load_mat_file,
)


DarcySplit = OperatorSplit
Darcy2DDataset = OperatorDataset


def build_darcy2d_splits(
    mat_path: str,
    target_size: Optional[Tuple[int, int]] = (64, 64),
    n_train: int = 800,
    n_val: int = 224,
    use_coord: bool = True,
    normalize_x: bool = True,
    normalize_y: bool = True,
    seed: int = 42,
) -> DarcySplit:
    data = load_mat_file(mat_path)
    if "coeff" not in data or "sol" not in data:
        raise KeyError(
            f"Missing keys in {mat_path!r}: need 'coeff' and 'sol', got {sorted(data.keys())}"
        )

    coeff = torch.from_numpy(data["coeff"]).float()
    sol = torch.from_numpy(data["sol"]).float()

    return build_operator_split(
        coeff,
        sol,
        target_size=target_size,
        interp_mode="bilinear",
        n_train=n_train,
        n_val=n_val,
        use_coord=use_coord,
        normalize_x=normalize_x,
        normalize_y=normalize_y,
        seed=seed,
    )


__all__ = [
    "UnitGaussianNormalizer",
    "DarcySplit",
    "Darcy2DDataset",
    "build_darcy2d_splits",
    "load_mat_file",
]
