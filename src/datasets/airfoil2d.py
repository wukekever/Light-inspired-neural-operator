from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from datasets.common import (
    OperatorDataset,
    OperatorSplit,
    UnitGaussianNormalizer,
    make_coord_grid_nd,
    resize_field_batch_nd,
)


Airfoil2DSplit = OperatorSplit
Airfoil2DDataset = OperatorDataset


def _resolve_npy_file(data_path: str | Path, filename: str) -> Path:
    """Resolve a Geo-FNO airfoil .npy file from either a directory or file path."""
    path = Path(data_path).expanduser()
    if path.is_dir():
        candidate = path / filename
    else:
        # If a concrete X/Y/Q file is passed, use its parent as the data directory.
        candidate = path.parent / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Cannot find {filename!r}. Expected it under {path if path.is_dir() else path.parent}."
        )
    return candidate.resolve()


def _load_airfoil_arrays(data_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Geo-FNO NACA airfoil arrays.

    Expected files:
        NACA_Cylinder_X.npy: [N, nx, ny]
        NACA_Cylinder_Y.npy: [N, nx, ny]
        NACA_Cylinder_Q.npy: [N, C, nx, ny] or [N, nx, ny, C]
    """
    x_path = _resolve_npy_file(data_path, "NACA_Cylinder_X.npy")
    y_path = _resolve_npy_file(data_path, "NACA_Cylinder_Y.npy")
    q_path = _resolve_npy_file(data_path, "NACA_Cylinder_Q.npy")

    x = torch.from_numpy(np.load(x_path)).float()
    y = torch.from_numpy(np.load(y_path)).float()
    q = torch.from_numpy(np.load(q_path)).float()

    if x.shape != y.shape:
        raise ValueError(f"X and Y mesh arrays must have the same shape, got {x.shape} vs {y.shape}")
    if x.ndim != 3:
        raise ValueError(f"Expected X/Y arrays with shape [N, nx, ny], got {tuple(x.shape)}")
    if q.ndim not in (3, 4):
        raise ValueError(f"Expected Q array with shape [N, nx, ny] or [N, C, nx, ny], got {tuple(q.shape)}")
    return x, y, q


def _extract_target_channel(q: torch.Tensor, target_channel: int) -> torch.Tensor:
    """Extract the scalar target field from Geo-FNO's conservative-variable tensor."""
    if q.ndim == 3:
        return q

    # Geo-FNO stores Q as [N, C, nx, ny] and uses Q[:, 4] for the Mach-number field.
    if q.shape[1] > target_channel:
        return q[:, target_channel]

    # Be robust to channel-last exports: [N, nx, ny, C].
    if q.shape[-1] > target_channel:
        return q[..., target_channel]

    raise ValueError(
        f"target_channel={target_channel} is unavailable for Q shape {tuple(q.shape)}"
    )


def _normalize_geometry_channels(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    normalizer: UnitGaussianNormalizer,
    geom_channels: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize physical geometry channels and keep computational coordinates unchanged."""
    train_geom = normalizer.encode(train_x[..., :geom_channels])
    val_geom = normalizer.encode(val_x[..., :geom_channels])
    train_x = torch.cat((train_geom, train_x[..., geom_channels:]), dim=-1)
    val_x = torch.cat((val_geom, val_x[..., geom_channels:]), dim=-1)
    return train_x, val_x


def build_airfoil2d_splits(
    data_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    n_train: int = 1000,
    n_val: int = 200,
    use_coord: bool = True,
    normalize_x: bool = True,
    normalize_y: bool = True,
    seed: int = 42,
    target_channel: int = 4,
    shuffle: bool = False,
) -> Airfoil2DSplit:
    """Build train/validation splits for the Geo-FNO NACA airfoil benchmark.

    Operator-learning task:
        input  a(i,j) = [x(i,j), y(i,j), xi(i,j), eta(i,j)] if use_coord else [x, y]
        target u(i,j) = Mach-number-like scalar field Q[:, target_channel]

    The canonical Geo-FNO split uses the first 1000 samples for training and the next
    200 samples for testing/validation, so ``shuffle=False`` by default.
    """
    mesh_x, mesh_y, q = _load_airfoil_arrays(data_path)
    target = _extract_target_channel(q, target_channel=target_channel)

    if mesh_x.shape != target.shape:
        raise ValueError(
            f"Mesh and target spatial shapes must match, got mesh {tuple(mesh_x.shape)} and target {tuple(target.shape)}"
        )

    n_total = mesh_x.shape[0]
    n_used = n_train + n_val
    if n_used > n_total:
        raise ValueError(f"split size ({n_used}) exceeds dataset size ({n_total})")

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=generator)
        mesh_x = mesh_x[perm]
        mesh_y = mesh_y[perm]
        target = target[perm]

    if target_size is not None:
        mesh_x = resize_field_batch_nd(mesh_x, target_size, mode="bilinear")
        mesh_y = resize_field_batch_nd(mesh_y, target_size, mode="bilinear")
        target = resize_field_batch_nd(target, target_size, mode="bilinear")

    inputs = torch.stack((mesh_x, mesh_y), dim=-1)  # [N, nx, ny, 2]
    targets = target.unsqueeze(-1)                 # [N, nx, ny, 1]

    if use_coord:
        coord = make_coord_grid_nd(inputs.shape[1:-1], dtype=inputs.dtype)
        coord = coord.unsqueeze(0).expand(inputs.shape[0], *coord.shape)
        inputs = torch.cat((inputs, coord), dim=-1)  # [N, nx, ny, 4]

    train_x = inputs[:n_train]
    train_y = targets[:n_train]
    val_x = inputs[n_train:n_used]
    val_y = targets[n_train:n_used]

    # Normalize only physical mesh coordinates x,y; keep xi,eta in [0,1].
    x_normalizer = UnitGaussianNormalizer(train_x[..., :2])
    y_normalizer = UnitGaussianNormalizer(train_y)

    if normalize_x:
        train_x, val_x = _normalize_geometry_channels(train_x, val_x, x_normalizer, geom_channels=2)
    if normalize_y:
        train_y = y_normalizer.encode(train_y)
        val_y = y_normalizer.encode(val_y)

    return OperatorSplit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        spatial_dims=2,
        input_channels=train_x.shape[-1],
        output_channels=train_y.shape[-1],
    )


__all__ = [
    "Airfoil2DDataset",
    "Airfoil2DSplit",
    "build_airfoil2d_splits",
]
