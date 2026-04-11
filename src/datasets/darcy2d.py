from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_mat_file(path: str) -> dict[str, np.ndarray]:
    """Load a .mat file and drop scipy-internal keys (``__header__``, etc.)."""
    raw = scipy.io.loadmat(path)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def resize_field_batch(
    x: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize a batch of 2D scalar fields.

    Args:
        x: Tensor of shape ``[N, H, W]``.
        size: Target ``(height, width)``.
        mode: ``interpolate`` mode (default bilinear).

    Returns:
        Tensor of shape ``[N, size[0], size[1]]``.
    """
    x = x.unsqueeze(1)
    x = F.interpolate(x, size=size, mode=mode, align_corners=False)
    return x.squeeze(1)


def make_coord_grid(
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Normalized coordinate grid in ``[0, 1]``.

    Returns:
        Tensor of shape ``[H, W, 2]`` with channels ``(x, y)`` (column, row).
    """
    ys = torch.linspace(0.0, 1.0, height, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


class UnitGaussianNormalizer:
    """
    Global mean/std normalization (scalar statistics over all elements):

        ``x_norm = (x - mean) / (std + eps)``
    """

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        self.mean = x.mean()
        self.std = x.std()
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * (std + self.eps) + mean

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state["eps"]


@dataclass
class DarcySplit:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    x_normalizer: UnitGaussianNormalizer
    y_normalizer: UnitGaussianNormalizer


class Darcy2DDataset(Dataset):
    """Pairs of input fields ``x`` and target fields ``y`` (both ``[N, H, W, C]``)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors, got x.ndim={x.ndim}, y.ndim={y.ndim}"
            )
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _append_coord_channels(
    coeff_ch: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Concatenate coefficient channel(s) with a per-batch coordinate grid.

    Uses ``expand`` instead of ``repeat`` so the grid shares storage until
    downstream ops materialize a copy.
    """
    coord_hw = make_coord_grid(height, width, dtype=coeff_ch.dtype)
    coord_bhw = coord_hw.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return torch.cat((coeff_ch, coord_bhw), dim=-1)


def _apply_coeff_normalization(
    batch_x: torch.Tensor,
    normalizer: UnitGaussianNormalizer,
    has_coord_channels: bool,
) -> torch.Tensor:
    """Normalize only the first channel (coefficient); leave coords in ``[0, 1]``."""
    coeff_norm = normalizer.encode(batch_x[..., :1])
    if has_coord_channels:
        return torch.cat((coeff_norm, batch_x[..., 1:]), dim=-1)
    return coeff_norm


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
    """
    Build train/validation tensors from a Darcy ``coeff`` -> ``sol`` .mat file.

    Default ``n_train`` / ``n_val`` sum to 1024 (typical bundled Darcy2D size).
    Any remaining samples after the shuffle are unused.

    Input channels (when ``use_coord``):
        - coefficient field
        - x, y coordinates in ``[0, 1]``

    Output channels:
        - solution field
    """
    data = load_mat_file(mat_path)

    coeff = torch.from_numpy(data["coeff"]).float()
    sol = torch.from_numpy(data["sol"]).float()

    n_total = coeff.shape[0]
    n_used = n_train + n_val
    if n_used > n_total:
        raise ValueError(f"split size ({n_used}) exceeds dataset size ({n_total})")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    coeff = coeff[perm]
    sol = sol[perm]

    if target_size is not None:
        th, tw = target_size
        coeff = resize_field_batch(coeff, (th, tw), mode="bilinear")
        sol = resize_field_batch(sol, (th, tw), mode="bilinear")

    n_samples, height, width = coeff.shape

    # Full batch: coefficient (+ optional coords); solution with trailing channel dim.
    inputs = coeff.unsqueeze(-1)
    targets = sol.unsqueeze(-1)

    if use_coord:
        inputs = _append_coord_channels(inputs, n_samples, height, width)

    train_x = inputs[:n_train]
    train_y = targets[:n_train]
    val_x = inputs[n_train:n_used]
    val_y = targets[n_train:n_used]

    x_normalizer = UnitGaussianNormalizer(train_x[..., :1])
    y_normalizer = UnitGaussianNormalizer(train_y)

    if normalize_x:
        train_x = _apply_coeff_normalization(train_x, x_normalizer, use_coord)
        val_x = _apply_coeff_normalization(val_x, x_normalizer, use_coord)

    if normalize_y:
        train_y = y_normalizer.encode(train_y)
        val_y = y_normalizer.encode(val_y)

    return DarcySplit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )
