from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.darcy2d import UnitGaussianNormalizer, load_mat_file


def resize_field_batch_1d(
    x: torch.Tensor,
    length: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    Resize a batch of 1D scalar signals.

    Args:
        x: Tensor of shape ``[N, L]``.
        length: Target number of grid points.
        mode: ``interpolate`` mode (default ``linear``).

    Returns:
        Tensor of shape ``[N, length]``.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected [N, L], got shape {tuple(x.shape)}")
    x = x.unsqueeze(1)
    x = F.interpolate(x, size=length, mode=mode, align_corners=False)
    return x.squeeze(1)


def make_coord_grid_1d(length: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Normalized coordinates in ``[0, 1]`` along the line.

    Returns:
        Tensor of shape ``[L, 1]`` (single channel for concatenation).
    """
    xs = torch.linspace(0.0, 1.0, length, dtype=dtype)
    return xs.unsqueeze(-1)


@dataclass
class BurgersSplit:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    x_normalizer: UnitGaussianNormalizer
    y_normalizer: UnitGaussianNormalizer


class Burgers1DDataset(Dataset):
    """Pairs of input sequences ``x`` and target sequences ``y`` (both ``[N, L, C]``)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(
                f"Expected 3D tensors [N, L, C], got x.ndim={x.ndim}, y.ndim={y.ndim}"
            )
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _append_coord_channels_1d(
    coeff_ch: torch.Tensor,
    batch_size: int,
    length: int,
) -> torch.Tensor:
    """Concatenate coefficient channel(s) with a shared coordinate channel in ``[0, 1]``."""
    coord_l = make_coord_grid_1d(length, dtype=coeff_ch.dtype)
    coord_bl = coord_l.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat((coeff_ch, coord_bl), dim=-1)


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


def build_burgers1d_splits(
    mat_path: str,
    input_key: str = "a",
    target_key: str = "u",
    target_length: Optional[int] = None,
    n_train: int = 1600,
    n_val: int = 448,
    use_coord: bool = True,
    normalize_x: bool = True,
    normalize_y: bool = True,
    seed: int = 42,
) -> BurgersSplit:
    """
    Build train/validation tensors from Burgers1D ``.mat`` (operator ``input_key`` → ``target_key``).

    Default split uses all 2048 samples (``1600 + 448``). Remaining indices are unused if
    ``n_train + n_val`` is smaller than the file size.

    Tensor layout (operator-friendly):
        - ``x``: ``[N, L, C]`` — coefficient (and optional coordinate in ``[0, 1]``).
        - ``y``: ``[N, L, 1]`` — solution on the same 1D grid.

    Args:
        mat_path: Path to ``burgers_data_R10.mat`` (or compatible layout).
        input_key: Field used as input, e.g. ``\"a\"`` or ``\"a_smooth\"``.
        target_key: Field used as target, typically ``\"u\"``.
        target_length: If set, resample each signal to this length (linear interpolation).
        n_train: Training batch count after shuffle.
        n_val: Validation batch count after shuffle.
        use_coord: If True, append an ``x \\in [0,1]`` channel to inputs.
        normalize_x: If True, Gaussian-normalize the first input channel using train stats.
        normalize_y: If True, Gaussian-normalize targets using train stats.
        seed: RNG seed for the train/val shuffle.
    """
    data = load_mat_file(mat_path)
    if input_key not in data or target_key not in data:
        raise KeyError(
            f"Missing keys in {mat_path!r}: need {input_key!r} and {target_key!r}, "
            f"got {sorted(data.keys())}"
        )

    coeff = torch.from_numpy(data[input_key]).float()
    sol = torch.from_numpy(data[target_key]).float()

    if coeff.ndim != 2 or sol.ndim != 2:
        raise ValueError(
            f"Expected [N, L] arrays for {input_key} and {target_key}, "
            f"got {input_key}={tuple(coeff.shape)}, {target_key}={tuple(sol.shape)}"
        )
    if coeff.shape != sol.shape:
        raise ValueError(
            f"{input_key} and {target_key} must match in shape, "
            f"got {tuple(coeff.shape)} vs {tuple(sol.shape)}"
        )

    n_total = coeff.shape[0]
    n_used = n_train + n_val
    if n_used > n_total:
        raise ValueError(f"split size ({n_used}) exceeds dataset size ({n_total})")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    coeff = coeff[perm]
    sol = sol[perm]

    if target_length is not None:
        coeff = resize_field_batch_1d(coeff, target_length, mode="linear")
        sol = resize_field_batch_1d(sol, target_length, mode="linear")

    n_samples, length = coeff.shape

    inputs = coeff.unsqueeze(-1)
    targets = sol.unsqueeze(-1)

    if use_coord:
        inputs = _append_coord_channels_1d(inputs, n_samples, length)

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

    return BurgersSplit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )
