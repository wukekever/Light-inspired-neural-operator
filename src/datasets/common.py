from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.io
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SpatialSize = tuple[int, ...]


# def load_mat_file(path: str) -> dict[str, np.ndarray]:
#     """Load a .mat file and drop scipy-internal keys (``__header__``, etc.)."""
#     raw = scipy.io.loadmat(path)
#     return {k: v for k, v in raw.items() if not k.startswith("__")}
def load_mat_file(path: str) -> dict[str, np.ndarray]:
    """Load .mat file, supports both legacy and v7.3 formats."""
    try:
        raw = scipy.io.loadmat(path)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        # v7.3 format, use h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]).T for k in f.keys() if not k.startswith("__")}


class UnitGaussianNormalizer:
    """Global scalar mean/std normalizer shared across 1D/2D operators (scalar statistics over all elements):

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
class OperatorSplit:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    x_normalizer: UnitGaussianNormalizer
    y_normalizer: UnitGaussianNormalizer
    spatial_dims: int
    input_channels: int
    output_channels: int


class OperatorDataset(Dataset):
    """Dataset for operator learning tensors in channel-last format.

    Pairs of input fields ``x`` and target fields ``y`` with shapes:
        1D: [N, L, C]
        2D: [N, H, W, C]
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != y.ndim:
            raise ValueError(f"x and y must have same ndim, got {x.ndim} vs {y.ndim}")
        if x.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D tensors, got ndim={x.ndim}")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def resize_field_batch_nd(
    x: torch.Tensor,
    size: int | Sequence[int],
    mode: str,
) -> torch.Tensor:
    """Resize [N, *spatial] scalar fields using PyTorch interpolate (d = 1, 2)."""
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected [N, L] or [N, H, W], got shape {tuple(x.shape)}")

    if isinstance(size, int):
        size = (size,)
    size = tuple(size)

    x = x.unsqueeze(1)
    kwargs = {"size": size, "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    x = F.interpolate(x, **kwargs)
    return x.squeeze(1)


def make_coord_grid_nd(
    spatial_shape: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build normalized coordinate grid over [0, 1]^d in channel-last form."""
    axes = [torch.linspace(0.0, 1.0, n, dtype=dtype) for n in spatial_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1)


def append_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized coordinates to coefficient channels.

    Input shape: [N, *spatial, C]
    Output shape: [N, *spatial, C + spatial_dims]
    """
    spatial_shape = x.shape[1:-1]
    coord = make_coord_grid_nd(spatial_shape, dtype=x.dtype)
    coord = coord.unsqueeze(0).expand(x.shape[0], *([-1] * coord.ndim))
    return torch.cat((x, coord), dim=-1)


def apply_coeff_normalization(
    x: torch.Tensor,
    normalizer: UnitGaussianNormalizer,
    has_coord_channels: bool,
) -> torch.Tensor:
    coeff_norm = normalizer.encode(x[..., :1])
    if has_coord_channels:
        return torch.cat((coeff_norm, x[..., 1:]), dim=-1)
    return coeff_norm


def build_operator_split(
    coeff: torch.Tensor,
    sol: torch.Tensor,
    *,
    target_size: int | Sequence[int] | None,
    interp_mode: str,
    n_train: int,
    n_val: int,
    use_coord: bool,
    normalize_x: bool,
    normalize_y: bool,
    seed: int,
) -> OperatorSplit:
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
    if coeff.shape != sol.shape:
        raise ValueError(
            f"Coefficient and solution shapes must match, got {coeff.shape} vs {sol.shape}"
        )
    if coeff.ndim not in (2, 3):
        raise ValueError(f"Expected [N, L] or [N, H, W], got {tuple(coeff.shape)}")

    n_total = coeff.shape[0]
    n_used = n_train + n_val
    if n_used > n_total:
        raise ValueError(f"split size ({n_used}) exceeds dataset size ({n_total})")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    coeff = coeff[perm]
    sol = sol[perm]

    if target_size is not None:
        coeff = resize_field_batch_nd(coeff, target_size, mode=interp_mode)
        sol = resize_field_batch_nd(sol, target_size, mode=interp_mode)

    spatial_dims = coeff.ndim - 1

    inputs = coeff.unsqueeze(-1)
    targets = sol.unsqueeze(-1)

    if use_coord:
        inputs = append_coord_channels(inputs)

    train_x = inputs[:n_train]
    train_y = targets[:n_train]
    val_x = inputs[n_train:n_used]
    val_y = targets[n_train:n_used]

    x_normalizer = UnitGaussianNormalizer(train_x[..., :1])
    y_normalizer = UnitGaussianNormalizer(train_y)

    if normalize_x:
        train_x = apply_coeff_normalization(train_x, x_normalizer, use_coord)
        val_x = apply_coeff_normalization(val_x, x_normalizer, use_coord)
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
        spatial_dims=spatial_dims,
        input_channels=train_x.shape[-1],
        output_channels=train_y.shape[-1],
    )
