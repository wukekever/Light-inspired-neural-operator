from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from datasets.common import (
    OperatorDataset,
    OperatorSplit,
    UnitGaussianNormalizer,
    append_coord_channels,
    load_mat_file,
    resize_field_batch_nd,
)


NavierStokes2DSplit = OperatorSplit
NavierStokes2DDataset = OperatorDataset


def _resize_navierstokes_sequence(
    u: torch.Tensor,
    target_size: Optional[Tuple[int, int]],
) -> torch.Tensor:
    """
    Resize Navier-Stokes trajectories from ``[N, H, W, T]`` to
    ``[N, H_new, W_new, T]`` by resizing each time slice independently.
    """
    if target_size is None:
        return u
    if u.ndim != 4:
        raise ValueError(f"Expected u with shape [N, H, W, T], got {tuple(u.shape)}")

    n_samples, _, _, n_time = u.shape
    u_perm = u.permute(0, 3, 1, 2).reshape(n_samples * n_time, u.shape[1], u.shape[2])
    u_resized = resize_field_batch_nd(u_perm, target_size, mode="bilinear")
    h_new, w_new = u_resized.shape[1], u_resized.shape[2]
    u_resized = u_resized.reshape(n_samples, n_time, h_new, w_new).permute(0, 2, 3, 1)
    return u_resized.contiguous()


def build_navierstokes2d_splits(
    mat_path: str,
    input_key: str = "u",
    target_size: Optional[Tuple[int, int]] = (64, 64),
    n_train: int = 4000,
    n_val: int = 1000,
    t_in: int = 10,
    t_out: int = 40,
    use_coord: bool = True,
    normalize_x: bool = True,
    normalize_y: bool = True,
    seed: int = 42,
) -> NavierStokes2DSplit:
    """
    Build train/validation splits for time-dependent 2D Navier-Stokes data.

    The dataset is stored as a trajectory tensor ``u`` with shape ``[N, H, W, T]`` and then split into input/output pairs
    according to the specified ``t_in`` and ``t_out``.
    Following the autoregressive FNO setup, the learning target is:

        input  = u[..., :t_in]
        target = u[..., t_in:t_in+t_out]

    Input channels (when ``use_coord``):
        - the previous ``t_in`` solution frames,
        - x, y coordinates in ``[0, 1]``.

    Output channels:
        - the next ``t_out`` solution frames.
    """
    data = load_mat_file(mat_path)
    if input_key not in data:
        raise KeyError(
            f"Missing key {input_key!r} in {mat_path!r}; got {sorted(data.keys())}"
        )

    u = torch.from_numpy(data[input_key]).float()
    if u.ndim != 4:
        raise ValueError(
            f"Expected Navier-Stokes tensor [N, H, W, T], got shape {tuple(u.shape)}"
        )
    n_total, _, _, total_steps = u.shape
    if t_in <= 0 or t_out <= 0:
        raise ValueError(f"t_in and t_out must be positive, got {t_in}, {t_out}")
    if t_in + t_out > total_steps:
        raise ValueError(
            f"Requested t_in + t_out = {t_in + t_out} exceeds available steps {total_steps}"
        )

    n_used = n_train + n_val
    if n_used > n_total:
        raise ValueError(f"split size ({n_used}) exceeds dataset size ({n_total})")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    u = u[perm]

    if target_size is not None:
        u = _resize_navierstokes_sequence(u, target_size)

    inputs = u[..., :t_in]
    targets = u[..., t_in : t_in + t_out]

    if use_coord:
        inputs = append_coord_channels(inputs)

    train_x = inputs[:n_train]
    train_y = targets[:n_train]
    val_x = inputs[n_train:n_used]
    val_y = targets[n_train:n_used]

    coeff_channels = t_in
    x_normalizer = UnitGaussianNormalizer(train_x[..., :coeff_channels])
    y_normalizer = UnitGaussianNormalizer(train_y)

    if normalize_x:
        coeff_norm = x_normalizer.encode(train_x[..., :coeff_channels])
        if use_coord:
            train_x = torch.cat((coeff_norm, train_x[..., coeff_channels:]), dim=-1)
        else:
            train_x = coeff_norm

        coeff_norm = x_normalizer.encode(val_x[..., :coeff_channels])
        if use_coord:
            val_x = torch.cat((coeff_norm, val_x[..., coeff_channels:]), dim=-1)
        else:
            val_x = coeff_norm

    if normalize_y:
        train_y = y_normalizer.encode(train_y)
        val_y = y_normalizer.encode(val_y)

    return NavierStokes2DSplit(
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
    "UnitGaussianNormalizer",
    "NavierStokes2DSplit",
    "NavierStokes2DDataset",
    "build_navierstokes2d_splits",
    "load_mat_file",
]
