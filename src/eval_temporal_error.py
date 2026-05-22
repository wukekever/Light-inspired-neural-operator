from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from termcolor import colored

from datasets import OperatorDataset, build_navierstokes2d_splits
from modules.model import LightNeuralOperator
from utils import resolve_mat_path


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str, username: str = "wukekever") -> None:
    timestamp = get_timestamp()
    formatted_msg = (
        colored(f"[⏳ {timestamp}]", "light_cyan")
        + colored(f"[🤖 {username}]", "blue")
        + colored(f": {message}", "magenta")
    )
    print(formatted_msg)


@torch.no_grad()
def rollout_autoregressive(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    t_in: int,
    step: int,
    use_coord: bool,
) -> torch.Tensor:
    """
    Autoregressive rollout for Navier--Stokes.

    Args:
        model: one-step or multi-step predictor.
        x: normalized input, shape [B, H, W, t_in + coord_dims].
        y: normalized target, shape [B, H, W, t_out].
        t_in: number of history frames.
        step: number of frames predicted per model call.
        use_coord: whether coordinate channels are appended after history frames.

    Returns:
        pred_full: normalized prediction, shape [B, H, W, t_out].
    """
    coeff_channels = t_in
    coord_channels = x[..., coeff_channels:] if use_coord else None

    state = x.clone()
    pred_steps = []
    total_out = y.shape[-1]

    for t in range(0, total_out, step):
        pred_step = model(state)

        # Safety: if the last interval is shorter than step, crop the output.
        remaining = total_out - t
        if pred_step.shape[-1] > remaining:
            pred_step = pred_step[..., :remaining]

        pred_steps.append(pred_step)

        next_history = torch.cat((state[..., :coeff_channels], pred_step), dim=-1)
        next_history = next_history[..., -coeff_channels:]

        if use_coord:
            state = torch.cat((next_history, coord_channels), dim=-1)
        else:
            state = next_history

    return torch.cat(pred_steps, dim=-1)


@torch.no_grad()
def relative_l2_per_step(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute sample-wise relative L2 error at each rollout step.

    Args:
        pred, target: physical-scale tensors with shape [B, H, W, T].

    Returns:
        errors: tensor with shape [B, T].
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if pred.ndim != 4:
        raise ValueError(
            f"Expected [B, H, W, T] tensors, got shape {tuple(pred.shape)}"
        )

    diff = pred - target
    diff_norm = torch.sqrt(torch.sum(diff**2, dim=(1, 2)) + eps)
    target_norm = torch.sqrt(torch.sum(target**2, dim=(1, 2)) + eps)
    return diff_norm / target_norm


def build_split_from_config(cfg: dict, use_coord: bool):
    dataset_name = cfg.get("dataset_name")
    if dataset_name != "navierstokes2d":
        raise ValueError(
            f"eval_temporal_error.py only supports navierstokes2d, got {dataset_name!r}"
        )

    target_size = cfg.get("target_size")
    if target_size is not None:
        target_size = tuple(target_size)

    return build_navierstokes2d_splits(
        mat_path=resolve_mat_path(cfg["data_path"]),
        target_size=target_size,
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        t_in=cfg["t_in"],
        t_out=cfg["t_out"],
        use_coord=use_coord,
        normalize_x=cfg.get("normalize_x", True),
        normalize_y=cfg.get("normalize_y", True),
        seed=cfg["seed"],
    )


def build_model_from_checkpoint(ckpt: dict, cfg: dict, split, device: torch.device):
    model_in_channels = ckpt.get(
        "in_channels", cfg.get("in_channels", split.input_channels)
    )
    model_out_channels = ckpt.get(
        "out_channels", cfg.get("out_channels", split.output_channels)
    )
    model_spatial_dims = ckpt.get("spatial_dims", split.spatial_dims)

    model = LightNeuralOperator(
        in_channels=model_in_channels,
        out_channels=model_out_channels,
        spatial_dims=model_spatial_dims,
        num_features=cfg["num_features"],
        depth=cfg["depth"],
        scattering_type=cfg["scattering_type"],
    ).to(device)

    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        state_dict = ckpt.get("model")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict' or 'model'.")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_temporal_error(
    mean_error: np.ndarray, std_error: np.ndarray, output_path: Path
) -> None:
    rollout_steps = np.arange(1, len(mean_error) + 1)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    from matplotlib.ticker import ScalarFormatter

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.plot(
        rollout_steps, mean_error, marker="o", linewidth=2.0, label="Efficient LiNO"
    )
    ax.fill_between(
        rollout_steps,
        mean_error - std_error,
        mean_error + std_error,
        alpha=0.20,
        label=r"$\pm$ one standard deviation",
    )

    ax.set_xlabel("Rollout step")
    ax.set_xticks(np.arange(1, len(mean_error) + 1))
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main(args) -> None:
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    )

    log_info(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    cfg = ckpt.get("config") or ckpt.get("args")
    if cfg is None:
        raise KeyError(
            "Checkpoint does not contain 'config' or 'args'; cannot infer settings."
        )

    use_coord = cfg.get("use_coord")
    if use_coord is None:
        use_coord = not cfg.get("no_coord", False)

    step = int(cfg.get("step", 1))
    t_in = int(cfg["t_in"])

    log_info("Building Navier-Stokes validation split")
    split = build_split_from_config(cfg, use_coord)

    if "x_normalizer" in ckpt:
        split.x_normalizer.load_state_dict(ckpt["x_normalizer"])
    if "y_normalizer" in ckpt:
        split.y_normalizer.load_state_dict(ckpt["y_normalizer"])

    val_set = OperatorDataset(split.val_x, split.val_y)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
        drop_last=False,
    )

    log_info("Initializing model")
    model = build_model_from_checkpoint(ckpt, cfg, split, device)

    all_errors = []

    log_info("Computing step-wise relative L2 errors")
    for batch_id, (x, y) in enumerate(val_loader):
        if args.max_batches is not None and batch_id >= args.max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = rollout_autoregressive(
            model=model,
            x=x,
            y=y,
            t_in=t_in,
            step=step,
            use_coord=use_coord,
        )

        # Decode to physical scale before computing reported errors.
        pred_phys = split.y_normalizer.decode(pred.detach().cpu())
        y_phys = split.y_normalizer.decode(y.detach().cpu())

        step_errors = relative_l2_per_step(pred_phys, y_phys)
        all_errors.append(step_errors)

    if not all_errors:
        raise RuntimeError("No validation batches were evaluated.")

    all_errors = torch.cat(all_errors, dim=0)  # [N_val_eval, T_out]
    mean_error = all_errors.mean(dim=0).numpy()
    std_error = all_errors.std(dim=0).numpy()
    rollout_steps = np.arange(1, len(mean_error) + 1)

    output_path = Path(args.output_path)
    plot_temporal_error(mean_error, std_error, output_path)

    data_path = output_path.with_suffix(".npz")
    np.savez(
        data_path,
        rollout_steps=rollout_steps,
        mean_error=mean_error,
        std_error=std_error,
        all_errors=all_errors.numpy(),
    )

    log_info(f"Saved temporal error figure to: {output_path}")
    log_info(f"Saved temporal error data to: {data_path}")
    log_info(f"Final-step mean relative L2 error: {mean_error[-1]:.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate temporal rollout error for Navier-Stokes."
    )
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument(
        "--output-path", type=str, default="../outputs/ns_temporal_error.png"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
