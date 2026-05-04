from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from datasets import (
    OperatorDataset,
    build_burgers1d_splits,
    build_darcy2d_splits,
    build_navierstokes2d_splits,
)
from modules.model import LightNeuralOperator
from utils import RelativeL2Loss, resolve_mat_path
from termcolor import colored


def get_timestamp():
    """Get current timestamp in format YYYY-MM-DD HH:MM:SS"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str, username: str = "wukekever") -> None:
    """Print formatted log message with colored output"""
    timestamp = get_timestamp()
    formatted_msg = (
        colored(f"[⏳ {timestamp}]", "light_cyan")
        + colored(f"[🤖 {username}]", "blue")
        + colored(f": {message}", "magenta")
    )
    print(formatted_msg)


@torch.no_grad()
def rollout_autoregressive(model, x, y, t_in, step, use_coord):
    coeff_channels = t_in
    coord_channels = x[..., coeff_channels:] if use_coord else None
    state = x.clone()
    pred_steps = []

    total_out = y.shape[-1]
    for t in range(0, total_out, step):
        pred_step = model(state)
        pred_steps.append(pred_step)

        next_history = torch.cat((state[..., :coeff_channels], pred_step), dim=-1)
        next_history = next_history[..., -coeff_channels:]
        if use_coord:
            state = torch.cat((next_history, coord_channels), dim=-1)
        else:
            state = next_history

    return torch.cat(pred_steps, dim=-1)


@torch.no_grad()
def main(args) -> None:
    ckpt_path = args.ckpt_path
    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    log_info(f"Loading checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config") or ckpt.get("args")
    if cfg is None:
        raise KeyError(
            "Checkpoint does not contain 'config' or 'args'; cannot infer dataset/model settings."
        )

    use_coord = cfg.get("use_coord")
    if use_coord is None:
        use_coord = not cfg.get("no_coord", False)

    log_info("Building dataset splits")
    dataset_name = cfg.get("dataset_name")
    if dataset_name == "burgers1d":
        split = build_burgers1d_splits(
            mat_path=resolve_mat_path(cfg["data_path"]),
            target_size=cfg.get("target_size"),
            n_train=cfg["n_train"],
            n_val=cfg["n_val"],
            use_coord=use_coord,
            normalize_x=cfg.get("normalize_x", True),
            normalize_y=cfg.get("normalize_y", True),
            seed=cfg["seed"],
        )
    elif dataset_name == "darcy2d":
        split = build_darcy2d_splits(
            mat_path=resolve_mat_path(cfg["data_path"]),
            target_size=tuple(cfg.get("target_size"))
            if cfg.get("target_size") is not None
            else None,
            n_train=cfg["n_train"],
            n_val=cfg["n_val"],
            use_coord=use_coord,
            normalize_x=cfg.get("normalize_x", True),
            normalize_y=cfg.get("normalize_y", True),
            seed=cfg["seed"],
        )
    elif dataset_name == "navierstokes2d":
        split = build_navierstokes2d_splits(
            mat_path=resolve_mat_path(cfg["data_path"]),
            target_size=tuple(cfg.get("target_size"))
            if cfg.get("target_size") is not None
            else None,
            n_train=cfg["n_train"],
            n_val=cfg["n_val"],
            t_in=cfg["t_in"],
            t_out=cfg["t_out"],
            use_coord=use_coord,
            normalize_x=cfg.get("normalize_x", True),
            normalize_y=cfg.get("normalize_y", True),
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    log_info("Initializing model")
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    split.x_normalizer.load_state_dict(ckpt["x_normalizer"])
    split.y_normalizer.load_state_dict(ckpt["y_normalizer"])

    val_set = OperatorDataset(split.val_x, split.val_y)
    x, y = val_set[args.index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    if dataset_name == "navierstokes2d":
        pred = rollout_autoregressive(
            model=model,
            x=x,
            y=y,
            t_in=cfg["t_in"],
            step=cfg.get("step", 1),
            use_coord=use_coord,
        )
    else:
        pred = model(x)

    rel_loss_fn = RelativeL2Loss()
    rel_err = rel_loss_fn(pred, y).item()
    pred_phys = split.y_normalizer.decode(pred.cpu())
    y_phys = split.y_normalizer.decode(y.cpu())

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if split.spatial_dims == 1:
        coeff_norm = x[0, ..., 0].detach().cpu()
        coeff_phys = split.x_normalizer.decode(coeff_norm.unsqueeze(-1)).squeeze(-1)
        xs = torch.linspace(0.0, 1.0, coeff_phys.shape[0])
        plt.figure(figsize=(10, 4))
        plt.plot(xs.numpy(), coeff_phys.numpy(), label="Input coeff")
        plt.plot(xs.numpy(), y_phys[0, ..., 0].numpy(), label="Ground truth")
        plt.plot(xs.numpy(), pred_phys[0, ..., 0].numpy(), label="Prediction")
        plt.title(f"Burgers1D evaluation | rel-L2={rel_err:.3e}")
        plt.legend()
        plt.tight_layout()
    elif dataset_name == "darcy2d":
        coeff_norm = x[0, ..., 0].detach().cpu()
        coeff_phys = split.x_normalizer.decode(coeff_norm.unsqueeze(-1)).squeeze(-1)
        target_field = y_phys[0, ..., 0]
        pred_field = pred_phys[0, ..., 0]
        err_field = pred_field - target_field

        fig = plt.figure(figsize=(16, 4))
        for i, (field, title) in enumerate(
            [
                (coeff_phys, "Coefficient"),
                (target_field, "Ground Truth"),
                (pred_field, "Prediction"),
                (err_field, "Error"),
            ],
            start=1,
        ):
            ax = fig.add_subplot(1, 4, i)
            im = ax.imshow(field.numpy(), origin="lower")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle(f"Darcy2D evaluation | rel-L2={rel_err:.3e}")
        plt.tight_layout()
    else:  # NavierStokes2D
        t_out = y_phys.shape[-1]
        frame_ids = sorted({0, max(0, t_out // 2), t_out - 1})
        frame_ids = [i for i in range(0, t_out, 1)]  # Every 2 frames
        fig = plt.figure(figsize=(4 * len(frame_ids), 12))
        for row, (name, fields) in enumerate(
            [
                ("Ground Truth", y_phys[0]),
                ("Prediction", pred_phys[0]),
                ("Error", pred_phys[0] - y_phys[0]),
            ],
            start=0,
        ):
            for col, frame_id in enumerate(frame_ids, start=0):
                ax = fig.add_subplot(3, len(frame_ids), row * len(frame_ids) + col + 1)
                im = ax.imshow(fields[..., frame_id].numpy(), origin="lower")
                ax.set_title(f"{name} | t={frame_id + 1}")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle(f"NavierStokes2D rollout evaluation | rel-L2={rel_err:.3e}")
        plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved evaluation plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Light Neural Operator model")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="evaluation_results.png")
    parser.add_argument("--device", type=str, default=None)
    main(parser.parse_args())
