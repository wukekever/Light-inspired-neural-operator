from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def set_publication_style():
    """Set a clean publication-style matplotlib configuration."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def beautify_axis(ax):
    """Remove unnecessary spines and use outward ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def beautify_image_axis(ax):
    """Clean image axes while keeping the physical domain implicit."""
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])


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

    set_publication_style()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if split.spatial_dims == 1:
        input_norm = x[0, ..., 0].detach().cpu()
        init_phys = split.x_normalizer.decode(input_norm.unsqueeze(-1)).squeeze(-1)

        xs = torch.linspace(0.0, 1.0, init_phys.shape[0])
        gt = y_phys[0, ..., 0]
        pred_1d = pred_phys[0, ..., 0]
        err_1d = pred_1d - gt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)

        ax = axes[0]
        ax.plot(
            xs.numpy(),
            init_phys.numpy(),
            color="0.55",
            linestyle=(0, (4, 2)),
            linewidth=1.6,
            label="Initial condition",
        )
        ax.plot(
            xs.numpy(),
            gt.numpy(),
            color="black",
            linestyle="-",
            linewidth=2.2,
            label="Ground truth",
        )
        ax.plot(
            xs.numpy(),
            pred_1d.numpy(),
            color="#1f77b4",
            linestyle="--",
            linewidth=2.0,
            label="Prediction",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(x)$")
        ax.set_title("Solution")
        beautify_axis(ax)
        ax.legend(frameon=False, loc="best", handlelength=2.8)
        ax.text(
            -0.14,
            1.03,
            "(a)",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )

        ax = axes[1]
        ax.plot(
            xs.numpy(),
            err_1d.numpy(),
            color="#b22222",
            linestyle="-",
            linewidth=1.8,
        )
        ax.axhline(0.0, color="0.7", linestyle=":", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel("Error")
        ax.set_title("Pointwise error")
        beautify_axis(ax)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax.text(
            -0.14,
            1.03,
            "(b)",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )

    elif dataset_name == "darcy2d":
        coeff_norm = x[0, ..., 0].detach().cpu()
        coeff_phys = split.x_normalizer.decode(coeff_norm.unsqueeze(-1)).squeeze(-1)

        target_field = y_phys[0, ..., 0]
        pred_field = pred_phys[0, ..., 0]
        err_field = pred_field - target_field

        sol_min = min(target_field.min().item(), pred_field.min().item())
        sol_max = max(target_field.max().item(), pred_field.max().item())
        err_abs = torch.abs(err_field).max().item()

        fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5), constrained_layout=True)

        panels = [
            (coeff_phys, "Coefficient", "viridis", None, None),
            (target_field, "Ground truth", "viridis", sol_min, sol_max),
            (pred_field, "Prediction", "viridis", sol_min, sol_max),
            (err_field, "Error", "viridis", -err_abs, err_abs),
        ]
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

        for ax, (field, title, cmap, vmin, vmax), panel_label in zip(
            axes, panels, panel_labels
        ):
            im = ax.imshow(
                field.numpy(),
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ax.set_title(title, pad=6)
            beautify_image_axis(ax)

            ax.text(
                -0.12,
                1.03,
                panel_label,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
            )

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8, width=0.6, length=2)

            if title == "Error":
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 0))
                cbar.formatter = formatter
                cbar.update_ticks()

    else:  # NavierStokes2D
        t_in = int(cfg["t_in"])
        t_out = int(y_phys.shape[-1])

        preferred_times = [t_in + 2, t_in + 4, t_in + 6, t_in + 8]
        available_times = list(range(t_in + 1, t_in + t_out + 1))
        selected_times = [tt for tt in preferred_times if tt in available_times]

        # If the current rollout horizon cannot cover all preferred times,
        # choose up to four approximately uniformly spaced predicted frames.
        if len(selected_times) < min(4, t_out):
            frame_ids = (
                torch.linspace(0, t_out - 1, steps=min(4, t_out)).long().tolist()
            )
            selected_times = [t_in + fid + 1 for fid in frame_ids]
        else:
            frame_ids = [tt - t_in - 1 for tt in selected_times]

        # Decode the first input-history channel as the initial vorticity.
        # Coordinate channels, when enabled, live after the first t_in channels.
        input_hist_norm = x[0, ..., :t_in].detach().cpu()
        input_hist_phys = split.x_normalizer.decode(input_hist_norm.unsqueeze(0))[0]
        init_vorticity = input_hist_phys[..., 0]

        gt_fields = y_phys[0]
        pred_fields = pred_phys[0]

        # Use one common color range across initial, ground truth, and prediction
        # so visual differences are not artifacts of per-panel normalization.
        all_fields = [init_vorticity]
        for fid in frame_ids:
            all_fields.append(gt_fields[..., fid])
            all_fields.append(pred_fields[..., fid])
        vmin = min(field.min().item() for field in all_fields)
        vmax = max(field.max().item() for field in all_fields)

        ncols = 1 + len(frame_ids)
        fig = plt.figure(figsize=(3.0 * ncols, 5.2))
        gs = fig.add_gridspec(
            2,
            ncols,
            width_ratios=[1.0] + [1.0] * len(frame_ids),
            height_ratios=[1.0, 1.0],
            wspace=0.35,
            hspace=0.15,
        )

        cmap = "jet"

        def _format_snapshot_axis(ax):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        ax_init = fig.add_subplot(gs[0, 0])
        ax_init.imshow(
            init_vorticity.numpy(),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax_init.set_title("Initial Vorticity", fontsize=18, pad=14)
        _format_snapshot_axis(ax_init)

        ax_label = fig.add_subplot(gs[1, 0])
        ax_label.axis("off")
        ax_label.text(
            0.10,
            0.50,
            "Prediction",
            fontsize=18,
            ha="left",
            va="center",
            transform=ax_label.transAxes,
        )

        for j, (frame_id, time_label) in enumerate(
            zip(frame_ids, selected_times), start=1
        ):
            ax_gt = fig.add_subplot(gs[0, j])
            ax_pred = fig.add_subplot(gs[1, j])

            ax_gt.imshow(
                gt_fields[..., frame_id].numpy(),
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ax_pred.imshow(
                pred_fields[..., frame_id].numpy(),
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )

            ax_gt.set_title(
                rf"$t={time_label}$", fontsize=18, fontstyle="italic", pad=14
            )
            _format_snapshot_axis(ax_gt)
            _format_snapshot_axis(ax_pred)

        # caption = (
        #     rf"Navier-Stokes vorticity rollout: ground truth on top and "
        #     rf"prediction on bottom; relative $L^2$ error = {rel_err:.3e}."
        # )
        # fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=14)
        plt.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.12)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved evaluation plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Light Neural Operator model")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="evaluation_results.png")
    parser.add_argument("--device", type=str, default=None)
    main(parser.parse_args())
