from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from termcolor import colored

from datasets import OperatorDataset, build_darcy2d_splits
from modules.model import LightNeuralOperator
from utils import RelativeL2Loss, resolve_mat_path


def set_publication_style() -> None:
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


def beautify_image_axis(ax) -> None:
    """Clean image axes while keeping the physical domain implicit."""
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str, username: str = "wukekever") -> None:
    """Print formatted log message with colored output."""
    timestamp = get_timestamp()
    formatted_msg = (
        colored(f"[⏳ {timestamp}]", "light_cyan")
        + colored(f"[🤖 {username}]", "blue")
        + colored(f": {message}", "magenta")
    )
    print(formatted_msg)


def _as_target_size(value):
    """Convert checkpoint target_size to the format expected by Darcy2D split builder."""
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
    raise ValueError(f"Invalid Darcy2D target_size stored in checkpoint: {value}")


def _parse_light_components(value):
    """Normalize light component configuration from checkpoint args/config."""
    if value is None:
        return ["reflection", "refraction", "scattering"]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"Invalid light_components type: {type(value)}")


def build_darcy_split(cfg: dict, use_coord: bool):
    """Build Darcy2D validation split from checkpoint configuration."""
    return build_darcy2d_splits(
        mat_path=resolve_mat_path(cfg["data_path"]),
        target_size=_as_target_size(cfg.get("target_size")),
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        use_coord=use_coord,
        normalize_x=cfg.get("normalize_x", True),
        normalize_y=cfg.get("normalize_y", True),
        seed=cfg["seed"],
    )


def build_model_from_checkpoint(ckpt: dict, cfg: dict, split, device: torch.device):
    """Reconstruct the exact Darcy2D LiNO/LiNO-ablation model used in training."""
    model_in_channels = ckpt.get(
        "in_channels", cfg.get("in_channels", split.input_channels)
    )
    model_out_channels = ckpt.get(
        "out_channels", cfg.get("out_channels", split.output_channels)
    )
    model_spatial_dims = ckpt.get("spatial_dims", split.spatial_dims)
    light_components = _parse_light_components(cfg.get("light_components"))

    log_info(
        "Model settings: "
        f"dataset=darcy2d, "
        f"spatial_dims={model_spatial_dims}, "
        f"in_channels={model_in_channels}, "
        f"out_channels={model_out_channels}, "
        f"num_features={cfg['num_features']}, "
        f"depth={cfg['depth']}, "
        f"scattering_type={cfg['scattering_type']}, "
        f"light_components={light_components}"
    )

    model = LightNeuralOperator(
        in_channels=model_in_channels,
        out_channels=model_out_channels,
        spatial_dims=model_spatial_dims,
        num_features=cfg["num_features"],
        depth=cfg["depth"],
        scattering_type=cfg["scattering_type"],
        light_components=light_components,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, light_components


def plot_darcy_sample(
    x, y_phys, pred_phys, split, rel_err: float, output_path: Path
) -> None:
    """Plot coefficient, reference solution, prediction, and pointwise error."""
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
        (err_field, "Pointwise error", "viridis", -err_abs, err_abs),
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

        if "Error" in title:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            cbar.formatter = formatter
            cbar.update_ticks()

        # fig.suptitle(
        #     rf"Relative $L^2$ error = {rel_err:.3e}",
        #     y=1.03,
        #     fontsize=12,
        # )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


@torch.no_grad()
def main(args) -> None:
    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = Path(args.ckpt_path).expanduser()
    log_info(f"Loading checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config") or ckpt.get("args")
    if cfg is None:
        raise KeyError(
            "Checkpoint does not contain 'config' or 'args'; cannot infer dataset/model settings."
        )

    dataset_name = cfg.get("dataset_name")
    if dataset_name != "darcy2d":
        raise ValueError(
            f"This evaluator only supports Darcy2D checkpoints, but got dataset_name={dataset_name!r}."
        )

    use_coord = cfg.get("use_coord")
    if use_coord is None:
        use_coord = not cfg.get("no_coord", False)

    log_info("Building Darcy2D dataset split")
    split = build_darcy_split(cfg=cfg, use_coord=use_coord)

    log_info("Initializing Darcy2D model")
    model, light_components = build_model_from_checkpoint(
        ckpt=ckpt,
        cfg=cfg,
        split=split,
        device=device,
    )

    split.x_normalizer.load_state_dict(ckpt["x_normalizer"])
    split.y_normalizer.load_state_dict(ckpt["y_normalizer"])

    val_set = OperatorDataset(split.val_x, split.val_y)
    if args.index < 0 or args.index >= len(val_set):
        raise IndexError(
            f"index={args.index} is out of range for validation set of size {len(val_set)}"
        )

    x, y = val_set[args.index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    pred = model(x)

    rel_loss_fn = RelativeL2Loss()
    rel_err = rel_loss_fn(pred, y).item()
    log_info(f"Validation sample index={args.index}, relative L2 error={rel_err:.4e}")

    pred_phys = split.y_normalizer.decode(pred.cpu())
    y_phys = split.y_normalizer.decode(y.cpu())

    set_publication_style()
    output_path = Path(args.output_path)
    plot_darcy_sample(
        x=x,
        y_phys=y_phys,
        pred_phys=pred_phys,
        split=split,
        rel_err=rel_err,
        output_path=output_path,
    )

    log_info(f"Saved Darcy2D evaluation plot to {output_path}")
    log_info(f"Light components used for evaluation: {light_components}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Darcy2D LiNO/LiNO-ablation checkpoint"
    )
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="evaluation_results.png")
    parser.add_argument("--device", type=str, default=None)
    main(parser.parse_args())
