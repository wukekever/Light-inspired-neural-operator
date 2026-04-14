from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from datasets.darcy2d import Darcy2DDataset, build_darcy2d_splits
from modules.model import LightNeuralOperator2D
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
def main(args) -> None:
    ckpt_path = args.ckpt_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    split = build_darcy2d_splits(
        mat_path=resolve_mat_path(cfg["data_path"]),
        target_size=tuple(cfg["target_size"])
        if cfg["target_size"] is not None
        else None,
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        use_coord=use_coord,
        normalize_x=cfg.get("normalize_x", True),
        normalize_y=cfg.get("normalize_y", True),
        seed=cfg["seed"],
    )

    log_info("Initializing model")
    model = LightNeuralOperator2D(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        num_features=cfg["num_features"],
        depth=cfg["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    split.x_normalizer.load_state_dict(ckpt["x_normalizer"])
    split.y_normalizer.load_state_dict(ckpt["y_normalizer"])
    for normalizer in (split.x_normalizer, split.y_normalizer):
        normalizer.mean = normalizer.mean.detach().cpu()
        normalizer.std = normalizer.std.detach().cpu()

    val_set = Darcy2DDataset(split.val_x, split.val_y)

    index = args.index
    log_info(f"Evaluating sample index {index}")
    x, y = val_set[index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    try:
        pred = model(x)
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            log_info("CUDA out of memory; retrying on CPU")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model = model.to(device)
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
        else:
            raise

    rel_loss_fn = RelativeL2Loss()
    rel_err_norm = rel_loss_fn(pred, y).item()

    y_phys = split.y_normalizer.decode(y.cpu())
    pred_phys = split.y_normalizer.decode(pred.cpu())

    coeff_norm = x[0, ..., 0].detach().cpu()
    coeff_phys = split.x_normalizer.decode(coeff_norm.unsqueeze(-1)).squeeze(-1)

    target_field = y_phys[0, ..., 0]
    pred_field = pred_phys[0, ..., 0]
    err_field = pred_field - target_field

    log_info(f"Relative L2 error in normalized space: {rel_err_norm:.6e}")

    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(1, 4, 1)
    im1 = ax1.imshow(coeff_phys.numpy(), origin="lower")
    ax1.set_title("Coefficient")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 4, 2)
    im2 = ax2.imshow(target_field.numpy(), origin="lower")
    ax2.set_title("Ground Truth Solution")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.imshow(pred_field.numpy(), origin="lower")
    ax3.set_title("Predicted Solution")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(1, 4, 4)
    im4 = ax4.imshow(err_field.numpy(), origin="lower")
    ax4.set_title("Prediction Error")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches="tight")
    log_info(f"Evaluation plot saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Light Neural Operator model")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of validation sample to evaluate",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="evaluation_results.png",
        help="Path to save evaluation plot",
    )
    args = parser.parse_args()
    main(args)
