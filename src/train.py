from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.darcy2d import Darcy2DDataset, build_darcy2d_splits
from modules.model import LightNeuralOperator2D
from utils import AverageMeter, RelativeL2Loss, resolve_mat_path, set_seed


@dataclass
class TrainConfig:
    """
    Configuration for Darcy2D training.
    """

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_path: str = "datasets/Darcy2D/piececonst_r241_N1024_smooth1.mat"
    target_size: tuple[int, int] | None = (64, 64)
    n_train: int = 800
    n_val: int = 224
    use_coord: bool = True
    normalize_x: bool = True
    normalize_y: bool = True

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    in_channels: int = 3  # coeff + x + y
    out_channels: int = 1  # sol
    num_features: int = 16
    depth: int = 2

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float | None = 1.0

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    seed: int = 42
    save_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq: int = 1


def build_loaders(cfg: TrainConfig):
    """
    Build train/validation datasets and dataloaders.

    The underlying dataset builder returns tensors in [N, H, W, C] layout,
    which matches the current model implementation directly.
    """
    split = build_darcy2d_splits(
        mat_path=resolve_mat_path(cfg.data_path),
        target_size=cfg.target_size,
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        use_coord=cfg.use_coord,
        normalize_x=cfg.normalize_x,
        normalize_y=cfg.normalize_y,
        seed=cfg.seed,
    )

    train_set = Darcy2DDataset(split.train_x, split.train_y)
    val_set = Darcy2DDataset(split.val_x, split.val_y)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    return split, train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mse_loss_fn: nn.Module,
    rel_loss_fn: RelativeL2Loss,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """
    Run one training epoch.

    The model predicts normalized targets if y-normalization is enabled in the
    dataset builder. Therefore, the loss is computed in normalized output space.
    """
    model.train()

    mse_meter = AverageMeter()
    rel_meter = AverageMeter()
    total_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x)

        mse_loss = mse_loss_fn(pred, y)
        rel_loss = rel_loss_fn(pred, y)

        # A simple composite objective:
        #   - MSE stabilizes pointwise regression,
        #   - relative L2 explicitly targets operator-level field accuracy.
        loss = mse_loss + 0.1 * rel_loss

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = x.shape[0]
        mse_meter.update(mse_loss.item(), batch_size)
        rel_meter.update(rel_loss.item(), batch_size)
        total_meter.update(loss.item(), batch_size)

    return {
        "loss": total_meter.avg,
        "mse": mse_meter.avg,
        "rel_l2": rel_meter.avg,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    mse_loss_fn: nn.Module,
    rel_loss_fn: RelativeL2Loss,
    device: torch.device,
) -> dict[str, float]:
    """
    Run one validation epoch.
    """
    model.eval()

    mse_meter = AverageMeter()
    rel_meter = AverageMeter()
    total_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)

        mse_loss = mse_loss_fn(pred, y)
        rel_loss = rel_loss_fn(pred, y)
        loss = mse_loss + 0.1 * rel_loss

        batch_size = x.shape[0]
        mse_meter.update(mse_loss.item(), batch_size)
        rel_meter.update(rel_loss.item(), batch_size)
        total_meter.update(loss.item(), batch_size)

    return {
        "loss": total_meter.avg,
        "mse": mse_meter.avg,
        "rel_l2": rel_meter.avg,
    }


def save_checkpoint(
    save_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_rel: float,
    split,
    cfg: TrainConfig,
) -> None:
    """
    Save training state.

    The normalizers are stored together with the model so that inference can
    consistently reproduce the same input/output scaling.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None
            if scheduler is None
            else scheduler.state_dict(),
            "best_val_rel": best_val_rel,
            "config": cfg.__dict__,
            "x_normalizer": split.x_normalizer.state_dict(),
            "y_normalizer": split.y_normalizer.state_dict(),
        },
        save_path,
    )


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    split, train_loader, val_loader = build_loaders(cfg)

    model = LightNeuralOperator2D(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_features=cfg.num_features,
        depth=cfg.depth,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step_size,
        gamma=cfg.scheduler_gamma,
    )

    mse_loss_fn = nn.MSELoss()
    rel_loss_fn = RelativeL2Loss()

    best_val_rel = float("inf")
    best_ckpt_path = os.path.join(cfg.save_dir, "lightno_darcy2d_best.pt")

    print("=" * 80)
    print("Start training")
    print(f"Device           : {device}")
    print(f"Data path        : {cfg.data_path}")
    print(f"Target size      : {cfg.target_size}")
    print(f"Train samples    : {cfg.n_train}")
    print(f"Val samples      : {cfg.n_val}")
    print(f"Batch size       : {cfg.batch_size}")
    print(f"Model features   : {cfg.num_features}")
    print(f"Model depth      : {cfg.depth}")
    print("=" * 80)

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mse_loss_fn=mse_loss_fn,
            rel_loss_fn=rel_loss_fn,
            device=device,
            grad_clip=cfg.grad_clip,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            mse_loss_fn=mse_loss_fn,
            rel_loss_fn=rel_loss_fn,
            device=device,
        )

        scheduler.step()

        if val_metrics["rel_l2"] < best_val_rel:
            best_val_rel = val_metrics["rel_l2"]
            save_checkpoint(
                save_path=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_rel=best_val_rel,
                split=split,
                cfg=cfg,
            )

        if epoch % cfg.print_freq == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
                f"lr={current_lr:.2e} | "
                f"train_loss={train_metrics['loss']:.6e} | "
                f"train_mse={train_metrics['mse']:.6e} | "
                f"train_rel={train_metrics['rel_l2']:.6e} | "
                f"val_loss={val_metrics['loss']:.6e} | "
                f"val_mse={val_metrics['mse']:.6e} | "
                f"val_rel={val_metrics['rel_l2']:.6e}"
            )

    print("=" * 80)
    print(f"Training finished. Best validation relative L2 = {best_val_rel:.6e}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
