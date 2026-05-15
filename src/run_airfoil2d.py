import os
import time
import json
import argparse
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import setup_logging
from datasets.airfoil2d import Airfoil2DDataset, build_airfoil2d_splits
from modules.model import LightNeuralOperator2D
from utils import set_seed, RelativeL2Loss


class Monitor:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train LiNO on the Geo-FNO NACA airfoil benchmark"
    )
    parser.add_argument("--dataset_name", type=str, default="airfoil2d")
    parser.add_argument("--data-path", type=str, default="./datasets/NACA")
    parser.add_argument("--output-dir", type=str, default="../outputs")
    parser.add_argument("--tag", type=str, default="airfoil2d_lino")

    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        help="Optional resize, e.g. --target-size 111 51. Keep None for native 221x51.",
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument(
        "--target-channel",
        type=int,
        default=4,
        help="Geo-FNO uses Q[:, 4] as the Mach-number target.",
    )
    parser.add_argument(
        "--shuffle-split",
        action="store_true",
        help="Shuffle samples before splitting. Disabled by default to match Geo-FNO.",
    )
    parser.add_argument(
        "--no-coord",
        action="store_true",
        help="Use only physical mesh coordinates x,y. Default adds computational coordinates xi,eta.",
    )
    parser.add_argument("--no-normalize-x", action="store_true")
    parser.add_argument("--no-normalize-y", action="store_true")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="Defaults to 4 with xi,eta and 2 with --no-coord.",
    )
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--num-features", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument(
        "--scattering-type",
        type=str,
        default="efficient",
        choices=["efficient", "standard"],
    )

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-step-size", type=int, default=400)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--rel-weight", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_run_dir(args) -> str:
    run_name = datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S_light_neural_operator"
    )
    if args.tag is not None:
        run_name += f"_{args.tag}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def build_loaders(args):
    split = build_airfoil2d_splits(
        data_path=args.data_path,
        target_size=tuple(args.target_size) if args.target_size is not None else None,
        n_train=args.n_train,
        n_val=args.n_val,
        use_coord=not args.no_coord,
        normalize_x=not args.no_normalize_x,
        normalize_y=not args.no_normalize_y,
        seed=args.seed,
        target_channel=args.target_channel,
        shuffle=args.shuffle_split,
    )

    train_set = Airfoil2DDataset(split.train_x, split.train_y)
    val_set = Airfoil2DDataset(split.val_x, split.val_y)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return split, train_loader, val_loader


def save_checkpoint(
    save_path, model, optimizer, scheduler, epoch, best_val_rel, split, args
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "best_val_rel": best_val_rel,
            "args": vars(args),
            "x_normalizer": split.x_normalizer.state_dict(),
            "y_normalizer": split.y_normalizer.state_dict(),
            "spatial_dims": split.spatial_dims,
            "in_channels": split.input_channels,
            "out_channels": split.output_channels,
        },
        save_path,
    )


@torch.no_grad()
def validate(
    model,
    loader,
    mse_loss_fn,
    rel_loss_fn,
    device,
    mse_weight: float,
    rel_weight: float,
):
    model.eval()
    loss_meter = Monitor()
    mse_meter = Monitor()
    rel_meter = Monitor()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        mse_loss = mse_loss_fn(pred, y)
        rel_loss = rel_loss_fn(pred, y)
        loss = mse_weight * mse_loss + rel_weight * rel_loss

        batch_size = x.shape[0]
        loss_meter.update(loss.item(), batch_size)
        mse_meter.update(mse_loss.item(), batch_size)
        rel_meter.update(rel_loss.item(), batch_size)

    return {"loss": loss_meter.avg, "mse": mse_meter.avg, "rel_l2": rel_meter.avg}


def main():
    args = get_parser()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = build_run_dir(args)
    logger = setup_logging(os.path.join(run_dir, "logs"), logger_name="wukekever")

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f"Configuration saved to {config_path}")
    logger.info(f"Starting Airfoil2D training from {run_dir}")

    split, train_loader, val_loader = build_loaders(args)
    in_channels = (
        args.in_channels if args.in_channels is not None else split.input_channels
    )
    if in_channels != split.input_channels:
        logger.warning(
            f"--in-channels={in_channels}, but dataset provides {split.input_channels} channels."
        )

    logger.info(
        f"Dataset tensors: train_x={tuple(split.train_x.shape)}, train_y={tuple(split.train_y.shape)}, "
        f"val_x={tuple(split.val_x.shape)}, val_y={tuple(split.val_y.shape)}"
    )

    model = LightNeuralOperator2D(
        in_channels=in_channels,
        out_channels=args.out_channels,
        num_features=args.num_features,
        depth=args.depth,
        scattering_type=args.scattering_type,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
    )
    mse_loss_fn = nn.MSELoss()
    rel_loss_fn = RelativeL2Loss()

    best_val_rel = float("inf")
    global_step = 0
    total_steps = args.epochs * len(train_loader)
    training_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        loss_meter = Monitor()
        error_meter = Monitor()
        batch_time_meter = Monitor()
        epoch_start = time.time()

        for _, (x, y) in enumerate(train_loader):
            iter_start = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            mse_loss = mse_loss_fn(pred, y)
            rel_loss = rel_loss_fn(pred, y)
            loss = args.mse_weight * mse_loss + args.rel_weight * rel_loss
            loss.backward()

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            iter_time = time.time() - iter_start
            global_step += 1
            batch_size = x.shape[0]
            loss_meter.update(loss.item(), batch_size)
            error_meter.update(rel_loss.item(), batch_size)
            batch_time_meter.update(iter_time)

            if global_step % args.log_interval == 0:
                eta = batch_time_meter.avg * (total_steps - global_step)
                memory_used = 0.0
                if torch.cuda.is_available() and device.type == "cuda":
                    memory_used = torch.cuda.max_memory_allocated(device=device) / (
                        1024.0**2
                    )
                logger.info(
                    f"Step [{global_step:5d}/{total_steps:5d}] - "
                    f"Loss: {loss_meter.avg:.4e} Error: {error_meter.avg:.4e} "
                    f"Memory: {memory_used:.2e}MB ETA: {str(datetime.timedelta(seconds=int(eta)))}"
                )

        scheduler.step()
        val_metrics = validate(
            model=model,
            loader=val_loader,
            mse_loss_fn=mse_loss_fn,
            rel_loss_fn=rel_loss_fn,
            device=device,
            mse_weight=args.mse_weight,
            rel_weight=args.rel_weight,
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch + 1:4d}/{args.epochs:4d}] - "
            f"TrainLoss: {loss_meter.avg:.4e} TrainError: {error_meter.avg:.4e} "
            f"ValLoss: {val_metrics['loss']:.4e} ValError: {val_metrics['rel_l2']:.4e} "
            f"LR: {current_lr:.4e} EpochTime: {str(datetime.timedelta(seconds=int(epoch_time)))}"
        )

        ckpt_dir = os.path.join(run_dir, "ckpts")
        last_ckpt_path = os.path.join(ckpt_dir, "last_model.pt")
        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_val_rel,
            split,
            args,
        )

        if val_metrics["rel_l2"] < best_val_rel:
            best_val_rel = val_metrics["rel_l2"]
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_val_rel,
                split,
                args,
            )
            logger.info(f"Best checkpoint updated: {best_ckpt_path}")

    total_time = time.time() - training_start
    logger.info(
        f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}"
    )
    logger.info(f"Best validation relative L2: {best_val_rel:.6e}")


if __name__ == "__main__":
    main()
