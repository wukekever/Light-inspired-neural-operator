import os
import time
import json
import argparse
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import setup_logging
from datasets.darcy2d import Darcy2DDataset, build_darcy2d_splits
from modules.model import LightNeuralOperator2D
from utils import set_seed, RelativeL2Loss


class Monitor:
    """
    Track the current value, cumulative sum, sample count, and running average.
    """

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
    parser = argparse.ArgumentParser(description="Light-inspired Neural Operator")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./datasets/Darcy2D/piececonst_r241_N1024_smooth1.mat",
    )
    parser.add_argument("--output-dir", type=str, default="../outputs")
    parser.add_argument("--tag", type=str, default="version_1")

    parser.add_argument("--target-size", type=int, nargs=2, default=[85, 85]) # downsample the data into 85 × 85 resolution 
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=224)
    parser.add_argument("--no-coord", action="store_true")

    parser.add_argument("--batch-size", type=int, default=4) # batch size = 4 (Transolver)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--num-features", type=int, default=128) # num of features = 128 (Transolver) 
    parser.add_argument("--depth", type=int, default=8) # depth = 8 (Transolver) 

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5
    )  # L2 regularization strength
    parser.add_argument(
        "--grad-clip", type=float, default=1.0
    )  # max norm for gradient clipping, set to None to disable

    parser.add_argument(
        "--scheduler-step-size", type=int, default=5
    )  # decay learning rate every N "epochs" (epochs // scheduler_step_size = 100: 0.96^100 = 0.0176)
    parser.add_argument(
        "--scheduler-gamma", type=float, default=0.96
    )  # decay learning rate by multiplying with this factor

    parser.add_argument(
        "--log-interval", type=int, default=50
    )  # print logs every N steps
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_run_dir(args) -> str:
    """
    Build the experiment directory:
        output_dir/YYYY-MM-DD_HH-MM-SS_light_neural_operator[_tag]
    """
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
    split = build_darcy2d_splits(
        mat_path=args.data_path,
        target_size=tuple(args.target_size) if args.target_size is not None else None,
        n_train=args.n_train,
        n_val=args.n_val,
        use_coord=not args.no_coord,
        normalize_x=True,
        normalize_y=True,
        seed=args.seed,
    )

    train_set = Darcy2DDataset(split.train_x, split.train_y)
    val_set = Darcy2DDataset(split.val_x, split.val_y)

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
        },
        save_path,
    )


def validate(model, loader, mse_loss_fn, rel_loss_fn, device):
    model.eval()

    loss_meter = Monitor()
    mse_meter = Monitor()
    rel_meter = Monitor()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)

            mse_loss = mse_loss_fn(pred, y)
            rel_loss = rel_loss_fn(pred, y)
            loss = mse_loss + 0.1 * rel_loss

            batch_size = x.shape[0]
            loss_meter.update(loss.item(), batch_size)
            mse_meter.update(mse_loss.item(), batch_size)
            rel_meter.update(rel_loss.item(), batch_size)

    return {
        "loss": loss_meter.avg,
        "mse": mse_meter.avg,
        "rel_l2": rel_meter.avg,
    }


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
    logger.info(f"Starting training process from {run_dir}")

    split, train_loader, val_loader = build_loaders(args)

    model = LightNeuralOperator2D(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        num_features=args.num_features,
        depth=args.depth,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
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

        for step, (x, y) in enumerate(train_loader):
            iter_start = time.time()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)

            mse_loss = mse_loss_fn(pred, y)
            rel_loss = rel_loss_fn(pred, y)
            loss = 1.0 * mse_loss + 1.0 * rel_loss

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

                if torch.cuda.is_available() and device.type == "cuda":
                    memory_used = torch.cuda.max_memory_allocated(device=device) / (
                        1024.0**2
                    )
                else:
                    memory_used = 0.0

                logger.info(
                    f"Step [{global_step:5d}/{total_steps:5d}] - "
                    f"Loss: {loss_meter.avg:.4e} "
                    f"Error: {error_meter.avg:.4e} "
                    f"Memory: {memory_used:.2e}MB "
                    f"ETA: {str(datetime.timedelta(seconds=int(eta)))}"
                )

        scheduler.step()

        val_metrics = validate(
            model=model,
            loader=val_loader,
            mse_loss_fn=mse_loss_fn,
            rel_loss_fn=rel_loss_fn,
            device=device,
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch + 1:4d}/{args.epochs:4d}] - "
            f"TrainLoss: {loss_meter.avg:.4e} "
            f"TrainError: {error_meter.avg:.4e} "
            f"ValLoss: {val_metrics['loss']:.4e} "
            f"ValError: {val_metrics['rel_l2']:.4e} "
            f"LR: {current_lr:.4e} "
            f"EpochTime: {str(datetime.timedelta(seconds=int(epoch_time)))}"
        )

        ckpt_dir = os.path.join(run_dir, "ckpts")
        last_ckpt_path = os.path.join(ckpt_dir, "last_model.pt")
        save_checkpoint(
            save_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_val_rel=best_val_rel,
            split=split,
            args=args,
        )

        if val_metrics["rel_l2"] < best_val_rel:
            best_val_rel = val_metrics["rel_l2"]
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_val_rel=best_val_rel,
                split=split,
                args=args,
            )
            logger.info(f"Best checkpoint updated: {best_ckpt_path}")

    total_time = time.time() - training_start
    logger.info(
        f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}"
    )


if __name__ == "__main__":
    main()
