from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def resolve_mat_path(path_str: str) -> str:
    """Resolve dataset .mat path from cwd, src, repo root, or bundled dataset dirs."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(f"MAT file not found: {path}")
        return str(path.resolve())

    src_root = Path(__file__).resolve().parent
    repo_root = src_root.parent
    rel = Path(path_str)
    tried: list[str] = []

    for root in (Path.cwd(), src_root, repo_root):
        candidate = (root / rel).resolve()
        tried.append(str(candidate))
        if candidate.is_file():
            return str(candidate)

    bundled_dirs = [
        src_root / "datasets" / "Darcy2D",
        src_root / "datasets" / "Burgers1D",
        src_root / "datasets",
    ]
    for base in bundled_dirs:
        candidate = (base / rel.name).resolve()
        tried.append(str(candidate))
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"MAT file not found: {path_str!r}\nTried:\n  " + "\n  ".join(tried)
    )


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """
    Track the running average of a scalar quantity.

    Useful for logging training / validation losses over an epoch.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return 0.0 if self.count == 0 else self.sum / self.count


class RelativeL2Loss:
    """
    Relative L2 error over a batch:

        ||pred - target||_2 / ||target||_2

    Input tensors are expected to have shape [B, H, W, C].
    """

    def __init__(self, eps: float = 1e-12, reduction: str = "mean") -> None:
        self.eps = eps
        self.reduction = reduction

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred.shape={pred.shape}, target.shape={target.shape}"
            )

        diff = pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1)
        tgt = target.reshape(target.shape[0], -1)

        diff_norm = torch.norm(diff, p=2, dim=1)
        tgt_norm = torch.norm(tgt, p=2, dim=1)
        rel = diff_norm / (tgt_norm + self.eps)

        if self.reduction == "mean":
            return rel.mean()
        if self.reduction == "sum":
            return rel.sum()
        return rel
