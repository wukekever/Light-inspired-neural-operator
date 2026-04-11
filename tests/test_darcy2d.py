from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

from datasets.darcy2d import (
    Darcy2DDataset,
    UnitGaussianNormalizer,
    build_darcy2d_splits,
    load_mat_file,
    make_coord_grid,
    resize_field_batch,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
PIECECONST_MAT = (
    _REPO_ROOT / "src/datasets/Darcy2D/piececonst_r241_N1024_smooth1.mat"
)


def test_load_mat_file_filters_private_keys(tmp_path: Path) -> None:
    path = tmp_path / "sample.mat"
    scipy.io.savemat(
        path,
        {"coeff": np.zeros((2, 4, 4), dtype=np.float32)},
        format="5",
    )
    data = load_mat_file(str(path))
    assert set(data.keys()) == {"coeff"}
    assert data["coeff"].shape == (2, 4, 4)


def test_resize_field_batch_shape() -> None:
    x = torch.randn(3, 8, 12)
    y = resize_field_batch(x, (5, 7))
    assert y.shape == (3, 5, 7)


def test_make_coord_grid_bounds_and_shape() -> None:
    h, w = 5, 7
    grid = make_coord_grid(h, w)
    assert grid.shape == (h, w, 2)
    assert grid.dtype == torch.float32
    assert grid[0, 0, 0].item() == 0.0 and grid[0, 0, 1].item() == 0.0
    assert grid[-1, -1, 0].item() == 1.0 and grid[-1, -1, 1].item() == 1.0


def test_unit_gaussian_normalizer_roundtrip() -> None:
    ref = torch.randn(40, 8, 8, 1)
    n = UnitGaussianNormalizer(ref)
    x = torch.randn(10, 8, 8, 1)
    decoded = n.decode(n.encode(x))
    torch.testing.assert_close(decoded, x)


def test_unit_gaussian_normalizer_state_dict() -> None:
    ref = torch.ones(4, 2, 2, 1)
    n = UnitGaussianNormalizer(ref, eps=1e-4)
    m = UnitGaussianNormalizer(torch.zeros(1))
    m.load_state_dict(n.state_dict())
    assert m.eps == n.eps
    torch.testing.assert_close(m.mean, n.mean)
    torch.testing.assert_close(m.std, n.std)


def test_darcy2d_dataset_len_and_getitem() -> None:
    x = torch.randn(5, 3, 3, 2)
    y = torch.randn(5, 3, 3, 1)
    ds = Darcy2DDataset(x, y)
    assert len(ds) == 5
    xi, yi = ds[2]
    assert xi.shape == (3, 3, 2) and yi.shape == (3, 3, 1)


def test_darcy2d_dataset_invalid_rank() -> None:
    with pytest.raises(ValueError, match="4D"):
        Darcy2DDataset(torch.randn(5, 3, 3), torch.randn(5, 3, 3, 1))


def _write_darcy_mat(path: Path, n: int, h: int, w: int) -> None:
    rng = np.random.default_rng(0)
    coeff = rng.standard_normal((n, h, w)).astype(np.float32)
    sol = rng.standard_normal((n, h, w)).astype(np.float32)
    scipy.io.savemat(path, {"coeff": coeff, "sol": sol}, format="5")


def test_build_darcy2d_splits_shapes_and_counts(tmp_path: Path) -> None:
    mat = tmp_path / "darcy.mat"
    n, h, w = 32, 8, 8
    _write_darcy_mat(mat, n, h, w)
    n_train, n_val = 20, 8
    split = build_darcy2d_splits(
        str(mat),
        target_size=None,
        n_train=n_train,
        n_val=n_val,
        use_coord=True,
        normalize_x=True,
        normalize_y=True,
        seed=1,
    )
    assert split.train_x.shape == (n_train, h, w, 3)
    assert split.train_y.shape == (n_train, h, w, 1)
    assert split.val_x.shape == (n_val, h, w, 3)
    assert split.val_y.shape == (n_val, h, w, 1)


def test_build_darcy2d_splits_no_coords(tmp_path: Path) -> None:
    mat = tmp_path / "darcy.mat"
    _write_darcy_mat(mat, 16, 4, 4)
    split = build_darcy2d_splits(
        str(mat),
        target_size=None,
        n_train=10,
        n_val=4,
        use_coord=False,
        seed=0,
    )
    assert split.train_x.shape == (10, 4, 4, 1)
    assert split.val_x.shape == (4, 4, 4, 1)


def test_build_darcy2d_splits_resize(tmp_path: Path) -> None:
    mat = tmp_path / "darcy.mat"
    _write_darcy_mat(mat, 12, 16, 16)
    split = build_darcy2d_splits(
        str(mat),
        target_size=(5, 6),
        n_train=6,
        n_val=4,
        seed=0,
    )
    assert split.train_x.shape == (6, 5, 6, 3)
    assert split.val_y.shape == (4, 5, 6, 1)


def test_build_darcy2d_splits_split_too_large(tmp_path: Path) -> None:
    mat = tmp_path / "darcy.mat"
    _write_darcy_mat(mat, 5, 4, 4)
    with pytest.raises(ValueError, match="exceeds"):
        build_darcy2d_splits(str(mat), target_size=None, n_train=4, n_val=4, seed=0)


def test_build_darcy2d_splits_reproducible_shuffle(tmp_path: Path) -> None:
    mat = tmp_path / "darcy.mat"
    _write_darcy_mat(mat, 24, 4, 4)
    a = build_darcy2d_splits(str(mat), target_size=None, n_train=8, n_val=4, seed=99)
    b = build_darcy2d_splits(str(mat), target_size=None, n_train=8, n_val=4, seed=99)
    torch.testing.assert_close(a.train_x, b.train_x)
    torch.testing.assert_close(a.val_y, b.val_y)


def test_build_darcy2d_splits_different_seed_differs(tmp_path: Path) -> None:
    mat = tmp_path / "darcy.mat"
    _write_darcy_mat(mat, 24, 4, 4)
    a = build_darcy2d_splits(str(mat), target_size=None, n_train=8, n_val=4, seed=1)
    b = build_darcy2d_splits(str(mat), target_size=None, n_train=8, n_val=4, seed=2)
    assert not torch.allclose(a.train_x, b.train_x)


def test_build_darcy2d_splits_normalize_y_changes_targets(tmp_path: Path) -> None:
    mat = tmp_path / "darcy.mat"
    _write_darcy_mat(mat, 20, 4, 4)
    on = build_darcy2d_splits(
        str(mat),
        target_size=None,
        n_train=12,
        n_val=4,
        normalize_y=True,
        seed=0,
    )
    off = build_darcy2d_splits(
        str(mat),
        target_size=None,
        n_train=12,
        n_val=4,
        normalize_y=False,
        seed=0,
    )
    assert not torch.allclose(on.train_y, off.train_y)


@pytest.mark.skipif(not PIECECONST_MAT.is_file(), reason="bundle MAT not in repo")
def test_piececonst_mat_raw_shapes() -> None:
    data = load_mat_file(str(PIECECONST_MAT))
    assert "coeff" in data and "sol" in data
    coeff, sol = data["coeff"], data["sol"]
    assert coeff.shape == sol.shape == (1024, 241, 241)


@pytest.mark.skipif(not PIECECONST_MAT.is_file(), reason="bundle MAT not in repo")
def test_piececonst_build_splits_default_resize() -> None:
    split = build_darcy2d_splits(
        str(PIECECONST_MAT),
        target_size=(64, 64),
        n_train=800,
        n_val=224,
        seed=42,
    )
    assert split.train_x.shape == (800, 64, 64, 3)
    assert split.train_y.shape == (800, 64, 64, 1)
    assert split.val_x.shape == (224, 64, 64, 3)
    assert split.val_y.shape == (224, 64, 64, 1)
