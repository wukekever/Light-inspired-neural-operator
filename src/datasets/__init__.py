from datasets.common import OperatorDataset, OperatorSplit, UnitGaussianNormalizer
from datasets.burgers1d import Burgers1DDataset, BurgersSplit, build_burgers1d_splits
from datasets.darcy2d import Darcy2DDataset, DarcySplit, build_darcy2d_splits
from datasets.airfoil2d import Airfoil2DDataset, Airfoil2DSplit, build_airfoil2d_splits
from datasets.navierstokes2d import (
    NavierStokes2DDataset,
    NavierStokes2DSplit,
    build_navierstokes2d_splits,
)

__all__ = [
    "OperatorDataset",
    "OperatorSplit",
    "UnitGaussianNormalizer",
    "Burgers1DDataset",
    "BurgersSplit",
    "build_burgers1d_splits",
    "Darcy2DDataset",
    "Airfoil2DDataset",
    "DarcySplit",
    "Airfoil2DSplit",
    "build_darcy2d_splits",
    "build_airfoil2d_splits",
    "NavierStokes2DDataset",
    "NavierStokes2DSplit",
    "build_navierstokes2d_splits",
]
