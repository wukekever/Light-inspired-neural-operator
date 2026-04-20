<!-- # Let There Be Light: Reflection, Refraction and Scattering for Neural Operators of Parametric PDEs -->
![](doc/rainbow_title.png)

## 🔍 Overview

This repository presents a neural operator framework that leverages principles from light transport and physics-informed learning to solve parametric partial differential equations (PDEs) efficiently.
The framework learns a infinite-dimensional operator $\mathcal{G}_\theta$ that maps input functions to solution functions:

$$\mathcal{G}_\theta: \mathcal{A} \rightarrow \mathcal{U}$$

where $\mathcal{A}$ and $\mathcal{U}$ are function spaces.


## 📦 Installation

```bash
git clone https://github.com/wukekever/Light-inspired-neural-operator.git
cd Light-inspired-neural-operator
pip install -r requirements.txt
```

## 🔗 Datasets
We provide datasets for Burgers equation, Darcy flow, and Navier-Stokes equation used in the paper. Data generation details are available in the paper.

**Download datasets:**
- [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

Alternatively, use the provided shell script in the Light-inspired-neural-operator repository to download the datasets directly:
```bash
bash ./scripts/download_burgers.sh 
bash ./scripts/download_darcy2d.sh
```

**Dataset Format:**

Datasets are provided as MATLAB files and each file is loaded as a tensor where the first index represents samples and remaining indices represent discretization dimensions.

Examples:
- `Burgers_R10.mat`: Shape [1000, 8192] — 1000 samples on a 1D grid of 8192 points
- `Darcy2D_piececonst_r241_N1024_smooth1.mat`: Shape [1024, 241, 241] — 1024 samples on a 2D grid of 241×241
- `NavierStokes_V1e-3_N5000_T50.mat`: Shape [5000, 64, 64, 50] — 5000 samples on a 2D grid of 64×64 with 50 time steps



## 📁 Project Architecture
```bash
tree

```

## 📌 Citation

If you find this work useful, please cite:

```bibtex
@article{wu2026light,
    title={Let There Be Light: Reflection, Refraction and Scattering for Neural Operators of Parametric PDEs},
    author={Keke Wu},
    year={2026}
}
```

