<!-- # Let There Be Light: Reflection, Refraction and Scattering for Neural Operators of Parametric PDEs -->
![](assets/rainbow_title.png)
<p align="center">
  <img src="assets/ustc-logo.png" alt="USTC Logo" height="50">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/transparent-ustc-logo.png" alt="UIUC Logo" height="50"><br>
  <sub><b>¹ University of Science and Technology of China (USTC)</b> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; <b>² University of Science and Technology of China (USTC)</b></sub>
<p>

## Title: Let There Be Light: Reflection, Refraction and Scattering for PDE Neural Operators

### Authors: Keke Wu and XXX 
[![ArXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2111.02541) 


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
.
├── doc
│   ├── light_inspired_neural_operator.pdf
│   ├── method.tex
│   └── rainbow_title.png
├── make_title.py
├── README.md
├── requirements.txt
├── scripts
│   ├── download_burgers1d.sh
│   ├── download_darcy2d.sh
│   ├── run_eval_burgers1d.sh
│   ├── run_eval_darcy2d.sh
│   ├── run_train_burgers1d.sh
│   └── run_train_darcy2d.sh
└── src
    ├── datasets
    │   ├── Burgers1D
    │   │   └── burgers_data_R10.mat
    │   ├── burgers1d.py
    │   ├── common.py
    │   ├── Darcy2D
    │   │   ├── piececonst_r241_N1024_smooth1.mat
    │   │   └── piececonst_r241_N1024_smooth2.mat
    │   ├── darcy2d.py
    │   ├── data_infos.md
    │   ├── datatest.ipynb
    │   └── __init__.py
    ├── eval.py
    ├── logger.py
    ├── modules
    │   └── model.py
    ├── run_burgers1d.py
    ├── run_darcy2d.py
    └── utils.py

```

## 📌 Citation

If you find this work useful, please cite:

```bibtex
@article{wu2026light,
    title={Let There Be Light: Reflection, Refraction and Scattering for PDE Neural Operators},
    author={Keke Wu},
    year={2026}
}
```

