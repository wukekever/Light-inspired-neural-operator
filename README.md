<!-- # Let There Be Light: Reflection, Refraction and Scattering for Neural Operators of Parametric PDEs -->
![](assets/rainbow_title.png)
<!-- <p align="center">
  <img src="assets/ustc-logo.png" alt="USTC Logo" height="50">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/transparent-ustc-sz-logo.png" alt="USTCSZ Logo" height="40"><br>
  <sub><b>¹ University of Science and Technology of China (USTC) </b> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; <b>² Suzhou Institute for Advanced Research (USTC-SZ) </b></sub>
<p> -->
<p align="center">
  <img src="assets/transparent-ustc-logo.png" alt="USTC Logo" height="40">
<p>

![](assets/framework-2.png)

#### **Title:** Let There Be Light: Reflection, Refraction and Scattering for PDE Neural Operators

#### **Authors:** Keke Wu and Jingrun Chen
`TODO: Update the arXiv link and other links after the paper is published`
 
[![ArXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2111.02541) 

[**Overview**](#-overview) | [**Installation**](#-installation) | [**Datasets**](#-datasets) | [**Project Architecture**](#-project-architecture) | [**Training and Evaluation**](#-training-and-evaluation) |
[**Citation**](#-citation)

## 🔍 Overview

This repository presents a neural operator framework that leverages principles from light transport and physics-informed learning to solve parametric partial differential equations (PDEs) efficiently.
The framework learns an infinite-dimensional operator $\mathcal{G}_\theta$ that maps input functions to solution functions:

$$\mathcal{G}_\theta: \mathcal{A} \rightarrow \mathcal{U}$$

where $\mathcal{A}$ and $\mathcal{U}$ are function spaces.


## 📦 Installation

```bash
# Clone the repository and install dependencies
git clone https://github.com/wukekever/Light-inspired-neural-operator.git
cd Light-inspired-neural-operator
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand): </summary>
  
  - Python >= 3.10
  - torch >= 2.0.0
  - numpy >= 1.24.0
  - scipy >= 1.10.0 (for loading `.mat` files)
  - h5py >= 3.8.0 (for loading v7.3 `.mat` files)
  - matplotlib >= 3.7.0 (for evaluation plots)
  - termcolor >= 2.3.0 (for colored logging)
  - Pillow >= 9.5.0 (for `make_title.py`)

</details>


## 🔗 Datasets
We provide datasets for Burgers equation, Darcy flow, and Navier-Stokes equation used in the paper. Data generation details are available in the paper.

**Download datasets:**
- [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

Alternatively, use the provided shell script in the Light-inspired-neural-operator repository to download the datasets directly:
```bash
# Download Darcy 2D dataset for example
bash ./scripts/download_darcy2d.sh
```

**Dataset Format:**

Datasets are provided as MATLAB files and each file is loaded as a tensor where the first index represents samples and remaining indices represent discretization dimensions.

Examples:
- `burgers_data_R10.mat`: Shape [1000, 8192] — 1000 samples on a 1D grid of 8192 points
- `Darcy2D_piececonst_r241_N1024_smooth1.mat`: Shape [1024, 241, 241] — 1024 samples on a 2D grid of 241×241
- `ns_V1e-3_N5000_T50.mat`: Shape [5000, 64, 64, 50] — 5000 samples on a 2D grid of 64×64 with 50 time steps


## 📁 Project Architecture

```bash
tree
.
├── README.md
├── assets
│   ├── framework-2.png
│   ├── rainbow_title.png
│   ├── transparent-ustc-logo.png
│   ├── transparent-ustc-sz-logo.png
│   └── ustc-logo.png
├── doc
│   └── main.tex
├── make_title.py
├── requirements.txt
├── scripts
│   ├── download_burgers1d.sh
│   ├── download_darcy2d.sh
│   ├── download_navierstokes2d.sh
│   ├── run_eval_burgers1d.sh
│   ├── run_eval_darcy2d.sh
│   ├── run_eval_navierstokes2d.sh
│   ├── run_train_burgers1d.sh
│   ├── run_train_darcy2d.sh
│   └── run_train_navierstokes2d.sh
└── src
    ├── datasets
    │   ├── __init__.py
    │   ├── burgers1d.py
    │   ├── common.py
    │   ├── darcy2d.py
    │   ├── data_infos.md
    │   ├── datatest.ipynb
    │   └── navierstokes2d.py
    ├── eval.py
    ├── logger.py
    ├── modules
    │   └── model.py
    ├── run_burgers1d.py
    ├── run_darcy2d.py
    ├── run_navierstokes2d.py
    └── utils.py
```

## 🚀 Training and Evaluation
For training and evaluation, we provide separate shell scripts for each dataset. You can run them as follows:
```bash
# Train on Darcy2D Problem
bash ./scripts/run_train_darcy2d.sh
# Evaluate on Darcy2D Problem after training
bash ./scripts/run_eval_darcy2d.sh
```

## 📌 Citation

If you find this work useful, please cite:

`TODO: Update the citation after the information is available`
```bibtex
@article{wu2026light,
    title={Let There Be Light: Reflection, Refraction and Scattering for PDE Neural Operators},
    author={Keke Wu, Jingrun Chen},
    year={2026},
    eprint={},
    archivePrefix={arXiv},
    url={},
}
```

