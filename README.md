# 🌟 Light-inspired Neural Operator

A PDE Neural Operator Based on Light Transport Simulation

## 📋 Overview

This repository presents a novel neural operator framework that leverages principles from light transport and physics-informed learning to solve partial differential equations (PDEs) efficiently.

The framework learns a infinite-dimensional operator $\mathcal{G}_\theta$ that maps input functions to solution functions:

$$\mathcal{G}_\theta: \mathcal{A} \rightarrow \mathcal{U}$$

where $\mathcal{A}$ and $\mathcal{U}$ are function spaces.

## ✨ Features

- **🔬 Physics-inspired Architecture**: Incorporates light transport principles into neural network design
- **⚡ Efficient PDE Solving**: Fast inference for parameterized PDEs via operator learning
- **🎯 Generalization**: Trained on one problem instance, applicable to various parameters and domains


## 📦 Installation

```bash
git clone https://github.com/wukekever/Light-inspired-neural-operator.git
cd Light-inspired-neural-operator
pip install -r requirements.txt
```

## Project Architecture
```bash
tree
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{wu2026light,
    title={Light-inspired Neural Operator},
    author={Keke Wu},
    year={2026}
}
```

