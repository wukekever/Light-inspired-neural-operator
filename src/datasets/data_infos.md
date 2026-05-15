*# Dataset Specifications

## 1. Burgers1D — `Burgers1D/burgers_data_R10.mat`

### 1.1 Problem & Geometry

- **Dimensionality**: 1D spatial problem; each sample is a function on a 1D grid.
- **Spatial resolution**: 8192 grid points (consistent with the last dimension of `a` and `u`).
- **Number of samples**: 2048.

### 1.2 Arrays in File

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `a` | `(2048, 8192)` | `float64` | Input coefficient or initial condition field (along x) |
| `a_smooth` | `(2048, 8192)` | `float64` | Smoothed version of `a` |
| `a_x` | `(2048, 8191)` | `float64` | First-order spatial derivative of `a` |
| `a_smooth_x` | `(2048, 8191)` | `float64` | Spatial derivative of `a_smooth` |
| `u` | `(2048, 8192)` | `float64` | Solution field; supervision target |

**Indexing**: `mat_data["a"][i]` and `mat_data["u"][i]` are 1D vectors of length 8192. Use `np.linspace(0, 1, 8192)` for coordinates on [0,1].

### 1.3 Design Notes

- Burgers: `[N, L]` or `[N, L, 1]` format (L=8192); Darcy: `[N, H, W]` + channel.
- Concatenate coefficients `[N, L, 1]` with coordinates `[N, L, 1]` along last dimension.
- Downsampling: use `F.interpolate` with `mode="linear"` on `(N,1,L)`.
- Normalization: apply `UnitGaussianNormalizer` separately to `a` and `u`; convert `float64` to `float32` for PyTorch.

### 1.4 File Naming

- `R10` relates to generation parameters (viscosity, resolution); document PDE settings in docstrings.

---

## 2. Darcy2D — `Darcy2D/piececonst_r241_N1024_smooth1.mat`

### 2.1 Problem & Geometry

- **Dimensionality**: 2D coefficient → solution (elliptic operator).
- **Spatial resolution**: 241 × 241.
- **Number of samples**: 1024.

### 2.2 Arrays in File

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `Kcoeff` | `(1024, 241, 241)` | `float64` | Smooth coefficient field |
| `Kcoeff_x` | `(1024, 241, 241)` | `float64` | x-derivative of `Kcoeff` |
| `Kcoeff_y` | `(1024, 241, 241)` | `float64` | y-derivative of `Kcoeff` |
| `coeff` | `(1024, 241, 241)` | `float32` | **Piecewise-constant coefficient** (default input) |
| `sol` | `(1024, 241, 241)` | `float32` | **Solution field** (default target) |

**Integration with code**: `darcy2d.build_darcy2d_splits` uses `coeff` → `sol`, with optional bilinear downsampling to (64,64) and optional coordinate grid concatenation.

### 2.3 Implementation Checklist

- Load `coeff` and `sol` from dictionary.
- Layout: `(N, H, W)` → `unsqueeze(-1)` → `(N, H, W, 1)`.
- Coordinates: `make_coord_grid(H, W)` → `[H, W, 2]`; concatenate with coefficients.
- Split: use `torch.randperm` with `n_train` + `n_val` ≤ N.

---

## 3. NavierStokes2D — `NavierStokes2D/ns_V1e-3_N5000_T50.mat`

### 3.1 Problem & Geometry

- **Dimensionality**: 2D space + time (temporal sequence).
- **Spatial resolution**: 64 × 64.
- **Time steps**: 50.
- **Number of samples**: 5000.

### 3.2 Arrays in File

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `a` | `(5000, 64, 64)` | `float32` | Initial condition (e.g., vorticity at t=0) |
| `u` | `(5000, 64, 64, 50)` | `float32` | Field across 50 time steps (last dim: time index) |
| `t` | `(50, 1)` | `float32` | Time coordinates |

**Indexing**:
- `u[i]` has shape `(64, 64, 50)`: extract frame k as `u[i, :, :, k]`.
- Use `a` (or `u[..., 0]`) as input; predict `u[..., 1:]` or specific frames.

*