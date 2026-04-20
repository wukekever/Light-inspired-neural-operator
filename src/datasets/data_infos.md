# Dataset file specifications

## 1. Burgers1D — `Burgers1D/burgers_data_R10.mat`

### 1.1 问题与几何

- **维数**：一维空间问题；每个样本是在一条一维网格上的函数。
- **空间分辨率**：8192 个网格点（与变量 `a`、`u` 的最后一维一致）。
- **样本数**：2048。

### 1.2 文件中的数组

| 键名 | `shape` | `dtype` | 含义（实现时的常见用法） |
|------|---------|---------|---------------------------|
| `a` | `(2048, 8192)` | `float64` | 输入系数 / 初值相关场（沿 \(x\)） |
| `a_smooth` | `(2048, 8192)` | `float64` | `a` 的光滑版本 |
| `a_x` | `(2048, 8191)` | `float64` | `a` 对空间的一阶差分（比 `a` 少 1 个点，与差分模板一致） |
| `a_smooth_x` | `(2048, 8191)` | `float64` | `a_smooth` 的空间一阶差分 |
| `u` | `(2048, 8192)` | `float64` | 与 `a` 同网格的解场（常作为监督目标） |

**索引**：`mat_data["a"][i]` 与 `mat_data["u"][i]` 为长度 8192 的 1D `numpy` 向量；可视化时用 `np.linspace(0, 1, 8192)` 作为 \([0,1]\) 上坐标是常见做法。

### 1.3 与 `darcy2d.py` 的对应关系（设计提示）

- Darcy 是 **`[N, H, W]` + channel**；Burgers 可视为 **`[N, L]`** 或 **`[N, L, 1]`**（\(L=8192\)）。
- 若要在输入中拼接坐标，可用 **`[N, L, 1]`** 的系数与 **`[N, L, 1]`** 的 \(x\in[0,1]\) 在最后一维 `cat`。
- **下采样**：一维可用 `F.interpolate` 对 `(N,1,L)` 做 `mode="linear"`，或先 `unsqueeze` 再插值，与 `resize_field_batch` 的 2D 版本对称实现。
- **归一化**：可对 `a`（或 `a_smooth`）与 `u` 分别做全局或按通道的 `UnitGaussianNormalizer`；注意 `dtype` 为 `float64`，进 `torch` 时常转为 `float32`。

### 1.4 文件名提示

- `R10` 与数据生成设置（如粘性 / 分辨率档）相关；实现时把具体 PDE 参数写进 docstring 或配置即可，不必从文件名解析，除非你有统一约定。

---

## 2. Darcy2D — `Darcy2D/piececonst_r241_N1024_smooth1.mat`

### 2.1 问题与几何

- **维数**：二维平面上的系数场 → 解场（椭圆型 Darcy 型算子常用设置）。
- **空间分辨率**：\(241\times 241\)（文件名中 `r241`）。
- **样本数**：1024（文件名中 `N1024`）。

### 2.2 文件中的数组

| 键名 | `shape` | `dtype` | 含义 |
|------|---------|---------|------|
| `Kcoeff` | `(1024, 241, 241)` | `float64` | 光滑系数场（部分工作流作输入） |
| `Kcoeff_x` | `(1024, 241, 241)` | `float64` | `Kcoeff` 对 \(x\) 的导数/梯度分量 |
| `Kcoeff_y` | `(1024, 241, 241)` | `float64` | `Kcoeff` 对 \(y\) 的导数/梯度分量 |
| `coeff` | `(1024, 241, 241)` | `float32` | **分段常数系数**（`darcy2d.py` 中默认输入） |
| `sol` | `(1024, 241, 241)` | `float32` | **解场**（`darcy2d.py` 中默认目标） |

**与现有代码一致**：`darcy2d.build_darcy2d_splits` 使用 `coeff` → `sol`，可选双线性缩放到 `(64,64)`，并在最后一维加单通道后可选拼接 \((x,y)\) 坐标网格。

### 2.3 实现清单（对齐 `darcy2d.py`）

- 加载：`load_mat_file` 返回字典后取 `coeff`、`sol`。
- 张量布局：`(N, H, W)` → `unsqueeze(-1)` 得 `(N, H, W, 1)`。
- 坐标：`make_coord_grid(H, W)` 得到 `[H, W, 2]`，按 batch 扩展后与系数在最后一维拼接。
- 划分：`torch.randperm` + `n_train` / `n_val`；注意 `n_train + n_val` 不超过 `N`。

---

## 3. NavierStokes2D — `NavierStokes2D/NavierStokes_V1e-5_N1200_T20.mat`

### 3.1 问题与几何

- **维数**：二维空间 + **时间序列**（多帧）。
- **空间分辨率**：\(64\times 64\)。
- **时间步数**：20（键 `t` 的长度）。
- **样本数**：1200。

### 3.2 文件中的数组

| 键名 | `shape` | `dtype` | 含义 |
|------|---------|---------|------|
| `a` | `(1200, 64, 64)` | `float32` | 通常作为 **\(t=0\)** 的输入（如初始涡量场）；与 notebook 中可视化一致 |
| `u` | `(1200, 64, 64, 20)` | `float32` | 在 20 个时间层上的场；最后一维为时间下标 |
| `t` | `(20,)` | `float32` | 时间坐标（notebook 注释：`[1, 2, …, 20]` 一类离散时间索引/时刻） |

**索引**：

- `u[i]` 形状 `(64, 64, 20)`：对固定样本 \(i\)，可取单帧 `u[i, :, :, k]` 作 \((64,64)\) 的 2D 场。
- 若预测「下一时刻」或整段轨迹，可将任务定义为：以 `a`（或 `u[...,0]`）为输入，预测 `u[...,1:]` 或其中某一帧。

### 3.3 与 `darcy2d.py` 的对应关系（设计提示）

- **仅单时刻**：把 `u[..., k]` 当作 `sol`，把 `a` 当作 `coeff`，可复用与 Darcy 相同的 `(N,H,W)` + channel + 可选坐标流程。
- **时间序列**：模型输入输出需明确时间维：例如 `x: (N, H, W, C_in)`，`y: (N, H, W, T_out)` 或 `(N, T, H, W, C)`；`Dataset.__getitem__` 返回哪一帧或整条序列由任务决定。
- **归一化**：`float32` 已对齐常见训练；若对多帧一起归一化，建议在 `train` 子集上统计标量 mean/std（与 `UnitGaussianNormalizer` 一致）。

### 3.4 文件名提示

- `V1e-5`：多与粘度 \(10^{-5}\) 量级有关；`N1200`、`T20` 与样本数、时间步数一致。

---

## 4. 通用加载说明（与 notebook 一致）

- **首选**：`scipy.io.loadmat`；过滤以 `__` 开头的键。
- **若抛出 `NotImplementedError`**（部分 v7.3 HDF5 格式 `.mat`）：需用 `h5py` 按 notebook 中 `load_mat_file` 的 fallback 读取；键与维度顺序可能与 `loadmat` 不同，需在首次加载时 `print` 校验。

---

## 5. 文件布局

| 数据集 | 目录（与 notebook 一致） |
|--------|------------------------------|
| Burgers1D | `src/datasets/Burgers1D/burgers_data_R10.mat` |
| Darcy2D | `src/datasets/Darcy2D/piececonst_r241_N1024_smooth1.mat` |
| NavierStokes2D | `src/datasets/NavierStokes2D/NavierStokes_V1e-5_N1200_T20.mat` |

下载脚本示例见 `scripts/download_burgers.sh`（Burgers）；其余数据集路径请自行放置或与脚本统一。
