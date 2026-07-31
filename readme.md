# RWSPM — Python 模拟代码 / Simulation Code

**Regional Window Selection via Partition Modeling (RWSPM)**

RWSPM 方法（区域窗口划分建模）的 Python 模拟实现。该方法是 R 版
`code_simulation/code_RWSPM` 的 Python 移植：将高维图像响应变量 `Y` 划分为多个
重叠的方形子区域（滑窗），对每个子区域做 LQD（对数分位密度）变换后，用
Ball Covariance（BCov）逐区域检验其与预测变量 `X` 的独立性，最后用 Cauchy
Combination Test（CCT）对空间上连通的显著子区域做聚类组合检验（RW-MTCCT）。

A Python implementation of RWSPM simulation — a Python port of the R project
`code_simulation/code_RWSPM`. It partitions a high-dimensional image response
`Y` into overlapping square sub-regions, applies an LQD (Log Quantile Density)
transform to each region, tests the independence between the predictor `X` and
each region with Ball Covariance (BCov), and finally combines the significant
spatially-connected regions with the Cauchy Combination Test (RW-MTCCT).

---

**语言 / Language**

- [简体中文](#简体中文)
- [繁體中文](#繁體中文)
- [English](#english)

---

## 简体中文

### 1. 项目简介

本目录包含 RWSPM 论文模拟实验的 Python 代码，用于评估 RWSPM 方法在 7 种
仿真设置下的表现（Type I error 与检验功效）。

模拟框架（与 R 版 `analysis_RWSPM_fixed.R` 对齐）：

1. **数据生成**：`gendata_xy.py` 按 Setting 1–7 生成 `(X, Y, Z)`；
2. **图像分区**：用滑窗把 `M1 × M2` 的图像划分为若干重叠方形子区域；
3. **LQD 变换**：对每个子区域估计密度并做 LQD 变换，得到区域化特征；
4. **逐区域检验**：对每个子区域计算 BCov² 统计量并得到 p 值
   （`gamma` / `limit` / `permu` 三种方法）；
5. **聚类组合**：用 RW-MTCCT 把空间连通的显著子区域合并，得到聚类级 p 值；
6. **保存结果**：逐区域 CSV + 聚类级 CSV。

### 2. 目录结构

```
Python/
├── run_simulation.py      # 串行模拟主程序（对标 analysis_RWSPM.R）
├── run_simu_parallel.py   # 多核并行模拟主程序
├── rwspm.py               # RWSPM 核心算法（分区、滑窗、CCT、RW-MTCCT）
├── bcov.py                # Ball Covariance 统计量与检验
├── gendata_xy.py          # 模拟数据生成（Setting 1–7）
├── lqd.py                 # LQD（对数分位密度）变换
├── gammafit.py            # Gamma 矩匹配（用于 method='gamma' 的 p 值）
├── seed.py                # 随机种子工具
├── lqd_mix.py             # LQD 混合模型变体（辅助）
├── permutation_null.py    # 置换零分布辅助脚本
├── example_rwspm.ipynb    # 示例 Notebook
├── cball_ext/             # C 扩展（加速 BDD 核矩阵，'limit' 方法需要）
│   ├── cball_ext.dll / .so
│   ├── compile_win.bat / compile_linux.sh
│   └── *.c / *.h
└── data/result/           # 模拟输出（见第 6 节）
    └── Setting{k}/        # k = 1..7
        ├── test_rwspm{l}.csv         # 第 l 轮重复的逐区域结果
        └── rwspm_cluster_set{k}.csv  # 聚类级组合 p 值
```

### 3. 环境依赖

- Python ≥ 3.8
- `numpy`、`scipy`、`networkx`
- （可选，仅 `limit` 方法）`ctypes` + 编译好的 `cball_ext.dll`（Windows 用
  MinGW 编译，见 `cball_ext/compile_win.bat`；Linux 用 `compile_linux.sh`）

```bash
pip install numpy scipy networkx
```

### 4. 文件功能说明

#### 4.1 `gendata_xy.py` — 数据生成

实现论文中的仿真设置：

| Setting | a | sigma_xy | model | 说明 |
|:---:|:---:|:---:|:---:|:---|
| 1 | 0 | 0 | independent | 零假设（X ⊥ Y），用于 Type I error |
| 2–4 | 1 | 1/2/3 | linear | 线性模型 |
| 5–7 | 1 | 1/2/3 | nonlinear | 非线性模型 |

- 网格：`[-50, 100] × [-50, 50]` 上 `M1 × M2` 个体素；
- 疾病区域：`A = [-10, 10] × [-10, 10]`，每个个体有随机位置/半径的受影响
  子区域 `A_i`，信号 `β(s) = 0.8·a·exp(−0.1·d)`；
- 返回：`X`（基因型剂量，n×1，取值 {0,1,2}）、`Y`（展平的图像响应，n×(M1·M2)）、
  `Z`（协变量 [age/100, gender]，n×2）。

```python
from gendata_xy import generate_data
X, Y, Z = generate_data(setting=2, n=200, M1=150, M2=100, random_state=2026)
```

#### 4.2 `bcov.py` — Ball Covariance

- `bcov(X, Y)`：计算经验 Ball Covariance 统计量 `BCov²_n(X, Y)`；
- `bcov_perm_test(x, y, n_perm=199, method='gamma', DY=None)`：BCov 检验，
  返回 `(stat, pval)`。`method`：
  - `'permu'`：传统置换 p 值 `(∑I(stat_perm ≥ stat)+1)/(n_perm+1)`；
  - `'gamma'`（默认）：对置换零分布做 Gamma 矩匹配近似（`gammafit.py`）；
  - `'limit'`：HBE 三累积量 Gamma 极限近似（需 `cball_ext` 编译库）；
- `bcov_perm_test_gwas(x, DY, ...)`：预计算 `DY` 的变体，避免重复计算；
- 内部通过 C 扩展 `cball_ext` 加速 BDD 核矩阵（找不到 DLL 时自动降级）。

#### 4.3 `rwspm.py` — RWSPM 核心算法

- `slide_width_step(b, width, M1, M2)`：滑窗分区，返回每个子区域的像素索引
  （`idx` 为 m×width² 的 0 基索引矩阵）及 `width_x` / `width_y`；
- `compute_ov` + `golden_section_search`：基于 JSD 目标函数（含惩罚项）自动
  选择最优滑窗宽度；
- `RWSPM(x, y, M1, M2, window_width=None, method='bcov')`：完整流程封装。
  > ⚠️ 注意：`RWSPM()` 中的 p 值为占位符（固定 1.0）。**实际模拟请使用**
  > `run_simulation.py` 中的 `run_rwspm_pipeline()`（或直接运行两个主程序），
  > 它会调用 `bcov_perm_test` 计算真实 p 值。
- `CCT_chisq(pvals, weights=None)`：Cauchy 组合检验；
- `RW_MTCCT(pvalue_mat, num_region_r, num_region_c, threshold=0.05)`：
  找出包含最显著子区域的显著连通分量，并用 CCT 组合其 p 值（聚类级检验）。

#### 4.4 `run_simulation.py` — 串行模拟主程序

对标 R 版 `analysis_RWSPM_fixed.R`，对指定 Setting 逐轮执行：
生成数据 → 固定滑窗（`window.width=20`）→ LQD → 逐区域 BCov 检验 →
RW-MTCCT → 保存 CSV。固定参数：`M1=150, M2=100, n=200, threshold=0.05`。

```bash
python run_simulation.py                    # Setting 1–7，各 100 轮
python run_simulation.py --settings 1 3 5   # 只跑 Setting 1、3、5
python run_simulation.py --n_rep 10         # 每设置只跑 10 轮（快速测试）
python run_simulation.py --method limit     # 改用极限分布 p 值（需 cball_ext）
python run_simulation.py --n_perm 199 --output_root ./data/result
```

参数：`--settings`（1–7 列表）、`--n_rep`（默认 100）、`--method`
（`gamma`/`limit`/`permu`，默认 `gamma`）、`--n_perm`（默认 99）、
`--output_root`（默认 `./data/result`）。

#### 4.5 `run_simu_parallel.py` — 多核并行主程序

与 `run_simulation.py` 功能相同，但实现三层并行：

1. LQD 区域级并行；
2. BCov 区域级并行；
3. 重复/设置级并行（`--rep_workers > 1` 时多个 Setting 同时跑）。

自动检测 CPU 核心数（默认 `n_workers = cpu_count - 1`）。Windows（spawn）
下子进程不允许嵌套进程池，程序会自动关闭区域级并行、改用重复级并行。

```bash
python run_simu_parallel.py                          # 默认全部 Setting
python run_simu_parallel.py --settings 1 3 5         # 指定 Setting
python run_simu_parallel.py --n_rep 10 --n_perm 20   # 快速测试
python run_simu_parallel.py --n_workers 4            # 区域级 worker 数
python run_simu_parallel.py --rep_workers 2          # 重复级并行（设置级）
python run_simu_parallel.py --window_width 0         # 0/负 → 自动搜索窗宽
python run_simu_parallel.py --window_width 0 --width_search single   # 单核 JSD 选窗
```

灵敏度分析参数（额外）：`--n`（样本量，CLI 默认 50）、`--M1`/`--M2`（图像尺寸）、
`--window_width`（滑窗宽度，默认 20，≤0 表示自动 JSD 搜索）、`--step_divisor`
（步长 = ceil(window_width/step_divisor)，默认 4）、`--n_quantile`（LQD 插值点数，
默认 21）、`--kde_bw`（固定 KDE 带宽，默认自动 bw.nrd0）、`--width_search`
（JSD 自动选窗的并行模式：默认 `parallel` 并行 / `single` 单核，仅自动选窗时生效）。

### 5. 输出结果说明（data 文件夹）

运行结束后，每个 Setting 的结果存放在 `data/result/Setting{k}/` 下：

#### 5.1 逐区域结果：`test_rwspm{l}.csv`（每一轮循环一个文件）

- `l` 表示第 `l` 轮重复（`l = 1, ..., n_rep`），即**外层循环**；
- 文件内每一行 `j` 对应**内层循环**的第 `j` 个子区域，共 `m` 行（无表头行
  时即区域数）；
- 三列含义（与 R 版 `colnames = c("region.j", method, "pvalue")` 一致）：

| 列名 | 含义 |
|---|---|
| `region.j` | 子区域编号，`j = 1, 2, ..., m` |
| `{method}`（如 `gamma`） | 该区域的 **BCov² 统计量**（注意：列名是 p 值方法名，但存放的是统计量） |
| `pvalue` | 该区域的检验 p 值 |

> 默认参数（`window_width=20`，步长 5，`M1=150, M2=100`）下共有
> **m = 486** 个子区域（27 行 × 18 列）。

**region.j 与图像坐标的对应关系**（区域网格按行优先排列）：

```python
width_x, width_y = 27, 18          # 默认配置下的行数、列数
row = (j - 1) // width_y + 1       # 图像行方向（x）
col = (j - 1) %  width_y + 1       # 图像列方向（y）
```

即 `region.j = 1` 对应左上角（row=1, col=1）的子区域。

**读取示例**（pandas）：

```python
import pandas as pd
df = pd.read_csv("data/result/Setting2/test_rwspm1.csv")
# 第 1 轮重复、第 5 个区域的统计量与 p 值：
row5 = df[df["region.j"] == 5].iloc[0]
stat, pval = row5["gamma"], row5["pvalue"]
```

#### 5.2 聚类级结果：`rwspm_cluster_set{k}.csv`

- 单个文件，共 `n_rep` 行（每轮重复一行）；
- 列名 `rw.mtcct`：该轮重复的**聚类级组合 p 值**（RW-MTCCT 输出）；
- 第 `l` 行对应 `test_rwspm{l}.csv` 的第 `l` 轮重复。

```python
cl = pd.read_csv("data/result/Setting2/rwspm_cluster_set2.csv", header=None,
                 names=["rw.mtcct"])
# 第 10 轮重复的聚类 p 值：
cl.iloc[9]["rw.mtcct"]
```

**小结**：`Setting{k}/test_rwspm{l}.csv` 的每一行 = 第 l 轮循环中第 j 个
region 的统计量（第 2 列）与 p 值（第 3 列）；`Setting{k}/rwspm_cluster_set{k}.csv`
的第 l 行 = 第 l 轮循环的聚类级 p 值。

### 6. 注意事项

- `rwspm.RWSPM()` 内的 p 值是占位符，正式模拟请用 `run_simulation.py` /
  `run_simu_parallel.py`（内部调用 `run_rwspm_pipeline` / `_run_one_rep`）；
- `method='limit'` 需要编译好的 `cball_ext`（DLL/SO），否则自动回退到 `gamma`；
- Windows 下 `run_simu_parallel.py` 的 `--rep_workers > 1` 时区域级并行自动关闭
  （避免嵌套进程池）；
- 每轮重复的随机种子固定为 `random_state = 2026 + l * 7`，结果可复现；
- 本实现对标 R 版 `analysis_RWSPM_fixed.R`，如需方法细节请参阅 RWSPM 论文。

---

## 繁體中文

### 1. 專案簡介

本目錄包含 RWSPM 論文的 Python 模擬程式碼，用於評估 RWSPM 方法在 7 種模擬
設定下的表現（Type I error 與檢定功效）。

模擬流程（對齊 R 版 `analysis_RWSPM_fixed.R`）：

1. **資料生成**：`gendata_xy.py` 依 Setting 1–7 生成 `(X, Y, Z)`；
2. **影像分割**：以滑窗將 `M1 × M2` 影像劃分為多個重疊方形子區域；
3. **LQD 轉換**：對每個子區域估計密度並做 LQD 轉換；
4. **逐區域檢定**：對每個子區域計算 BCov² 統計量與 p 值
   （`gamma` / `limit` / `permu` 三種方法）；
5. **叢集組合**：以 RW-MTCCT 合併空間連通的顯著子區域，得到叢集級 p 值；
6. **儲存結果**：逐區域 CSV + 叢集級 CSV。

### 2. 目錄結構

```
Python/
├── run_simulation.py      # 串列模擬主程式（對標 analysis_RWSPM.R）
├── run_simu_parallel.py   # 多核心平行模擬主程式
├── rwspm.py               # RWSPM 核心演算法（分割、滑窗、CCT、RW-MTCCT）
├── bcov.py                # Ball Covariance 統計量與檢定
├── gendata_xy.py          # 模擬資料生成（Setting 1–7）
├── lqd.py                 # LQD（對數分位密度）轉換
├── gammafit.py            # Gamma 動差配適（method='gamma' 用）
├── seed.py                # 隨機種子工具
├── lqd_mix.py             # LQD 混合模型變體（輔助）
├── permutation_null.py    # 置換零分佈輔助腳本
├── example_rwspm.ipynb    # 範例 Notebook
├── cball_ext/             # C 擴充（加速 BDD 核矩陣，'limit' 方法需要）
└── data/result/           # 模擬輸出（見第 5 節）
    └── Setting{k}/        # k = 1..7
        ├── test_rwspm{l}.csv         # 第 l 輪重複的逐區域結果
        └── rwspm_cluster_set{k}.csv  # 叢集級組合 p 值
```

### 3. 環境需求

- Python ≥ 3.8
- `numpy`、`scipy`、`networkx`
- （選用，僅 `limit` 方法）`ctypes` + 已編譯的 `cball_ext.dll`
  （Windows 用 MinGW，見 `cball_ext/compile_win.bat`；Linux 用 `compile_linux.sh`）

```bash
pip install numpy scipy networkx
```

### 4. 檔案功能說明

#### 4.1 `gendata_xy.py` — 資料生成

| Setting | a | sigma_xy | model | 說明 |
|:---:|:---:|:---:|:---:|:---|
| 1 | 0 | 0 | independent | 虛無假設（X ⊥ Y），用於 Type I error |
| 2–4 | 1 | 1/2/3 | linear | 線性模型 |
| 5–7 | 1 | 1/2/3 | nonlinear | 非線性模型 |

- 網格：`[-50, 100] × [-50, 50]` 上的 `M1 × M2` 個體素；
- 疾病區域 `A = [-10, 10] × [-10, 10]`；每個個體有隨機位置/半徑的受影響
  子區域 `A_i`，訊號 `β(s) = 0.8·a·exp(−0.1·d)`；
- 回傳：`X`（基因型劑量，n×1，值域 {0,1,2}）、`Y`（攤平的影像響應，n×(M1·M2)）、
  `Z`（共變量 [age/100, gender]，n×2）。

```python
from gendata_xy import generate_data
X, Y, Z = generate_data(setting=2, n=200, M1=150, M2=100, random_state=2026)
```

#### 4.2 `bcov.py` — Ball Covariance

- `bcov(X, Y)`：計算經驗 Ball Covariance 統計量 `BCov²_n(X, Y)`；
- `bcov_perm_test(x, y, n_perm=199, method='gamma', DY=None)`：BCov 檢定，
  回傳 `(stat, pval)`。`method`：
  - `'permu'`：傳統置換 p 值 `(∑I(stat_perm ≥ stat)+1)/(n_perm+1)`；
  - `'gamma'`（預設）：對置換零分佈做 Gamma 動差配適近似（`gammafit.py`）；
  - `'limit'`：HBE 三累積量 Gamma 極限近似（需 `cball_ext` 編譯庫）；
- `bcov_perm_test_gwas(x, DY, ...)`：預先計算 `DY` 的變體，避免重複計算。

#### 4.3 `rwspm.py` — RWSPM 核心演算法

- `slide_width_step(b, width, M1, M2)`：滑窗分割，回傳各子區域像素索引
  （`idx` 為 m×width² 的 0 基索引矩陣）及 `width_x` / `width_y`；
- `compute_ov` + `golden_section_search`：以 JSD 目標函數（含懲罰項）自動
  選擇最優滑窗寬度；
- `RWSPM(x, y, M1, M2, window_width=None, method='bcov')`：完整流程封裝。
  > ⚠️ 注意：`RWSPM()` 內的 p 值為佔位符（固定 1.0）。**實際模擬請使用**
  > `run_simulation.py` 的 `run_rwspm_pipeline()`（或直接執行兩個主程式），
  > 它會呼叫 `bcov_perm_test` 計算真實 p 值。
- `CCT_chisq(pvals, weights=None)`：Cauchy 組合檢定；
- `RW_MTCCT(pvalue_mat, num_region_r, num_region_c, threshold=0.05)`：
  找出包含最顯著子區域的顯著連通分量，並以 CCT 組合其 p 值（叢集級檢定）。

#### 4.4 `run_simulation.py` — 串列模擬主程式

對標 R 版 `analysis_RWSPM_fixed.R`，對指定 Setting 逐輪執行：
生成資料 → 固定滑窗（`window.width=20`）→ LQD → 逐區域 BCov 檢定 →
RW-MTCCT → 儲存 CSV。固定參數：`M1=150, M2=100, n=200, threshold=0.05`。

```bash
python run_simulation.py                    # Setting 1–7，各 100 輪
python run_simulation.py --settings 1 3 5   # 只跑 Setting 1、3、5
python run_simulation.py --n_rep 10         # 每設定只跑 10 輪（快速測試）
python run_simulation.py --method limit     # 改用極限分佈 p 值（需 cball_ext）
python run_simulation.py --n_perm 199 --output_root ./data/result
```

參數：`--settings`（1–7 列表）、`--n_rep`（預設 100）、`--method`
（`gamma`/`limit`/`permu`，預設 `gamma`）、`--n_perm`（預設 99）、
`--output_root`（預設 `./data/result`）。

#### 4.5 `run_simu_parallel.py` — 多核心平行主程式

與 `run_simulation.py` 功能相同，但實現三層平行：

1. LQD 區域級平行；
2. BCov 區域級平行；
3. 重複/設定級平行（`--rep_workers > 1` 時多個 Setting 同時跑）。

自動偵測 CPU 核心數（預設 `n_workers = cpu_count - 1`）。Windows（spawn）
下子行程不允許巢狀行程池，程式會自動關閉區域級平行、改用重複級平行。

```bash
python run_simu_parallel.py                          # 預設全部 Setting
python run_simu_parallel.py --settings 1 3 5         # 指定 Setting
python run_simu_parallel.py --n_rep 10 --n_perm 20   # 快速測試
python run_simu_parallel.py --n_workers 4            # 區域級 worker 數
python run_simu_parallel.py --rep_workers 2          # 重複級平行（設定級）
python run_simu_parallel.py --window_width 0         # 0/負 → 自動搜尋窗寬
python run_simu_parallel.py --window_width 0 --width_search single   # 單核 JSD 選窗
```

靈敏度分析參數（額外）：`--n`（樣本量，CLI 預設 50）、`--M1`/`--M2`（影像尺寸）、
`--window_width`（滑窗寬度，預設 20，≤0 表示自動 JSD 搜尋）、`--step_divisor`
（步長 = ceil(window_width/step_divisor)，預設 4）、`--n_quantile`（LQD 插值點數，
預設 21）、`--kde_bw`（固定 KDE 頻寬，預設自動 bw.nrd0）、`--width_search`
（JSD 自動選窗的平行模式：預設 `parallel` 平行 / `single` 單核，僅自動選窗時生效）。

### 5. 輸出結果說明（data 資料夾）

每個 Setting 的結果存放在 `data/result/Setting{k}/` 下：

#### 5.1 逐區域結果：`test_rwspm{l}.csv`（每一輪迴圈一個檔案）

- `l` 為第 `l` 輪重複（`l = 1, ..., n_rep`），即**外層迴圈**；
- 檔內第 `j` 列對應**內層迴圈**的第 `j` 個子區域，共 `m` 列；
- 三欄（與 R 版 `colnames = c("region.j", method, "pvalue")` 一致）：

| 欄名 | 意義 |
|---|---|
| `region.j` | 子區域編號，`j = 1, 2, ..., m` |
| `{method}`（如 `gamma`） | 該區域的 **BCov² 統計量**（注意：欄名是 p 值方法名，但存放的是統計量） |
| `pvalue` | 該區域的檢定 p 值 |

> 預設參數（`window_width=20`，步長 5，`M1=150, M2=100`）下共有
> **m = 486** 個子區域（27 列 × 18 行）。

**region.j 與影像座標的對應**（區域網格按列優先排列）：

```python
width_x, width_y = 27, 18          # 預設設定的列數、行數
row = (j - 1) // width_y + 1       # 影像列方向（x）
col = (j - 1) %  width_y + 1       # 影像行方向（y）
```

即 `region.j = 1` 對應左上角（row=1, col=1）的子區域。

**讀取範例**（pandas）：

```python
import pandas as pd
df = pd.read_csv("data/result/Setting2/test_rwspm1.csv")
row5 = df[df["region.j"] == 5].iloc[0]      # 第 1 輪、第 5 個區域
stat, pval = row5["gamma"], row5["pvalue"]  # 統計量與 p 值
```

#### 5.2 叢集級結果：`rwspm_cluster_set{k}.csv`

- 單一檔案，共 `n_rep` 列（每輪一列）；
- 欄名 `rw.mtcct`：該輪的**叢集級組合 p 值**（RW-MTCCT 輸出）；
- 第 `l` 列對應 `test_rwspm{l}.csv` 的第 `l` 輪。

```python
cl = pd.read_csv("data/result/Setting2/rwspm_cluster_set2.csv", header=None,
                 names=["rw.mtcct"])
cl.iloc[9]["rw.mtcct"]     # 第 10 輪的叢集 p 值
```

**小結**：`Setting{k}/test_rwspm{l}.csv` 每列 = 第 l 輪迴圈中第 j 個 region 的
統計量（第 2 欄）與 p 值（第 3 欄）；`Setting{k}/rwspm_cluster_set{k}.csv`
第 l 列 = 第 l 輪迴圈的叢集級 p 值。

### 6. 注意事項

- `rwspm.RWSPM()` 內的 p 值是佔位符，正式模擬請用 `run_simulation.py` /
  `run_simu_parallel.py`；
- `method='limit'` 需要已編譯的 `cball_ext`（DLL/SO），否則自動回退 `gamma`；
- Windows 下 `--rep_workers > 1` 時區域級平行自動關閉（避免巢狀行程池）；
- 每輪隨機種子固定為 `random_state = 2026 + l * 7`，結果可重現；
- 本實作對標 R 版 `analysis_RWSPM_fixed.R`，方法細節請參閱 RWSPM 論文。

---

## English

### 1. Overview

This directory contains the Python simulation code for the RWSPM paper,
evaluating the method under 7 simulation settings (Type I error and power).

Pipeline (mirrors the R version `analysis_RWSPM_fixed.R`):

1. **Data generation** — `gendata_xy.py` creates `(X, Y, Z)` for Settings 1–7;
2. **Image partitioning** — a sliding window splits the `M1 × M2` image into
   overlapping square sub-regions;
3. **LQD transform** — estimate the density of each sub-region and apply the
   LQD (Log Quantile Density) transform;
4. **Region-wise test** — compute the BCov² statistic and p-value for every
   sub-region (`gamma` / `limit` / `permu`);
5. **Cluster combination** — RW-MTCCT merges spatially connected significant
   regions into a cluster-level p-value;
6. **Save results** — per-region CSV + cluster-level CSV.

### 2. Directory Structure

```
Python/
├── run_simulation.py      # Sequential simulation driver (port of analysis_RWSPM.R)
├── run_simu_parallel.py   # Multi-core parallel simulation driver
├── rwspm.py               # Core RWSPM algorithm (partition, sliding window, CCT, RW-MTCCT)
├── bcov.py                # Ball Covariance statistic and tests
├── gendata_xy.py          # Simulation data generator (Settings 1–7)
├── lqd.py                 # LQD (Log Quantile Density) transform
├── gammafit.py            # Gamma moment matching (for method='gamma')
├── seed.py                # Random seed utilities
├── lqd_mix.py             # LQD mixture-model variant (auxiliary)
├── permutation_null.py    # Permutation null-distribution helper
├── example_rwspm.ipynb    # Example notebook
├── cball_ext/             # C extension (accelerates BDD kernel, needed for 'limit')
└── data/result/           # Simulation output (see Section 5)
    └── Setting{k}/        # k = 1..7
        ├── test_rwspm{l}.csv         # Per-region results of replication l
        └── rwspm_cluster_set{k}.csv  # Cluster-level combined p-values
```

### 3. Requirements

- Python ≥ 3.8
- `numpy`, `scipy`, `networkx`
- (Optional, only for `limit`) `ctypes` + compiled `cball_ext.dll`
  (MinGW on Windows, see `cball_ext/compile_win.bat`; Linux: `compile_linux.sh`)

```bash
pip install numpy scipy networkx
```

### 4. File Descriptions

#### 4.1 `gendata_xy.py` — Data Generation

| Setting | a | sigma_xy | model | Description |
|:---:|:---:|:---:|:---:|:---|
| 1 | 0 | 0 | independent | Null hypothesis (X ⊥ Y), for Type I error |
| 2–4 | 1 | 1/2/3 | linear | Linear model |
| 5–7 | 1 | 1/2/3 | nonlinear | Non-linear model |

- Grid: `M1 × M2` voxels over `[-50, 100] × [-50, 50]`;
- Disease region `A = [-10, 10] × [-10, 10]`; each subject has an affected
  sub-region `A_i` with random center/radius, signal `β(s) = 0.8·a·exp(−0.1·d)`;
- Returns: `X` (genotype dosage, n×1, values {0,1,2}), `Y` (flattened image
  response, n×(M1·M2)), `Z` (covariates [age/100, gender], n×2).

```python
from gendata_xy import generate_data
X, Y, Z = generate_data(setting=2, n=200, M1=150, M2=100, random_state=2026)
```

#### 4.2 `bcov.py` — Ball Covariance

- `bcov(X, Y)`: empirical Ball Covariance statistic `BCov²_n(X, Y)`;
- `bcov_perm_test(x, y, n_perm=199, method='gamma', DY=None)`: BCov test
  returning `(stat, pval)`. `method`:
  - `'permu'` — traditional permutation p-value `(∑I(stat_perm ≥ stat)+1)/(n_perm+1)`;
  - `'gamma'` (default) — Gamma approximation via moment matching on the
    permutation null (`gammafit.py`);
  - `'limit'` — HBE 3-cumulant Gamma limit approximation (requires `cball_ext`);
- `bcov_perm_test_gwas(x, DY, ...)`: variant with precomputed `DY` to avoid
  repeated distance computations.

#### 4.3 `rwspm.py` — Core RWSPM Algorithm

- `slide_width_step(b, width, M1, M2)`: sliding-window partition; returns the
  0-based pixel indices per region (`idx`, m×width²) plus `width_x` / `width_y`;
- `compute_ov` + `golden_section_search`: automatic optimal window-width
  selection via a JSD objective with a penalty term;
- `RWSPM(x, y, M1, M2, window_width=None, method='bcov')`: full pipeline wrapper.
  > ⚠️ Note: p-values inside `RWSPM()` are placeholders (fixed at 1.0).
  > **For real simulations use** `run_rwspm_pipeline()` from `run_simulation.py`
  > (or the two driver scripts), which calls `bcov_perm_test` for true p-values.
- `CCT_chisq(pvals, weights=None)`: Cauchy Combination Test;
- `RW_MTCCT(pvalue_mat, num_region_r, num_region_c, threshold=0.05)`: finds the
  significant connected component containing the most significant region and
  combines their p-values with CCT (cluster-level test).

#### 4.4 `run_simulation.py` — Sequential Driver

Mirrors R's `analysis_RWSPM_fixed.R`: for each requested Setting, run
data generation → fixed sliding window (`window.width=20`) → LQD → region-wise
BCov test → RW-MTCCT → save CSVs. Fixed params: `M1=150, M2=100, n=200,
threshold=0.05`.

```bash
python run_simulation.py                    # Settings 1–7, 100 reps each
python run_simulation.py --settings 1 3 5   # only Settings 1, 3, 5
python run_simulation.py --n_rep 10         # 10 reps per setting (quick test)
python run_simulation.py --method limit     # use limit-distribution p-values (needs cball_ext)
python run_simulation.py --n_perm 199 --output_root ./data/result
```

Args: `--settings` (list of 1–7), `--n_rep` (default 100), `--method`
(`gamma`/`limit`/`permu`, default `gamma`), `--n_perm` (default 99),
`--output_root` (default `./data/result`).

#### 4.5 `run_simu_parallel.py` — Parallel Driver

Same functionality as `run_simulation.py` but with three levels of parallelism:

1. Region-level parallel LQD;
2. Region-level parallel BCov tests;
3. Replication/setting-level parallelism (multiple Settings at once with
   `--rep_workers > 1`).

Auto-detects CPU count (default `n_workers = cpu_count - 1`). On Windows
(spawn), child processes cannot create nested process pools, so the script
automatically disables region-level parallelism and uses rep-level parallelism.

```bash
python run_simu_parallel.py                          # all Settings by default
python run_simu_parallel.py --settings 1 3 5         # specific Settings
python run_simu_parallel.py --n_rep 10 --n_perm 20   # quick test
python run_simu_parallel.py --n_workers 4            # region-level workers
python run_simu_parallel.py --rep_workers 2          # rep-level parallelism
python run_simu_parallel.py --window_width 0         # 0/negative → auto width search
python run_simu_parallel.py --window_width 0 --width_search single  # single-core JSD search
```

Extra sensitivity-analysis args: `--n` (sample size, CLI default 50),
`--M1`/`--M2` (image size), `--window_width` (default 20; ≤0 → auto JSD
search), `--step_divisor` (step = ceil(window_width/step_divisor), default 4),
`--n_quantile` (LQD interpolation points, default 21), `--kde_bw` (fixed KDE
bandwidth, default auto bw.nrd0), `--width_search` (JSD width-search mode when
`window_width` is auto: `parallel` default / `single`).

### 5. Output: How to Find the Statistic & p-value of Each Region in Each Loop

All results for setting `k` are stored under `data/result/Setting{k}/`:

#### 5.1 Per-region results: `test_rwspm{l}.csv` (one file per replication)

- `l` is the replication number (`l = 1, ..., n_rep`) — the **outer loop**;
- Row `j` in the file corresponds to sub-region `j` of the **inner loop**;
  there are `m` data rows in total;
- The three columns (matching R's `colnames = c("region.j", method, "pvalue")`):

| Column | Meaning |
|---|---|
| `region.j` | Region index, `j = 1, 2, ..., m` |
| `{method}` (e.g. `gamma`) | The region's **BCov² statistic** (note: the column is *named* after the p-value method, but it holds the statistic) |
| `pvalue` | Region-level test p-value |

> With the default parameters (`window_width=20`, step 5, `M1=150, M2=100`)
> there are **m = 486** sub-regions (27 rows × 18 columns).

**Mapping `region.j` to image coordinates** (region grid in row-major order):

```python
width_x, width_y = 27, 18          # rows / columns under the default setup
row = (j - 1) // width_y + 1       # image row direction (x)
col = (j - 1) %  width_y + 1       # image column direction (y)
```

So `region.j = 1` is the top-left sub-region (row=1, col=1).

**Reading example** (pandas):

```python
import pandas as pd
df = pd.read_csv("data/result/Setting2/test_rwspm1.csv")
row5 = df[df["region.j"] == 5].iloc[0]      # replication 1, region 5
stat, pval = row5["gamma"], row5["pvalue"]  # statistic and p-value
```

#### 5.2 Cluster-level results: `rwspm_cluster_set{k}.csv`

- One file with `n_rep` rows (one per replication);
- Column `rw.mtcct`: the **cluster-level combined p-value** (RW-MTCCT output)
  of that replication;
- Row `l` corresponds to replication `l` of `test_rwspm{l}.csv`.

```python
cl = pd.read_csv("data/result/Setting2/rwspm_cluster_set2.csv", header=None,
                 names=["rw.mtcct"])
cl.iloc[9]["rw.mtcct"]     # cluster p-value of replication 10
```

**Summary**: each row of `Setting{k}/test_rwspm{l}.csv` gives the statistic
(column 2) and p-value (column 3) of region `j` in replication `l`;
row `l` of `Setting{k}/rwspm_cluster_set{k}.csv` gives the cluster-level
p-value of replication `l`.

### 6. Notes

- The p-values inside `rwspm.RWSPM()` are placeholders; use `run_simulation.py`
  / `run_simu_parallel.py` for real simulations;
- `method='limit'` requires a compiled `cball_ext` (DLL/SO); otherwise it
  automatically falls back to `gamma`;
- On Windows, region-level parallelism is disabled when `--rep_workers > 1`
  (avoids nested process pools);
- The random seed per replication is fixed at `random_state = 2026 + l * 7`,
  so results are reproducible;
- This is a Python port of R's `analysis_RWSPM_fixed.R`; see the RWSPM paper
  for methodological details.
