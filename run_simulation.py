"""
run_simulation.py — RWSPM 模拟脚本 (Python 版 analysis_RWSPM_fixed.R)

对标 R 项目 code_simulation/code_RWSPM/analysis_RWSPM_fixed.R，
对 Settings 1-7 分别执行 n_rep 次重复模拟:

  1. 通过 gendata_xy.generate_data() 生成数据（代替 R 从 .RData 加载）
  2. 固定滑窗宽度 window.width = 20（与 R 一致）
  3. 图像分区 + LQD 变换
  4. 逐区域 BCov 检验 + p-value（method 可选 gamma / limit / permu）
  5. RW-MTCCT 空间聚类组合检验
  6. 保存逐区域 CSV 和 cluster-level CSV

Usage
-----
    python run_simulation.py                   # Settings 1-7, 各 100 次
    python run_simulation.py --settings 1 3 5  # 只跑 Settings 1,3,5
    python run_simulation.py --n_rep 10        # 快速测试 10 次

R 版关键参数对照:
    k       = setting (1-7)
    n_rep   = 100 (重复次数)
    M1, M2  = 150, 100 (图像尺寸)
    window.width = 20 (滑窗宽度)
    threshold    = 0.05 (MTCCT 显著性阈值)
"""

import os
import sys
import time
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---- 添加项目路径 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwspm import slide_width_step, RW_MTCCT
from lqd import LQD
from bcov import bcov_perm_test
from gendata_xy import generate_data


# ===========================================================================
#  Constants (matching R: analysis_RWSPM_fixed.R)
# ===========================================================================
M1 = 150          # image height
M2 = 100          # image width
N  = 200          # sample size
WINDOW_WIDTH = 20 # fixed sliding window width (R uses window.width=20)
N_REP = 100       # number of replications
THRESHOLD = 0.05  # significance threshold for MTCCT


# ===========================================================================
#  Helper: run RWSPM with proper p-values for one dataset
# ===========================================================================

def run_rwspm_pipeline(x, y, M1, M2, window_width=20, method="gamma", n_perm=99):
    """
    Execute the full RWSPM pipeline for one (X, Y) dataset.

    Equivalent to R's::
        rw.pvalue <- RWSPM(x, y, M1, M2, window.width=20)

    but computes real p-values instead of placeholders.

    Parameters
    ----------
    x : ndarray, shape (n, p_x)
        Predictor variable(s).
    y : ndarray, shape (n, M1*M2)
        Flattened image data.
    M1, M2 : int
        Image dimensions.
    window_width : int
        Sliding window width (default 20, matching R).
    method : str
        ``"gamma"`` — Gamma approximation via permutation null (default).
        ``"limit"`` — Limit distribution via HBE (needs cball_ext DLL).
        ``"permu"`` — Traditional permutation p-value.
    n_perm : int
        Number of permutations (default 99; only used for gamma/permu).

    Returns
    -------
    dict with keys:
        rw_pvalue : ndarray, shape (m, 3)
            Columns: region index (1-based), BCov² statistic, p-value.
        num_region_r : int
            Number of sub-regions along rows.
        num_region_c : int
            Number of sub-regions along columns.
        width : int
            Window width used.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.shape[0]

    # --- Image partition (same as RWSPM with window.width=20) ---
    b_step = int(np.ceil(window_width / 4))  # = 5 when width=20
    idx_dict = slide_width_step(b_step, window_width, M1, M2)
    sub_idx = idx_dict["idx"]
    m = sub_idx.shape[0]

    # --- LQD transformation ---
    G = LQD(y, sub_idx)

    # --- Per-region BCov test ---
    rw_pvalue = np.zeros((m, 3))
    rw_pvalue[:, 0] = np.arange(1, m + 1)  # region index (1-based)

    print(f"    Testing {m} regions (method={method})...")
    for j in range(m):
        g_j = np.asarray(G[j])
        try:
            stat, pval = bcov_perm_test(x, g_j, n_perm=n_perm, method=method)
        except Exception as e:
            # Fallback: if 'limit' method fails (e.g. no C extension),
            # use 'gamma' with more permutations
            if method == "limit":
                print(f"      [WARN] limit method failed (j={j}), falling back to gamma: {e}")
                stat, pval = bcov_perm_test(x, g_j, n_perm=n_perm * 2, method="gamma")
            else:
                print(f"      [WARN] bcov_perm_test failed at region {j}: {e}")
                stat, pval = 0.0, 1.0
        rw_pvalue[j, 1] = stat
        rw_pvalue[j, 2] = pval

        if (j + 1) % 20 == 0:
            print(f"      ... {j+1}/{m} regions done")

    return {
        "rw_pvalue": rw_pvalue,
        "num_region_r": idx_dict["width_x"],
        "num_region_c": idx_dict["width_y"],
        "width": window_width,
    }


# ===========================================================================
#  Main simulation loop
# ===========================================================================

def run_simulation(settings=None, n_rep=N_REP, method="gamma",
                   n_perm=99, output_root="./data/result"):
    """
    Run the RWSPM simulation for one or more settings.

    Parameters
    ----------
    settings : list of int or None
        Setting numbers to run (default: all 1-7).
    n_rep : int
        Number of replications per setting (default 100).
    method : str
        P-value method: "gamma", "limit", or "permu".
    n_perm : int
        Permutations for gamma/permu methods (default 99).
    output_root : str
        Root directory for output (default "./data/result").
    """
    if settings is None:
        settings = list(range(1, 8))

    total_start = time.time()

    for k in settings:
        print(f"\n{'='*60}")
        print(f"Setting {k}  ({n_rep} replications, method={method})")
        print(f"{'='*60}")

        save_root = os.path.join(output_root, f"Setting{k}")
        os.makedirs(save_root, exist_ok=True)

        rw_mtcct = np.zeros(n_rep)

        setting_start = time.time()

        for l in range(1, n_rep + 1):
            rep_start = time.time()

            # ---- Step 1: Generate data ----
            x, y, _ = generate_data(
                setting=k, n=N, M1=M1, M2=M2,
                random_state=2026 + l * 7   # deterministic seed per rep
            )

            # ---- Step 2: RWSPM pipeline ----
            print(f"\n  Rep {l}/{n_rep}")
            result = run_rwspm_pipeline(
                x, y, M1, M2,
                window_width=WINDOW_WIDTH,
                method=method,
                n_perm=n_perm,
            )

            rw_pvalue = result["rw_pvalue"]

            # ---- Step 3: Save per-region p-values (matching R) ----
            csv_path = os.path.join(save_root, f"test_rwspm{l}.csv")
            # R: colnames = c("region.j", method, "pvalue")
            header = f"region.j,{method},pvalue"
            np.savetxt(csv_path, rw_pvalue, delimiter=",",
                       header=header, comments="", fmt="%.10f")

            # ---- Step 4: RW-MTCCT cluster test ----
            # Note: pass only the p-value column (index 2), not the full
            # [region.j, statistic, pvalue] matrix.  RW_MTCCT expects
            # shape (n_hypotheses, n_regions) or (n_regions,).
            rw_mtcct[l - 1] = RW_MTCCT(
                rw_pvalue[:, 2],       # p-value column only
                result["num_region_r"],
                result["num_region_c"],
                threshold=THRESHOLD,
            )

            rep_elapsed = time.time() - rep_start
            print(f"    -> cluster p = {rw_mtcct[l-1]:.6e}  "
                  f"({rep_elapsed:.1f}s)")

        # ---- Save cluster-level p-values (matching R) ----
        cluster_csv = os.path.join(save_root, f"rwspm_cluster_set{k}.csv")
        np.savetxt(cluster_csv, rw_mtcct, delimiter=",",
                   header="rw.mtcct", comments="", fmt="%.10f")

        setting_elapsed = time.time() - setting_start
        print(f"\n  [Done] Setting {k}: {n_rep} replications in "
              f"{setting_elapsed:.1f}s -> {cluster_csv}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All done. Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RWSPM simulation script (Python port of analysis_RWSPM_fixed.R)"
    )
    parser.add_argument(
        "--settings", nargs="+", type=int, default=None,
        help="Setting numbers to run (1-7). Default: all."
    )
    parser.add_argument(
        "--n_rep", type=int, default=N_REP,
        help=f"Number of replications per setting (default {N_REP})."
    )
    parser.add_argument(
        "--method", type=str, default="gamma",
        choices=["gamma", "limit", "permu"],
        help="P-value method: gamma (default), limit, or permu."
    )
    parser.add_argument(
        "--n_perm", type=int, default=99,
        help="Number of permutations for gamma/permu (default 99)."
    )
    parser.add_argument(
        "--output_root", type=str, default="./data/result",
        help="Output directory root (default ./data/result)."
    )
    args = parser.parse_args()

    run_simulation(
        settings=args.settings,
        n_rep=args.n_rep,
        method=args.method,
        n_perm=args.n_perm,
        output_root=args.output_root,
    )
