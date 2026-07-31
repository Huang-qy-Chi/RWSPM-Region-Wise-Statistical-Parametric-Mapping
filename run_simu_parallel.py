"""
run_simu_parallel.py — RWSPM 模拟脚本（多核并行版）

对标 run_simulation.py，使用 concurrent.futures.ProcessPoolExecutor 实现三层并行：

  1. LQD 区域级并行  — _lqd_parallel()
  2. BCov 区域级并行 — _bcov_test_parallel()
  3. 重复模拟级并行  — Settings 分配到不同进程同时运行

自动检测 CPU 核心数（multiprocessing.cpu_count()），默认使用 n_workers - 1 个 worker
（预留 1 核给系统）。

Usage
-----
    python run_simu_parallel.py                          # Settings 1-7, 并行
    python run_simu_parallel.py --settings 1 3 5         # 只跑特定 Settings
    python run_simu_parallel.py --n_rep 10 --n_perm 20   # 快速测试
    python run_simu_parallel.py --n_workers 4            # 指定 worker 数
    python run_simu_parallel.py --rep_workers 2          # 重复级并行 worker 数
"""

import os
import sys
import time
import argparse
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

warnings.filterwarnings("ignore")

# ---- 添加项目路径 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwspm import slide_width_step, golden_section_search, compute_ov, RW_MTCCT
from bcov import bcov_perm_test
from gendata_xy import generate_data


# ===========================================================================
#  Constants (remaining)
# ===========================================================================
THRESHOLD = 0.05


# ===========================================================================
#  Parallel worker helpers (top-level functions — required for
#  ProcessPoolExecutor pickling)
# ===========================================================================

def _lqd_one_region(args):
    """
    Compute LQD for a single region.  Top-level for ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple (y, col_idx, t, kde_bw)
        y : ndarray (n, p) — full image data
        col_idx : ndarray (w*w,) — 0-based pixel indices
        t : ndarray (L,) — quantile grid
        kde_bw : float or None — fixed KDE bandwidth, or None for auto bw.nrd0

    Returns
    -------
    G_j : ndarray (n, L) — LQD values for this region
    """
    y, col_idx, t, kde_bw = args
    n = y.shape[0]
    L = len(t)
    qd = np.zeros((n, L))

    for i in range(n):
        y_slice = y[i, col_idx]

        if kde_bw is not None:
            bw = kde_bw
        else:
            # Bandwidth (R's bw.nrd0)
            sd_val = np.std(y_slice, ddof=1)
            iqr_val = np.percentile(y_slice, 75) - np.percentile(y_slice, 25)
            bw = 0.9 * min(sd_val, iqr_val / 1.34) * len(y_slice) ** (-0.2)

        if len(y_slice) < 2 or bw <= 1e-15:
            qd[i, :] = 1.0 / L
            continue

        # KDE grid
        grid_min = y_slice.min() - 3.0 * bw
        grid_max = y_slice.max() + 3.0 * bw
        g = np.linspace(grid_min, grid_max, 256)

        # Fast KDE
        scaled = (g[:, None] - y_slice[None, :]) / bw
        kernel = np.exp(-0.5 * scaled ** 2)
        dens = kernel.sum(axis=1) / (len(y_slice) * bw * np.sqrt(2.0 * np.pi))
        dens[dens <= 0] = 1e-4

        # Quantiles and interpolation
        q = np.quantile(y_slice, t)
        qd[i, :] = np.interp(q, g, dens)

    G_j = -np.log(qd + 1e-300)
    return G_j


def _bcov_one_region(args):
    """
    Run BCov permutation test for a single region.
    Top-level for ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple (x, G_j, method, n_perm)

    Returns
    -------
    (j, stat, pval) tuple
    """
    x, G_j, method, n_perm = args
    g_j = np.asarray(G_j)
    try:
        stat, pval = bcov_perm_test(x, g_j, n_perm=n_perm, method=method)
    except Exception as e:
        if method == "limit":
            # Fallback to gamma with more permutations
            try:
                stat, pval = bcov_perm_test(x, g_j, n_perm=n_perm * 2, method="gamma")
            except Exception:
                stat, pval = 0.0, 1.0
        else:
            stat, pval = 0.0, 1.0
    return stat, pval


def _run_one_rep(args):
    """
    Run one full replication (data generation + RWSPM pipeline).
    Top-level for ProcessPoolExecutor.

    IMPORTANT (nested-pool safety):
    On Windows (spawn) a child process must NOT create nested
    ProcessPoolExecutors.  The caller sets *use_region_parallel*
    accordingly:
      - Linux (fork): always True → allow nested pools (双层并行)
      - Windows + rep_workers>1: False → regions run sequentially
      - Windows + rep_workers=1: True → region parallel in parent

    Parameters
    ----------
    args : tuple
        (setting, rep_id, M1, M2, n, window_width, step_divisor,
         n_quantile, kde_bw, method, n_perm, threshold,
         n_workers, use_region_parallel)

        window_width : int or None
            If None (or 0), auto-select via JSD golden-section search.
        step_divisor : int
            Denominator for step size: b = ceil(window_width / step_divisor).
        n_quantile : int
            Number of LQD quantile interpolation points.
        kde_bw : float or None
            Fixed KDE bandwidth, or None for auto bw.nrd0.

    Returns
    -------
    dict with keys:
        rep_id, setting, rw_pvalue, num_region_r, num_region_c,
        cluster_pvalue, status, window_width
    """
    (setting, rep_id, M1, M2, n, window_width, step_divisor,
     n_quantile, kde_bw, method, n_perm, threshold,
     n_workers, use_region_parallel) = args

    result = {"rep_id": rep_id, "setting": setting, "status": "ok",
              "window_width": window_width}

    try:
        # ---- Step 1: Generate data ----
        x, y, _ = generate_data(
            setting=setting, n=n, M1=M1, M2=M2,
            random_state=2026 + rep_id * 7,
        )

        # ---- Step 2: Image partition ----
        # Auto window width via JSD golden-section search
        if window_width is None or window_width <= 0:
            window_width = golden_section_search(
                y, compute_ov,
                a=2, b=int(np.ceil(min(M1, M2) / 2)),
                M1=M1, M2=M2,
            )
            result["window_width"] = window_width

        b_step = int(np.ceil(window_width / step_divisor))
        idx_dict = slide_width_step(b_step, window_width, M1, M2)
        sub_idx = idx_dict["idx"]
        m = sub_idx.shape[0]

        # ---- Step 3: LQD (optional region-level parallel) ----
        t = np.linspace(0, 1, n_quantile)
        lqd_args = [(y, sub_idx[j, :].astype(int), t, kde_bw)
                    for j in range(m)]

        if use_region_parallel and n_workers > 1 and m > 1:
            n_lqd_workers = max(1, min(n_workers, m // 2))
            with ProcessPoolExecutor(max_workers=n_lqd_workers) as pool:
                G = list(pool.map(_lqd_one_region, lqd_args))
        else:
            G = [_lqd_one_region(a) for a in lqd_args]

        # ---- Step 4: BCov test (optional region-level parallel) ----
        bcov_args = [(x, G[j], method, n_perm) for j in range(m)]

        rw_pvalue = np.zeros((m, 3))
        rw_pvalue[:, 0] = np.arange(1, m + 1)

        if use_region_parallel and n_workers > 1 and m > 1:
            n_bcov_workers = max(1, min(n_workers, m // 2))
            with ProcessPoolExecutor(max_workers=n_bcov_workers) as pool:
                bcov_results = list(pool.map(_bcov_one_region, bcov_args))
        else:
            bcov_results = [_bcov_one_region(a) for a in bcov_args]

        for j, (stat, pval) in enumerate(bcov_results):
            rw_pvalue[j, 1] = stat
            rw_pvalue[j, 2] = pval

        # ---- Step 5: RW-MTCCT ----
        cluster_pval = RW_MTCCT(
            rw_pvalue[:, 2],
            idx_dict["width_x"],
            idx_dict["width_y"],
            threshold=threshold,
        )

        result["rw_pvalue"] = rw_pvalue
        result["num_region_r"] = idx_dict["width_x"]
        result["num_region_c"] = idx_dict["width_y"]
        result["cluster_pvalue"] = cluster_pval

    except Exception as e:
        result["status"] = f"error: {e}"
        result["cluster_pvalue"] = 1.0

    return result


# ===========================================================================
#  Parallel simulation runner
# ===========================================================================

def run_simulation_parallel(settings=None, n_rep=100, method="gamma",
                            n_perm=99, output_root="./data/result",
                            n_workers=None, rep_workers=None,
                            n=200, M1=150, M2=100,
                            window_width=20, step_divisor=4,
                            n_quantile=21, kde_bw=None):
    """
    Run RWSPM simulation with multi-core parallelization.

    并行策略：
    - LQD 区域级：每个子区域的 KDE+LQD 计算分配到不同进程
    - BCov 区域级：每个子区域的置换检验分配到不同进程
    - 重复模拟级：不同 Setting 分配到不同进程同时运行

    Parameters
    ----------
    settings : list of int or None
        Setting numbers to run (default: 1-7).
    n_rep : int
        Replications per setting (default 100).
    method : str
        "gamma", "limit", or "permu".
    n_perm : int
        Permutations per region (default 99).
    output_root : str
        Output directory root.
    n_workers : int or None
        Workers for region-level parallelism.
    rep_workers : int or None
        Workers for replication-level parallelism.

    --- Sensitivity analysis parameters ---
    n : int
        Sample size (default 200).
    M1, M2 : int
        Image rows / columns (default 150, 100).
    window_width : int or None
        Sliding window width. None → auto JSD golden-section search.
    step_divisor : int
        Step size = ceil(window_width / step_divisor).  Default 4.
    n_quantile : int
        LQD interpolation points L (default 21).
    kde_bw : float or None
        Fixed KDE bandwidth. None → auto bw.nrd0 per sample.
    """
    if settings is None:
        settings = list(range(1, 8))

    cpu_count = multiprocessing.cpu_count()
    if n_workers is None:
        n_workers = max(1, cpu_count - 1)
    if rep_workers is None:
        rep_workers = 1

    # --- Nested-pool safety ---
    # Linux (fork): 子进程可安全创建嵌套进程池（共享内存页，无序列化开销）
    #               → rep_workers>1 时仍允许区域级并行（双层并行）
    # Windows (spawn): 不支持嵌套池 → rep_workers>1 时关闭区域级并行
    # OLD logic (always blocks on rep_workers>1):
    #   use_region_parallel = (rep_workers <= 1)
    _nested_pool_safe = (multiprocessing.get_start_method() == 'fork')
    use_region_parallel = (rep_workers <= 1 or _nested_pool_safe)
    effective_n_workers = n_workers if use_region_parallel else 1

    print(f"System: {cpu_count} cores detected")
    if use_region_parallel:
        print(f"  Mode: region-level parallel ({effective_n_workers} workers)")
    else:
        print(f"  Mode: rep-level parallel ({rep_workers} workers, regions serial)")
    print(f"  Params: n={n}, M1={M1}, M2={M2}, "
          f"window_width={'auto' if window_width is None or window_width <= 0 else window_width}, "
          f"step_divisor={step_divisor}, n_quantile={n_quantile}, "
          f"kde_bw={'auto' if kde_bw is None else kde_bw}")
    print()

    total_start = time.time()

    if rep_workers > 1 and len(settings) > 1:
        # === Replication-level parallel: run all reps concurrently ===
        # (regions run SERIALLY inside each child to avoid nested pools)
        rep_args = []
        for k in settings:
            save_root = os.path.join(output_root, f"Setting{k}")
            os.makedirs(save_root, exist_ok=True)
            for l in range(1, n_rep + 1):
                rep_args.append((
                    k, l, M1, M2, n, window_width, step_divisor,
                    n_quantile, kde_bw, method, n_perm, THRESHOLD,
                    effective_n_workers,
                    False,  # use_region_parallel = False in child processes
                ))

        print(f"Submitting {len(rep_args)} replications "
              f"across {rep_workers} workers...")
        all_cluster = {k: [] for k in settings}

        with ProcessPoolExecutor(max_workers=rep_workers) as pool:
            futures = {
                pool.submit(_run_one_rep, args): (args[0], args[1])
                for args in rep_args
            }
            done_count = 0
            for future in as_completed(futures):
                setting, rep_id = futures[future]
                res = future.result()
                all_cluster[setting].append(res["cluster_pvalue"])
                done_count += 1

                # Save per-region CSV immediately
                if res["status"] == "ok":
                    save_root = os.path.join(output_root, f"Setting{setting}")
                    csv_path = os.path.join(save_root, f"test_rwspm{rep_id}.csv")
                    header = f"region.j,{method},pvalue"
                    np.savetxt(csv_path, res["rw_pvalue"], delimiter=",",
                               header=header, comments="", fmt="%.10f")

                if done_count % 10 == 0 or done_count == len(rep_args):
                    elapsed = time.time() - total_start
                    print(f"  [{done_count}/{len(rep_args)}] "
                          f"rep setting={setting} id={rep_id} "
                          f"p={res['cluster_pvalue']:.4e} "
                          f"({elapsed:.0f}s)")

        # Save cluster-level CSVs
        for k in settings:
            cluster_csv = os.path.join(output_root, f"Setting{k}",
                                       f"rwspm_cluster_set{k}.csv")
            np.savetxt(cluster_csv, np.array(all_cluster[k]), delimiter=",",
                       header="rw.mtcct", comments="", fmt="%.10f")
            print(f"  Setting {k}: {len(all_cluster[k])} reps saved -> "
                  f"{cluster_csv}")

    else:
        # === Sequential settings, each with region-level parallelism ===
        for k in settings:
            print(f"\n{'='*60}")
            print(f"Setting {k}  ({n_rep} replications, method={method}, "
                  f"region_workers={n_workers})")
            print(f"{'='*60}")

            save_root = os.path.join(output_root, f"Setting{k}")
            os.makedirs(save_root, exist_ok=True)

            # Prepare all replication arguments
            rep_args = [
                (k, l, M1, M2, n, window_width, step_divisor,
                 n_quantile, kde_bw, method, n_perm, THRESHOLD,
                 effective_n_workers,
                 True)   # use_region_parallel = True in parent process
                for l in range(1, n_rep + 1)
            ]
            rw_mtcct = np.zeros(n_rep)

            setting_start = time.time()

            # Run reps with region-level parallelism inside each
            # (they still run sequentially since rep_workers=1)
            for l_idx, args in enumerate(rep_args):
                rep_start = time.time()
                res = _run_one_rep(args)
                l = l_idx + 1

                if res["status"] == "ok":
                    # Save per-region CSV
                    csv_path = os.path.join(save_root, f"test_rwspm{l}.csv")
                    header = f"region.j,{method},pvalue"
                    np.savetxt(csv_path, res["rw_pvalue"], delimiter=",",
                               header=header, comments="", fmt="%.10f")

                    rw_mtcct[l - 1] = res["cluster_pvalue"]
                else:
                    rw_mtcct[l - 1] = 1.0

                rep_elapsed = time.time() - rep_start
                print(f"  Rep {l}/{n_rep}  p={rw_mtcct[l-1]:.6e}  "
                      f"({rep_elapsed:.1f}s)  [{res['status']}]")

            # Save cluster-level CSV
            cluster_csv = os.path.join(save_root, f"rwspm_cluster_set{k}.csv")
            np.savetxt(cluster_csv, rw_mtcct, delimiter=",",
                       header="rw.mtcct", comments="", fmt="%.10f")

            setting_elapsed = time.time() - setting_start
            print(f"  [Done] Setting {k}: {n_rep} reps in "
                  f"{setting_elapsed:.1f}s -> {cluster_csv}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All done. Total time: {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(
        description="RWSPM parallel simulation script"
    )
    parser.add_argument(
        "--settings", nargs="+", type=int, default=None,
        help="Setting numbers to run (1-7). Default: all."
    )
    parser.add_argument(
        "--n_rep", type=int, default=100,
        help="Number of replications per setting (default 100)."
    )
    parser.add_argument(
        "--method", type=str, default="gamma",
        choices=["gamma", "limit", "permu"],
        help="P-value method (default gamma)."
    )
    parser.add_argument(
        "--n_perm", type=int, default=99,
        help="Permutations per region (default 99)."
    )
    parser.add_argument(
        "--output_root", type=str, default="./data/result",
        help="Output directory root (default ./data/result)."
    )
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help=f"Region-level workers (default cpu_count-1 = {max(1, cpu_count-1)})."
    )
    parser.add_argument(
        "--rep_workers", type=int, default=None,
        help="Rep-level workers (default 1 = sequential across reps). "
             "Set >1 to run reps/settings concurrently."
    )

    # --- Sensitivity analysis parameters ---
    parser.add_argument(
        "--n", type=int, default=50, dest="n",
        help="Sample size (default 50)."
    )
    parser.add_argument(
        "--M1", type=int, default=150,
        help="Image rows (default 150)."
    )
    parser.add_argument(
        "--M2", type=int, default=100,
        help="Image columns (default 100)."
    )
    parser.add_argument(
        "--window_width", type=int, default=20,
        help="Sliding window width.  0 or negative → auto JSD search (default 20)."
    )
    parser.add_argument(
        "--step_divisor", type=int, default=4,
        help="Step = ceil(window_width / step_divisor)  (default 4)."
    )
    parser.add_argument(
        "--n_quantile", type=int, default=21,
        help="LQD quantile interpolation points L (default 21)."
    )
    parser.add_argument(
        "--kde_bw", type=float, default=None,
        help="Fixed KDE bandwidth.  Omit for auto bw.nrd0 (default)."
    )
    args = parser.parse_args()

    run_simulation_parallel(
        settings=args.settings,
        n_rep=args.n_rep,
        method=args.method,
        n_perm=args.n_perm,
        output_root=args.output_root,
        n_workers=args.n_workers,
        rep_workers=args.rep_workers,
        n=args.n,
        M1=args.M1,
        M2=args.M2,
        window_width=args.window_width,
        step_divisor=args.step_divisor,
        n_quantile=args.n_quantile,
        kde_bw=args.kde_bw,
    )
