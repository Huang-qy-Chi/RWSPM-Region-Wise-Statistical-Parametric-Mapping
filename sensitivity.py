"""
sensitivity.py — RWSPM 敏感性分析脚本 (Sensitivity analysis driver)

在 run_simu_parallel.py 的并行框架之上，对以下参数做**单变量敏感性分析**
（每次只改变一个参数，其余固定在 baseline 值）：

    step_divisor : 滑窗步长除数，步长 b = ceil(window_width / step_divisor)
                   （默认 4；取 2/4/8 时步长分别为 ceil(w/2)、ceil(w/4)、ceil(w/8)）
    n            : 样本量
    n_quantile   : LQD 分位插值点数 L（对应 run_simu_parallel 的 --n_quantile）
    window_width : 移动窗宽 \\tilde h（滑窗边长，像素）
    kde_bw       : KDE 带宽 h（None = 自动 bw.nrd0）

对每个 (axis, value, setting) 组合运行 n_rep 轮重复。与 run_simu_parallel.py
不同，本脚本**只收集最终的 RW-MTCCT 聚类 p 值**（不保存分 region 的 p 值），
并按 "p < significance level"（默认 0.05）计算拒绝率：

    Setting 1      -> type I error（零假设 X ⊥ Y 下的拒绝率，应接近 α）
    Setting 2 - 7  -> power（备择假设下的拒绝率）

每个 (setting, rep) 的随机种子固定为 random_state = 2026 + rep * 7（与
run_simu_parallel.py 一致），因此同一 rep 在不同参数配置下使用相同数据，
配置之间具有可比性。

输出（默认 data/result/ 下）：
    sensitivity.csv           汇总：每 (setting, axis, param_value) 的
                              type I / power、平均 p 值、比例标准误
    sensitivity_pvalues.csv   原始最终聚类 p 值（每 rep 一行）

Usage
-----
    python sensitivity.py                                 # 5 个轴全部默认取值
    python sensitivity.py --step 2,4,8                     # 只分析 step（2/4/8）
    python sensitivity.py --n 100,200 --L 11,21,41         # 只分析 n 与 L
    python sensitivity.py --n_rep 30 --settings 1 3 5      # 快速测试/指定 Setting
    python sensitivity.py --threshold 0.05                 # 显著性水平 α
    python sensitivity.py --output_root data/result/sens   # 输出目录
    python sensitivity.py --rep_workers 4                  # 重复级并行 worker 数
    python sensitivity.py --no_pvalues                     # 不保存原始 p 值
"""

import os
import sys
import csv
import time
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ---- 复用 run_simu_parallel.py 的单轮 worker（含数据生成/分区/LQD/BCov/RW-MTCCT）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_simu_parallel import _run_one_rep  # noqa: E402


# ===========================================================================
#  默认参数
# ===========================================================================
BASELINE = {
    "step_divisor": 4,     # 步长 b = ceil(window_width / step_divisor)
    "n": 200,              # 样本量
    "n_quantile": 21,      # LQD 分位插值点数 L
    "window_width": 20,    # 移动窗宽 \tilde h
    "kde_bw": None,        # KDE 带宽 h（None = 自动 bw.nrd0）
}

AXIS_DEFAULTS: dict[str, list[float | None]] = {
    "step_divisor": [2, 4, 8],
    "n": [50, 100, 200, 400],
    "n_quantile": [11, 21, 41],
    "window_width": [10, 20, 30, 40],
    "kde_bw": [None, 0.5, 1.0, 2.0],
}


# ===========================================================================
#  单配置运行器：对一个 (setting, 参数配置) 跑 n_rep 轮，只返回最终聚类 p 值
# ===========================================================================

def _run_one_config(setting, n_rep, method, n_perm, threshold, M1, M2,
                    n_workers, rep_workers, width_search,
                    step_divisor, n, n_quantile, window_width, kde_bw):
    """
    Run ``n_rep`` replications for one (setting, params) configuration.

    Mirrors run_simu_parallel.py's parallel logic:
      - rep_workers > 1 -> replications run in a process pool; on Windows
        (spawn) region-level parallelism inside the children is disabled
        (nested pools not allowed);
      - otherwise -> replications run sequentially in the parent with
        region-level parallelism enabled.

    Parameters
    ----------
    setting : int
        Simulation setting 1-7.
    n_rep : int
        Number of replications.
    method, n_perm, threshold, M1, M2, n_workers, rep_workers, width_search
        See run_simu_parallel.py.
    step_divisor, n, n_quantile, window_width, kde_bw
        The configuration under test (one of them is the sensitivity axis).

    Returns
    -------
    pvals : list of float
        The ``n_rep`` cluster-level p-values (RW-MTCCT), one per successful
        replication.
    """
    _nested_pool_safe = (multiprocessing.get_start_method() == "fork")
    use_region_parallel = (rep_workers <= 1 or _nested_pool_safe)
    effective_n_workers = n_workers if use_region_parallel else 1

    rep_args = [
        (setting, l, M1, M2, n, window_width, step_divisor,
         n_quantile, kde_bw, method, n_perm, threshold,
         effective_n_workers, use_region_parallel, width_search)
        for l in range(1, n_rep + 1)
    ]

    pvals = []
    if rep_workers > 1:
        with ProcessPoolExecutor(max_workers=rep_workers) as pool:
            futures = {pool.submit(_run_one_rep, args): args[1]
                       for args in rep_args}
            for fut in as_completed(futures):
                res = fut.result()
                if res["status"] == "ok":
                    pvals.append(res["cluster_pvalue"])
                else:
                    print(f"    [WARN] rep failed: {res['status']}")
    else:
        for args in rep_args:
            res = _run_one_rep(args)
            if res["status"] == "ok":
                pvals.append(res["cluster_pvalue"])
            else:
                print(f"    [WARN] rep failed: {res['status']}")

    return pvals


# ===========================================================================
#  主入口：逐轴遍历参数取值，计算并存储 type I / power
# ===========================================================================

def run_sensitivity(settings=None, n_rep=50, method="gamma", n_perm=99,
                    threshold=0.05, M1=150, M2=100,
                    n_workers=None, rep_workers=None,
                    width_search="single",
                    axes=None, baseline=None,
                    output_root="./data/result", save_pvalues=True):
    """
    Run the single-variable sensitivity analysis.

    Parameters
    ----------
    settings : list of int or None
        Settings to run (default 1-7).  Setting 1 -> type I error,
        Settings 2-7 -> power.
    n_rep : int
        Replications per (axis, value, setting) (default 50).
    method : str
        P-value method: "gamma", "limit" or "permu" (default "gamma").
    n_perm : int
        Permutations per region (default 99).
    threshold : float
        Significance level alpha (default 0.05).
    M1, M2 : int
        Image rows / columns (default 150, 100).
    n_workers, rep_workers : int or None
        Region-level / replication-level workers (same semantics as
        run_simu_parallel.py).
    width_search : str
        "single" (default; widths are fixed here) or "parallel".
    axes : dict or None
        ``{param_name: [values, ...]}``.  None -> use AXIS_DEFAULTS.
    baseline : dict or None
        Values of the non-varying parameters.  None -> BASELINE.
    output_root : str
        Output directory (default "./data/result").
    save_pvalues : bool
        Also save the raw cluster p-values (default True).

    Returns
    -------
    summary_path : str
        Path of the written sensitivity.csv summary.
    """
    if settings is None:
        settings = list(range(1, 8))
    if baseline is None:
        baseline = dict(BASELINE)
    if axes is None:
        axes = {k: list(v) for k, v in AXIS_DEFAULTS.items()}

    cpu_count = multiprocessing.cpu_count()
    if n_workers is None:
        n_workers = max(1, cpu_count - 1)
    if rep_workers is None:
        rep_workers = 1

    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, "sensitivity.csv")
    pval_path = os.path.join(output_root, "sensitivity_pvalues.csv")

    summary_rows = []
    total_start = time.time()
    n_configs = sum(len(v) for v in axes.values()) * len(settings)
    done = 0

    print(f"System: {cpu_count} cores, rep_workers={rep_workers}, "
          f"n_workers={n_workers}, n_rep={n_rep}, alpha={threshold}, "
          f"method={method}, n_perm={n_perm}")
    print(f"Settings: {settings}  (1 = type I, 2-7 = power)")
    print(f"Baseline: {baseline}")
    for param, values in axes.items():
        print(f"  axis {param}: {values}")

    # 原始 p 值文件（仅最终聚类 p 值，不保存分 region 的 p 值）
    pval_fh = None
    writer_p = None
    if save_pvalues:
        pval_fh = open(pval_path, "w", newline="")
        writer_p = csv.writer(pval_fh)
        writer_p.writerow(["setting", "axis", "param_value",
                           "rep_id", "cluster_pvalue"])

    try:
        for param, values in axes.items():
            for value in values:
                cfg = dict(baseline)
                cfg[param] = value
                for k in settings:
                    pvals = _run_one_config(
                        setting=k, n_rep=n_rep, method=method,
                        n_perm=n_perm, threshold=threshold,
                        M1=M1, M2=M2,
                        n_workers=n_workers, rep_workers=rep_workers,
                        width_search=width_search, **cfg,
                    )
                    done += 1

                    pv = np.asarray(pvals, dtype=float)
                    n_ok = pv.size
                    if n_ok == 0:
                        reject = mean_p = se = float("nan")
                    else:
                        reject = float(np.mean(pv < threshold))
                        mean_p = float(np.mean(pv))
                        se = (float(np.sqrt(reject * (1 - reject) / n_ok))
                              if n_ok > 1 else float("nan"))

                    metric = "type1" if k == 1 else "power"
                    pv_disp = "auto" if value is None else value
                    summary_rows.append([
                        k, metric, param, pv_disp, n_ok,
                        round(reject, 6), round(mean_p, 8), round(se, 6),
                    ])
                    if writer_p is not None:
                        for rep_id, p in enumerate(pvals, start=1):
                            writer_p.writerow(
                                [k, param, pv_disp, rep_id, p])

                    elapsed = time.time() - total_start
                    print(f"[{done}/{n_configs}] setting={k} {param}={pv_disp} "
                          f"{metric}={reject:.4f} (n_ok={n_ok}, "
                          f"mean_p={mean_p:.4g}, {elapsed:.0f}s)")
    finally:
        if pval_fh is not None:
            pval_fh.close()

    with open(summary_path, "w", newline="") as sf:
        w = csv.writer(sf)
        w.writerow(["setting", "metric", "axis", "param_value", "n_rep",
                    "reject_rate", "mean_pvalue", "se"])
        w.writerows(summary_rows)

    print(f"\nDone. Summary -> {summary_path}")
    if save_pvalues:
        print(f"Raw cluster p-values -> {pval_path}")
    return summary_path


# ===========================================================================
#  Entry point
# ===========================================================================

def _parse_axis(s, param):
    """Parse a comma-separated axis value list from the CLI."""
    if s is None:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.lower() in ("auto", "none", ""):
            out.append(None)
        elif param == "kde_bw":
            out.append(float(tok))
        else:
            out.append(int(float(tok)))
    return out


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(
        description="RWSPM sensitivity analysis script (type I error & power)"
    )
    parser.add_argument("--settings", nargs="+", type=int, default=None,
                        help="Settings to run (default 1-7).")
    parser.add_argument("--n_rep", type=int, default=50,
                        help="Replications per configuration (default 50).")
    parser.add_argument("--method", type=str, default="gamma",
                        choices=["gamma", "limit", "permu"],
                        help="P-value method (default gamma).")
    parser.add_argument("--n_perm", type=int, default=99,
                        help="Permutations per region (default 99).")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Significance level alpha (default 0.05).")
    parser.add_argument("--M1", type=int, default=150, help="Image rows.")
    parser.add_argument("--M2", type=int, default=100, help="Image columns.")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Region-level workers (default cpu_count-1).")
    parser.add_argument("--rep_workers", type=int, default=None,
                        help="Rep-level workers (default 1).")
    parser.add_argument("--width_search", type=str, default="single",
                        choices=["parallel", "single"],
                        help="JSD width-search mode (only relevant if a "
                             "window_width value <= 0 is given).")
    parser.add_argument("--output_root", type=str, default="./data/result",
                        help="Output directory (default ./data/result).")
    parser.add_argument("--no_pvalues", action="store_true",
                        help="Do not save the raw cluster p-values.")

    # ---- 敏感性轴（传了哪个轴就跑哪个；都不传则 5 个轴全用默认取值）----
    parser.add_argument("--step", default=None,
                        help="step_divisor values, e.g. 2,4,8")
    parser.add_argument("--n", dest="n_axis", default=None,
                        help="sample size n values, e.g. 50,100,200,400")
    parser.add_argument("--L", dest="L_axis", default=None,
                        help="LQD points L values, e.g. 11,21,41")
    parser.add_argument("--window", dest="window_axis", default=None,
                        help="window width h_tilde values, e.g. 10,20,30,40")
    parser.add_argument("--kde_bw", dest="kde_bw_axis", default=None,
                        help="KDE bandwidth h values; 'auto' = None, "
                             "e.g. auto,0.5,1,2")
    args = parser.parse_args()

    axis_flags = {
        "step_divisor": args.step,
        "n": args.n_axis,
        "n_quantile": args.L_axis,
        "window_width": args.window_axis,
        "kde_bw": args.kde_bw_axis,
    }
    axes = {}
    for param, raw in axis_flags.items():
        vals = _parse_axis(raw, param)
        if vals is not None:
            axes[param] = vals
    if not axes:
        axes = {k: list(v) for k, v in AXIS_DEFAULTS.items()}

    run_sensitivity(
        settings=args.settings,
        n_rep=args.n_rep,
        method=args.method,
        n_perm=args.n_perm,
        threshold=args.threshold,
        M1=args.M1,
        M2=args.M2,
        n_workers=args.n_workers,
        rep_workers=args.rep_workers,
        width_search=args.width_search,
        axes=axes,
        output_root=args.output_root,
        save_pvalues=not args.no_pvalues,
    )
