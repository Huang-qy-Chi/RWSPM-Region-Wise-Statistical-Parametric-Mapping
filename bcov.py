"""
Empirical Ball Covariance BCov^2_n(X, Y)

Optimized with vectorized inner loops for speed.
"""

import os
import numpy as np
from scipy import stats
import ctypes
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ============================================================
# C extension loading (cball_ext.dll / cball_ext.so)
# ============================================================
_CBALL: "ctypes.CDLL | None" = None
_CBALL_LOADED = False
_CBALL_ERROR: "str | None" = None


def _load_cball():
    global _CBALL, _CBALL_LOADED, _CBALL_ERROR
    if _CBALL_LOADED:
        return _CBALL is not None

    ext_dir = Path(__file__).parent / "cball_ext"
    candidates = (["cball_ext.so", "cball_ext.dll"] if os.name != "nt"
                  else ["cball_ext.dll", "cball_ext.so"])
    lib_path = None
    for name in candidates:
        if (ext_dir / name).exists():
            lib_path = str(ext_dir / name)
            break
    if lib_path is None:
        _CBALL_LOADED = True
        _CBALL_ERROR = (
            f"no compiled extension found in {ext_dir} "
            f"(looked for: {', '.join(candidates)}). Compile it first:\n"
            "  Linux:   cd cball_ext && bash compile_linux.sh   (needs gcc)\n"
            "  Windows: cball_ext\\compile_win.bat"
        )
        return False

    try:
        # Add MinGW runtime paths so DLL dependencies (libgomp, etc.) are found
        _mingw_paths = [
            r"C:\msys64\ucrt64\bin",
        ]
        for _p in _mingw_paths:
            if _p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")
        lib = ctypes.CDLL(lib_path)
        lib.compute_bdd_bias.argtypes = [
            np.ctypeslib.ndpointer(np.float64),
            np.ctypeslib.ndpointer(np.float64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        _CBALL = lib
        _CBALL_LOADED = True
        _CBALL_ERROR = None
        return True
    except Exception as e:
        _CBALL_LOADED = True
        _CBALL_ERROR = (
            f"found {lib_path} but ctypes failed to load it: {e!r}\n"
            "Common causes on Linux:\n"
            "  - compiled with -fopenmp but libgomp is not installed "
            "(recompile WITHOUT -fopenmp)\n"
            "  - .so was built for a different OS/arch (recompile on this machine)\n"
            "  - missing runtime dependencies (check with: ldd cball_ext.so)"
        )
        return False


# ============================================================
# Distance helpers
# ============================================================

# 因变量 Y 的距离 DY 可用的范数选项：'1' / '2' / 'inf'
# （'2' 为默认，与原始 np.linalg.norm 默认的二范数行为一致）
_NORM_ORD = {"1": 1, "2": 2, "inf": np.inf}


def _pairwise_distance(A, norm="2"):
    """
    Pairwise distance matrix of the rows of ``A`` under the given norm.

    Parameters
    ----------
    A : ndarray, shape (n, q)
    norm : str
        ``'1'``   — L1 (Manhattan) distance.
        ``'2'``   — L2 (Euclidean) distance (default).
        ``'inf'`` — L-infinity (Chebyshev / max-abs) distance.

    Returns
    -------
    D : ndarray, shape (n, n)
        Symmetric pairwise distance matrix (diagonal = 0).
    """
    ord = _NORM_ORD.get(norm, 2)
    return np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1, ord=ord)


# ============================================================
# HBE (Hall-Buckley-Eagleson) Gamma approximation
# ============================================================


def _center_bdd_matrix(K):
    """Double-centre a kernel matrix."""
    return K - K.mean(axis=1, keepdims=True) - K.mean(axis=0, keepdims=True) + K.mean()


def _hbe(coeff, x):
    """Hall-Buckley-Eagleson: p-value via 3-cumulant Gamma approximation."""
    K1 = np.sum(coeff)
    K2 = 2 * np.sum(coeff ** 2)
    K3 = 8 * np.sum(coeff ** 3)
    nu = 8 * K2 ** 3 / K3 ** 2
    x_trans = np.sqrt(2 * nu / K2) * (x - K1) + nu
    # Avoid cancellation in the extreme upper tail.
    return stats.gamma.sf(x_trans, a=nu / 2, scale=2)


def _bdd_matrix_bias_c(D, weight='constant'):
    """Compute BDD kernel via C extension (cball_ext)."""
    if not _load_cball():
        raise RuntimeError("cball_ext unavailable: " + (_CBALL_ERROR or "unknown reason"))
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    dist_vec = D[triu_idx].astype(np.float64, copy=True)
    out_size = n * (n + 1) // 2
    out = np.zeros(out_size, dtype=np.float64)
    weight_map = {'constant': 1, 'probability': 2, 'chisquare': 3, 'rbf': 4}
    wt = ctypes.c_int(weight_map.get(weight, 1))
    n_val = ctypes.c_int(n)
    nth = ctypes.c_int(4 if n > 500 else 1)
    assert _CBALL is not None  # guaranteed by _load_cball() call above
    _CBALL.compute_bdd_bias(out, dist_vec,
                            ctypes.byref(n_val),
                            ctypes.byref(nth),
                            ctypes.byref(wt))
    # The C routine packs entries in row-wise upper-triangle order:
    # (0,0), (0,1), ..., (0,n-1), (1,1), ... .  ``tril_indices`` permutes
    # these entries and corrupts the eigenvalues used by method='limit'.
    rows, cols = np.triu_indices(n)
    K = np.zeros((n, n))
    K[rows, cols] = out
    K = K + K.T
    np.fill_diagonal(K, np.diag(K) / 2)
    return K


def diagnose_cball():
    """Print why the C extension is (un)available on the current machine."""
    global _CBALL_LOADED
    _CBALL_LOADED = False
    ok = _load_cball()
    print("cball loaded:", ok)
    print("reason:", _CBALL_ERROR)
    ext_dir = Path(__file__).parent / "cball_ext"
    print("ext_dir:", ext_dir)
    print("ext_dir exists:", ext_dir.exists())
    if ext_dir.exists():
        for p in sorted(ext_dir.iterdir()):
            print("   ", p.name, p.stat().st_size if p.is_file() else "<dir>")
    return ok


def _bcov_limit_pvalue(DX, DY, weight='constant'):
    """Compute p-value via limit distribution (HBE approximation)."""
    Kx = _bdd_matrix_bias_c(DX, weight)
    Ky = _bdd_matrix_bias_c(DY, weight)
    Kx_c = _center_bdd_matrix(Kx)
    Ky_c = _center_bdd_matrix(Ky)
    K_joint = Kx_c * Ky_c
    n = DX.shape[0]
    eigenvalues = np.linalg.eigh(K_joint)[0]
    eigenvalues = eigenvalues[eigenvalues > 0] / n
    bcov_stat = _bcov_from_dm(DX, DY)
    return _hbe(eigenvalues, n * bcov_stat)


def _bcov_from_dm(DX, DY):
    """
    Compute BCov^2 from pre-computed squared Euclidean distance matrices.

    Parameters
    ----------
    DX : ndarray, shape (n, n)
        Pairwise Euclidean distances for X.
    DY : ndarray, shape (n, n)
        Pairwise Euclidean distances for Y.

    Returns
    -------
    bcov2 : float
    """
    n = DX.shape[0]
    diff = np.zeros((n, n))

    for i in range(n):
        di = DX[i, :]                     # (n,)
        # AX[j, k] = di[k] <= di[j]
        AX = di[None, :] <= di[:, None]   # (n, n)

        di_y = DY[i, :]
        AY = di_y[None, :] <= di_y[:, None]

        PX = np.mean(AX, axis=1)          # (n,)
        PY = np.mean(AY, axis=1)
        PXY = np.mean(AX & AY, axis=1)

        diff[i, :] = PXY - PX * PY

    return np.mean(diff ** 2)


def bcov(X, Y, norm="2"):
    """
    Compute the empirical Ball Covariance statistic BCov^2_n(X, Y).

    Parameters
    ----------
    X : ndarray, shape (n, p)
    Y : ndarray, shape (n, q)
    norm : str
        Norm used for the Y distance matrix DY: ``'1'``, ``'2'`` (default)
        or ``'inf'``.  X always uses the Euclidean (L2) distance.

    Returns
    -------
    bcov2 : float
        Empirical squared Ball Covariance.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]

    # Distance matrices: X always L2; Y uses the selected norm
    DX = _pairwise_distance(X, "2")
    DY = _pairwise_distance(Y, norm)

    return _bcov_from_dm(DX, DY)


def bcov_perm_test(x, y, n_perm=199, method='gamma', DY=None, norm='2'):
    """
    Ball Covariance permutation test.

    Parameters
    ----------
    x : ndarray, shape (n, p)
    y : ndarray, shape (n, q)
    n_perm : int
        Number of permutations (default 199).
    method : str
        ``'permu'`` — permutation p-value (default, matches original behaviour).
        ``'gamma'`` — Gamma approximation p-value using moment-matched Gamma
        distribution fitted to the permutation null distribution.
    DY : ndarray, shape (n, n) or None
        Distance matrix of Y; can insert to avoid repeated calculation.
        If None it is computed here with the selected ``norm``.
    norm : str
        Norm used for the Y distance matrix DY: ``'1'``, ``'2'`` (default)
        or ``'inf'``.  X always uses the Euclidean (L2) distance.
    Returns
    -------
    stat : float
        Observed BCov^2 statistic.
    pval : float
        Permutation p-value (method='permu') or Gamma-approximation p-value
        (method='gamma').
    cancel: gamma_fit : dict, optional (only when method='gamma')
        Gamma parameters {'shape', 'scale', 'mean', 'variance'}.
    """
    n = x.shape[0]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Precompute DY (stays the same for permutations)
    if DY is None:
        DY = _pairwise_distance(y, norm)
    # Precompute full DX
    DX = _pairwise_distance(x, "2")

    stat = _bcov_from_dm(DX, DY)
    perm_stats = np.zeros(n_perm)

    for b in range(n_perm):
        perm = np.random.permutation(n)
        DX_perm = DX[np.ix_(perm, perm)]
        perm_stats[b] = _bcov_from_dm(DX_perm, DY)

    if method == 'limit':
        pval = _bcov_limit_pvalue(DX, DY, weight='constant')
        return stat, pval
    elif method == 'gamma':
        from gammafit import gamma_moment_match
        gamma_fit = gamma_moment_match(perm_stats)
        alpha = gamma_fit['shape']
        theta = gamma_fit['scale']
        pval = 1.0 - stats.gamma.cdf(stat, a=alpha, scale=theta)
        return stat, pval
    else:
        pval = (np.sum(perm_stats >= stat) + 1.0) / (n_perm + 1.0)
        return stat, pval


def bcov_perm_test_gwas(x, DY, n_perm=199, method='gamma'):
    """
    Ball Covariance permutation test.

    Parameters
    ----------
    x : ndarray, shape (n, p)
    n_perm : int
        Number of permutations (default 199).
    method : str
        ``'permu'`` — permutation p-value (default, matches original behaviour).
        ``'gamma'`` — Gamma approximation p-value using moment-matched Gamma
        distribution fitted to the permutation null distribution.
    DY: Distance of Y, can insert to avoid repeated calculation
    Returns
    -------
    stat : float
        Observed BCov^2 statistic.
    pval : float
        Permutation p-value (method='permu') or Gamma-approximation p-value
        (method='gamma').
    cancel: gamma_fit : dict, optional (only when method='gamma')
        Gamma parameters {'shape', 'scale', 'mean', 'variance'}.
    """
    n = x.shape[0]
    x = np.asarray(x, dtype=float)
    # y = np.asarray(y, dtype=float)

    # Precompute DY (stays the same for permutations)
    # if DY is None:
        # DY = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    # else: 
        # DY = DY
    # Precompute full DX
    DX = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)

    stat = _bcov_from_dm(DX, DY)
    perm_stats = np.zeros(n_perm)

    for b in range(n_perm):
        perm = np.random.permutation(n)
        DX_perm = DX[np.ix_(perm, perm)]
        perm_stats[b] = _bcov_from_dm(DX_perm, DY)

    if method == 'limit':
        pval = _bcov_limit_pvalue(DX, DY, weight='constant')
        return stat, pval
    elif method == 'gamma':
        from gammafit import gamma_moment_match
        gamma_fit = gamma_moment_match(perm_stats)
        alpha = gamma_fit['shape']
        theta = gamma_fit['scale']
        pval = 1.0 - stats.gamma.cdf(stat, a=alpha, scale=theta)
        return stat, pval
    else:
        pval = (np.sum(perm_stats >= stat) + 1.0) / (n_perm + 1.0)
        return stat, pval


if __name__ == "__main__":
    from seed import set_seed
    import time 
    set_seed(123)
    n = 1000
    B = 100
    X_test = np.random.randn(n, 2)
    Y_test = np.random.randn(n, 2)
    print(f"bcov (n=10): {bcov(X_test, Y_test):.6f}")
    print(f"n * bcov:    {n * bcov(X_test, Y_test):.6f}")
    t0 = time.time()
    stat, pval = bcov_perm_test(X_test, Y_test, n_perm=200, method='gamma')
    elapsed = time.time() - t0
    print(f"perm test: stat={stat:.6f}, pval={pval:.4f}, time={elapsed:.4f}s") #method=gamma, 52.5532s
    # t1 = time.time()
    # stat, pval = bcov_perm_test(X_test, Y_test, n_perm=200,method='permu')
    # elapsed1 = time.time() - t1
    # print(f"perm test: stat={stat:.6f}, pval={pval:.4f}, time={elapsed1:.4f}s") #method=permu, 314.0065s
    t2 = time.time()
    stat_l, pval_l = bcov_perm_test(X_test, Y_test, method='limit')
    elapsed2 = time.time() - t2
    print(f"limit test: stat={stat_l:.6f}, pval={pval_l:.4f}, time={elapsed2:.4f}s") #method=limit, 592.7041s


