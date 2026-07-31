"""
Log Quantile Density (LQD) transformation.

Reimplements the LQD function from the R package code_RWSPM/LQD.R.

Uses a fast numpy-based 1D KDE (no scipy overhead) for efficiency.
"""

import numpy as np


def _kde_1d(x, grid, bw):
    """
    Fast 1D Gaussian KDE evaluated at *grid* points.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Data points.
    grid : ndarray, shape (g,)
        Evaluation grid.
    bw : float
        Bandwidth.

    Returns
    -------
    dens : ndarray, shape (g,)
        Density values.
    """
    n = len(x)
    if n < 2 or bw <= 1e-15:
        return np.ones_like(grid) / max(len(grid), 1)
    scaled = (grid[:, None] - x[None, :]) / bw
    kernel = np.exp(-0.5 * scaled ** 2)
    dens = kernel.sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))
    return dens


def _estimate_bandwidth(y_slice):
    """
    R's bw.nrd0: 0.9 * min(sd, IQR/1.34) * n^(-1/5)
    """
    sd_val = np.std(y_slice, ddof=1)
    iqr_val = np.percentile(y_slice, 75) - np.percentile(y_slice, 25)
    bw_nrd = 0.9 * min(sd_val, iqr_val / 1.34) * len(y_slice) ** (-0.2)
    return bw_nrd


def LQD(y, idx, t=None):
    """
    Log Quantile Density (LQD) transformation.

    For each sub-region *j* and each sample *i*:

    1. Extract the pixel values in sub-region *j* for sample *i*.
    2. Estimate the probability density via KDE.
    3. Compute quantiles of pixel values at grid *t*.
    4. Interpolate the density at those quantile positions -> QD.
    5. Take the negative logarithm: G = -log(QD).

    Parameters
    ----------
    y : ndarray, shape (n, p)
        Data matrix: *n* samples, *p* variables (flattened image).
    idx : ndarray, shape (m, w*w)
        Index matrix — each row contains **0-based** column indices
        of pixels in the *j*-th sub-region.
    t : ndarray, optional
        Quantile sequence in [0, 1].  Default ``numpy.linspace(0, 1, 21)``.

    Returns
    -------
    G : list of ndarray
        Each element has shape ``(n, len(t))`` — LQD values per sub-region.
    """
    if t is None:
        t = np.linspace(0, 1, 21)

    y = np.asarray(y, dtype=float)
    m, n = idx.shape[0], y.shape[0]

    QD = []
    for j in range(m):
        qd = np.zeros((n, len(t)))
        col_idx = idx[j, :].astype(int)
        for i in range(n):
            y_slice = y[i, col_idx]

            # Bandwidth (R's bw.nrd0)
            bw = _estimate_bandwidth(y_slice)
            # Grid
            grid_min = y_slice.min() - 3.0 * bw
            grid_max = y_slice.max() + 3.0 * bw
            g = np.linspace(grid_min, grid_max, 256)
            # KDE
            dens = _kde_1d(y_slice, g, bw)
            dens[dens <= 0] = 1e-4

            # Quantiles and interpolation
            q = np.quantile(y_slice, t)
            qd[i, :] = np.interp(q, g, dens)

        QD.append(qd)

    G = [-np.log(qd + 1e-300) for qd in QD]
    return G
