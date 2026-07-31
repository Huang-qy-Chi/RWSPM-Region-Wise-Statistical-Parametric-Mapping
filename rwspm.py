"""
Regional Window Selection via Partition Modeling (RWSPM).

Reimplements the R functions from code_RWSPM/RWSPM.R and code_RWSPM/slide_width.R.
"""
#%%---------------------------------------------------------------------------------------------
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import cauchy
import networkx as nx

from lqd import LQD
from bcov import bcov

warnings.filterwarnings("ignore")


# ===========================================================================
#  Helper: Jensen-Shannon Divergence between two KDE estimates
# ===========================================================================

def jsd_from_kde_samples(dens_x, dens_y):
    """
    Compute the Jensen-Shannon divergence between two KDE density estimates.

    Parameters
    ----------
    dens_x : tuple (grid, values)
        KDE output: x-coordinates and y-values of the density.
    dens_y : tuple (grid, values)
        KDE output: x-coordinates and y-values of the density.

    Returns
    -------
    jsd : float
        Jensen-Shannon divergence.
    """
    # Use a common grid (the first density's grid)
    dx = dens_x[0][1] - dens_x[0][0]

    p = dens_x[1] / (np.sum(dens_x[1]) * dx)
    q = dens_y[1] / (np.sum(dens_y[1]) * dx)

    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    kl = lambda a, b: np.sum(a * np.log(a / b))
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return jsd


# ===========================================================================
#  Objective function for golden section search
# ===========================================================================

def _ov_region_jsd(sub_region, n_grid=256):
    """
    Mean pairwise Jensen-Shannon divergence within one sub-region.

    Top-level helper so that :func:`compute_ov` can parallelise its
    sub-region loop with a process pool.  Returns exactly the same value
    as the original sequential loop body.

    Parameters
    ----------
    sub_region : ndarray, shape (n_sam, width*width)
        Pixel values of one sub-region for all samples.
    n_grid : int, optional
        Number of grid points for KDE (default 256).

    Returns
    -------
    jsd_mean : float
        Mean of the pairwise JSD matrix of this sub-region.
    """
    n_sam = sub_region.shape[0]

    # Unified grid across all values in this sub-region
    all_vals = sub_region.ravel()
    grid = np.linspace(all_vals.min(), all_vals.max(), n_grid)

    # KDE for each sample
    prob_mat = np.zeros((n_sam, n_grid))
    for i in range(n_sam):
        kde = stats.gaussian_kde(sub_region[i, :])
        prob_mat[i, :] = kde.evaluate(grid)
        prob_mat[i, :] = prob_mat[i, :] / np.sum(prob_mat[i, :])

    # JSD matrix (vectorised via broadcasting)
    eps = 1e-10
    P = np.tile(prob_mat, (n_sam, 1, 1))            # (n_sam, n_sam, n_grid)
    Q = np.transpose(P, (1, 0, 2))
    M = 0.5 * (P + Q) + eps
    P = P + eps
    Q = Q + eps

    KL_PM = np.sum(P * np.log(P / M), axis=2)        # (n_sam, n_sam)
    KL_QM = np.sum(Q * np.log(Q / M), axis=2)

    jsd_matrix = 0.5 * (KL_PM + KL_QM)
    return np.mean(jsd_matrix)


def compute_ov(y, window_width, M1, M2, n_grid=256, n_workers=1):
    """
    Objective function: given a sliding window width *w*, compute mean(ov2).

    Used by :func:`golden_section_search` to find the optimal window width.

    Parameters
    ----------
    y : ndarray, shape (n_sam, M1 * M2)
        Image data.
    window_width : int
        Sliding window width (candidate).
    M1 : int
        Image height (number of rows).
    M2 : int
        Image width (number of columns).
    n_grid : int, optional
        Number of grid points for KDE (default 256).
    n_workers : int, optional
        Number of worker processes for the sub-region loop (default 1 =
        single-core, i.e. the original sequential behaviour).

    Returns
    -------
    ov_loss : float
        Objective value = mean(JSD) + 0.6 * penalty term.
    """
    n_sam = y.shape[0]
    window_width = int(np.ceil(window_width))
    w1 = int(np.ceil(window_width / 4))
    idx1 = slide_width_step(w1, window_width, M1, M2)
    sub_idx = idx1["idx"]
    m = sub_idx.shape[0]

    if n_workers is None or n_workers <= 1 or m <= 1:
        # --- Sequential path (original single-core behaviour) ---
        ov2 = np.zeros(m)
        for h in range(m):
            col_idx = sub_idx[h, :].astype(int)
            sub_region = y[:, col_idx]          # (n_sam, window_width)
            ov2[h] = _ov_region_jsd(sub_region, n_grid)
    else:
        # --- Parallel path: sub-regions spread over a process pool ---
        n_workers = int(min(n_workers, m))
        region_gen = (
            y[:, sub_idx[h, :].astype(int)]
            for h in range(m)
        )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            ov2 = np.array(list(pool.map(
                partial(_ov_region_jsd, n_grid=n_grid),
                region_gen,
            )))

    ov_mean = np.mean(ov2)
    penalty = 0.6 * (window_width - 2) / (np.ceil(min(M1, M2) / 2) - 2)
    ov_loss = ov_mean + penalty
    # print(f"width_{window_width}: {ov_mean:.4f}-{ov_loss:.4f}")
    return ov_loss


# ===========================================================================
#  Golden-section search
# ===========================================================================

def golden_section_search(y, f, a, b, M1, M2, tol=1):
    """
    Golden-section search to minimise *f* over the interval [a, b].

    Parameters
    ----------
    y : ndarray
        Data passed to the objective function *f*.
    f : callable
        Objective function ``f(y, width, M1, M2)``.
    a : float
        Left bound.
    b : float
        Right bound.
    M1 : int
        Image height.
    M2 : int
        Image width.
    tol : float, optional
        Convergence tolerance (default 1).

    Returns
    -------
    width_opt : int
        Ceiling of the optimal width.
    """
    phi = (1 + np.sqrt(5)) / 2
    resphi = 1 / phi

    x1 = b - (b - a) * resphi
    x2 = a + (b - a) * resphi

    f1 = f(y, x1, M1, M2)
    f2 = f(y, x2, M1, M2)

    while abs(x1 - x2) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) * resphi
            f1 = f(y, x1, M1, M2)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * resphi
            f2 = f(y, x2, M1, M2)

    return int(np.ceil((a + b) / 2))


# ===========================================================================
#  Slide / partition the image
# ===========================================================================

def slide_width_step(b, width, M1=150, M2=100):
    """
    Partition an image of size ``M1 x M2`` into overlapping square sub-regions
    using a sliding window.

    The window moves by *b* pixels (step) in both the row and column
    directions.  When the window exceeds the image boundary, it is shifted
    back so that the last window aligns with the edge.

    Parameters
    ----------
    b : int
        Step size (pixels) between consecutive windows.
    width : int
        Side length (pixels) of the square sliding window.
    M1 : int
        Image height (number of rows).  Default 150.
    M2 : int
        Image width (number of columns).  Default 100.

    Returns
    -------
    dict with keys:

        idx : ndarray, shape (num_regions, width*width)
            Each row contains the **0-based** linear indices of the pixels
            belonging to one sub-region.  The ordering matches R's
            ``as.vector(t(g))`` — i.e. row-major (x varies slowest).
        width_x : int
            Number of sub-regions along the row (x) direction.
        width_y : int
            Number of sub-regions along the column (y) direction.
    """
    idx_list = []
    x1 = 0               # 0-based start row
    width_x = 1
    final_x = False

    while True:
        y1 = 0           # 0-based start column
        while True:
            x2 = x1 + width - 1
            y2 = y1 + width - 1

            if y2 < M2:
                # Build mask in a M1 x M2 matrix, then flatten to match R's
                # as.vector(t(g)) ordering.
                g = np.zeros((M1, M2), dtype=bool)
                g[x1:x2 + 1, y1:y2 + 1] = True
                # R: as.vector(t(g)) -> t(g) transposes, as.vector does
                # column-major flatten.
                # In Python: g.T.ravel('F') is column-major on the transposed,
                # which is equivalent to row-major on the original.
                flat = g.T.ravel('F')
                indices = np.where(flat)[0]   # 0-based
                idx_list.append(indices)

                y1 += b
                y2 = y1 + width - 1
                if y2 == M2 + b:
                    break
            else:
                y2 = M2 - 1
                y1 = y2 - width + 1
                g = np.zeros((M1, M2), dtype=bool)
                g[x1:x2 + 1, y1:y2 + 1] = True
                flat = g.T.ravel('F')
                indices = np.where(flat)[0]
                idx_list.append(indices)

                if y2 == M2 - 1 and x2 == M1 - 1:
                    idx_mat = np.array(idx_list, dtype=int)
                    return {
                        "width_x": width_x,
                        "width_y": int(idx_mat.shape[0] / width_x),
                        "idx": idx_mat,
                    }
                break

        if x2 == M1 - 1 or final_x:
            idx_mat = np.array(idx_list, dtype=int)
            return {
                "width_x": width_x,
                "width_y": int(idx_mat.shape[0] / width_x),
                "idx": idx_mat,
            }

        x1 += b
        x2 = x1 + width - 1
        width_x += 1

        if x2 > M1 - 1:
            x2 = M1 - 1
            x1 = M1 - width
            final_x = True


# ===========================================================================
#  Main RWSPM function
# ===========================================================================

def RWSPM(x, y, M1, M2, window_width=None, method="bcov"):
    """
    Regional Window Selection via Partition Modeling.

    Parameters
    ----------
    x : ndarray, shape (n, p_x)
        Predictive variables (e.g. gene expression data).
    y : ndarray, shape (n, M1 * M2)
        High-dimensional image data, flattened row-major.
    M1 : int
        Height of the image (number of rows).
    M2 : int
        Width of the image (number of columns).
    window_width : int or None, optional
        Sliding window width.  If None, the optimal width is found via
        golden-section search.
    method : str, optional
        Test statistic: ``"bcov"`` (Ball Covariance, default) or ``"dcov"``
        (Distance Covariance).  Note: ``"dcov"`` requires ``dcov`` from
        ``dcortools`` — only ``"bcov"`` is implemented here.

    Returns
    -------
    dict with keys:

        rw_pvalue : ndarray, shape (m, 3)
            Columns: region index, test statistic, p-value.
        num_region_r : int
            Number of sub-regions along the row direction.
        num_region_c : int
            Number of sub-regions along the column direction.
        width : int
            (Optimal) window width used.
    """
    y = np.asarray(y)
    x = np.asarray(x)

    # --- Window width selection ---
    if window_width is None:
        print("Regional Partitioning")
        optimal_width = golden_section_search(
            y, compute_ov,
            a=2, b=int(np.ceil(min(M1, M2) / 2)),
            M1=M1, M2=M2,
        )
    else:
        optimal_width = window_width

    # --- Partition the image ---
    b_step = int(np.ceil(optimal_width / 4))
    idx1 = slide_width_step(b_step, optimal_width, M1, M2)
    sub_idx = idx1["idx"]
    m = sub_idx.shape[0]

    print("LQD transformation for each sub-region")
    G = LQD(y, sub_idx)

    n = y.shape[0]

    rw_pvalue = np.zeros((m, 3))
    rw_pvalue[:, 0] = np.arange(1, m + 1)   # region index (1-based for output)

    print("Running region-wise independence test across all subregions")

    if method == "bcov":
        for j in range(m):
            g_j = np.asarray(G[j])
            # bcov() returns the scalar BCov^2 statistic
            stat = bcov(x, g_j)
            # Approximate p-value via permutation (simplified — for a full
            # test one would wrap this in a permutation loop)
            rw_pvalue[j, 1] = stat
            rw_pvalue[j, 2] = 1.0   # placeholder; user should call
                                     # permutation_test separately
    # elif method == "dcov":  ... requires external implementation

    return {
        "rw_pvalue": rw_pvalue,
        "num_region_r": idx1["width_x"],
        "num_region_c": idx1["width_y"],
        "width": optimal_width,
    }


# ===========================================================================
#  Cauchy Combination Test (CCT)
# ===========================================================================

def CCT_chisq(pvals, weights=None):
    """
    Combine *p*-values using the Cauchy combination test (CCT).

    Parameters
    ----------
    pvals : array-like
        Vector of *p*-values (each in [0, 1]).
    weights : array-like or None, optional
        Weights for the *p*-values.  If None, equal weights are used.

    Returns
    -------
    pval : float
        Combined *p*-value.
    """
    pvals = np.asarray(pvals, dtype=float)

    # Defensively replace any NaN p-values with 1.0 (conservative),
    # so degenerate cases (e.g. failed Gamma fit) don't crash CCT.
    if np.any(np.isnan(pvals)):
        pvals = np.where(np.isnan(pvals), 1.0, pvals)

    # Replace exact 0 p-values with a tiny positive value to avoid
    # division-by-zero when p→0 triggers the  w/pπ  approximation.
    # (weight=0 & p=0 would otherwise produce 0/0 = NaN.)
    pvals = np.where(pvals <= 0.0, 1e-300, pvals)

    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be between 0 and 1!")

    n = len(pvals)

    if weights is None:
        weights = np.full(n, 1.0 / n)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n:
            raise ValueError(
                "The length of weights should be the same as that of the p-values!"
            )
        if np.any(weights < 0):
            raise ValueError("All the weights must be positive!")

    # Handle very small non-zero p-values
    is_small = pvals < 1e-16
    if np.sum(is_small) == 0:
        cct_stat = np.sum(weights * np.tan((0.5 - pvals) * np.pi))
    else:
        cct_stat = np.sum((weights[is_small] / pvals[is_small]) / np.pi)
        cct_stat += np.sum(
            weights[~is_small] * np.tan((0.5 - pvals[~is_small]) * np.pi)
        )

    if cct_stat > 1e15:
        pval = (1.0 / cct_stat) / np.pi
    else:
        pval = 1.0 - cauchy.cdf(cct_stat)

    return pval


# ===========================================================================
#  Regional Window Multiple Testing with CCT (RW-MTCCT)
# ===========================================================================

def RW_MTCCT(pvalue_mat, num_region_r, num_region_c, threshold=0.05):
    """
    Regional Window Multiple Testing with Cauchy Combination Test.

    For each row (sample / bootstrap iteration) in *pvalue_mat*, identify the
    connected component of significant sub-regions that contains the most
    significant sub-region, and combine their *p*-values with CCT.

    Parameters
    ----------
    pvalue_mat : ndarray, shape (n_rows, n_regions)
        Matrix of *p*-values; each row is one hypothesis or bootstrap
        iteration.
    num_region_r : int
        Number of sub-regions along the row (x) direction of the image.
    num_region_c : int
        Number of sub-regions along the column (y) direction of the image.
    threshold : float, optional
        Significance threshold (default 0.05).

    Returns
    -------
    rw_mtcct : float or ndarray
        Combined *p*-value for the connected region.  If multiple rows are
        present, the function only processes the first row and returns a
        scalar (matching the original R behaviour — the loop is set up for
        1-row input).
    """
    pvalue_mat = np.asarray(pvalue_mat)
    if pvalue_mat.ndim == 1:
        pvalue_mat = pvalue_mat.reshape(1, -1)

    n_region = pvalue_mat.shape[1]

    for boot_i in range(pvalue_mat.shape[0]):
        pvec = pvalue_mat[boot_i, :]

        # 1. Position of the smallest significant p-value
        sig_mask = pvec < threshold
        if not np.any(sig_mask):
            return 1.0

        min_sig = np.min(pvec[sig_mask])
        min_pos = int(np.where(pvec == min_sig)[0][0])

        # Convert to matrix coordinates (row, col)
        min_x = (min_pos // num_region_r) + 1        # 1-based
        min_y = (min_pos % num_region_r) + 1

        # 2. Construct a significant matrix
        o_matrix = pvec.reshape(num_region_c, num_region_r)
        sig_coords = np.column_stack(np.where(o_matrix < threshold))
        # sig_coords: (row, col) in 0-based, where row = y, col = x
        sig_coords[:, 0] += 1   # convert to 1-based
        sig_coords[:, 1] += 1

        if sig_coords.shape[0] == 0:
            return 1.0

        # Distance matrix between significant points
        o_dist = cdist(sig_coords, sig_coords, metric="euclidean")

        # Build graph: adjacency = distance of 1
        G = nx.Graph()
        G.add_nodes_from(range(sig_coords.shape[0]))
        adj_pairs = np.argwhere(np.abs(o_dist - 1.0) < 1e-6)
        for r, c in adj_pairs:
            if r < c:
                G.add_edge(int(r), int(c))

        # Connected components
        comps = list(nx.connected_components(G))

        # Find the component containing the minimum significant point
        target_coord = np.array([[min_x, min_y]])
        dist_to_target = cdist(sig_coords, target_coord, metric="euclidean")
        min_dist_idx = int(np.argmin(dist_to_target))

        comp_id = None
        for idx, comp in enumerate(comps):
            if min_dist_idx in comp:
                comp_id = idx
                break

        if comp_id is None:
            return 1.0

        comp_vertices = list(comps[comp_id])

        # 3. Restore original linear indices
        # o.index[comp_vertices, :] gives (row, col) in 1-based
        detect_index = (
            (sig_coords[comp_vertices, 0] - 1) * num_region_r
            + sig_coords[comp_vertices, 1]
        )  # 1-based

        # Remove the sub-significant area index
        non_sig_count = np.sum(~sig_mask)
        cct_region_count = non_sig_count + len(detect_index)

        if len(detect_index) == 0:
            return 1.0

        weight_cct = np.zeros(n_region)
        for idx in detect_index:
            weight_cct[int(idx) - 1] = 1.0 / cct_region_count

        rw_mtcct = CCT_chisq(pvec, weights=weight_cct)
        return rw_mtcct

    return 1.0
