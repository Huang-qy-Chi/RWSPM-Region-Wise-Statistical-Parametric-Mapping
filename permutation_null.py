import numpy as np
from bcov import bcov
# from scipy import stats
# import matplotlib.pyplot as plt


#%%---------------------------------------------------------------------------------------------

def permutation_null_distribution(X, Y, B=1000, seed=123):
    """
    通过置换 Y 的行得到 n·BCov²ₙ 的零分布
    
    参数
    ----------
    X : ndarray, shape (n, p)
    Y : ndarray, shape (n, q)
    B : int — 置换次数
    seed : int — 随机种子
    
    返回
    -------
    null_stats : ndarray, shape (B,) — 置换零分布统计量
    """
    np.random.seed(seed)
    Y = np.asarray(Y)
    n = Y.shape[0]
    
    null_stats = np.zeros(B)
    
    for b in range(B):
        perm_id = np.random.permutation(n)
        Y_perm = Y[perm_id, :]
        null_stats[b] = n * bcov(X, Y_perm)
        
        if (b + 1) % 100 == 0:
            print(f"Permutation {b+1} of {B} completed")
    
    return null_stats


if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    # parameter settings
    n = 200; p = 5; q = 5; B = 200; seed = 2026
    
    # =========================================================
    # Generate independent X and Y under H₀ 
    # =========================================================
    X = np.random.randn(n, p)
    Y = np.random.randn(n, q)
    null_stats = permutation_null_distribution(X, Y, B=B, seed=seed + 1)
    # =========================================================
    # Density plot of null_stats
    # =========================================================
    # KDE estimation
    kde = gaussian_kde(null_stats)

    # Generate the range of null_stats
    x_range = np.linspace(null_stats.min() , null_stats.max() + 1, 500)
    kde_values = kde(x_range)

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, kde_values, 'b-', linewidth=2.5, label='Kernel Density Estimate')
    plt.fill_between(x_range, kde_values, alpha=0.3, color='blue')

    # add the histogram
    plt.hist(null_stats, bins=50, density=True, alpha=0.4, color='gray', edgecolor='black', label='Histogram')

    plt.title('Density Estimation Plot for null_stats', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
 

















