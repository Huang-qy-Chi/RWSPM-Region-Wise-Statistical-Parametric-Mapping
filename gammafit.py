import numpy as np 

#%%---------------------------------------------------------------------------------------------

def gamma_moment_match(null_stats):
    """
    fit Gamma distribution by moment estimation
    
    参数
    ----------
    null_stats : ndarray — 置换零分布统计量
    
    返回
    -------
    dict — shape (α), scale (θ), mean, variance
    """
    mean_val = np.mean(null_stats)
    var_val = np.var(null_stats, ddof=1)  # 样本方差 (ddof=1 对应 R 的 var)
    
    alpha = mean_val ** 2 / var_val
    theta = var_val / mean_val
    
    return {
        'shape': alpha,
        'scale': theta,
        'mean': mean_val,
        'variance': var_val
    }


if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    from permutation_null import permutation_null_distribution
    from scipy import stats
    from bcov import bcov
    
    # parameter settings
    n = 200; p = 5; q = 5; B = 200; seed = 2026
    # =========================================================
    np.random.seed(seed)
    X = np.random.randn(n, p)
    Y = np.random.randn(n, q)
    T_obs = n * bcov(X, Y)
    print(f"\nObserved statistic T_obs = n * BCov²ₙ:")
    print(f"{T_obs:.6f}\n")
    null_stats = permutation_null_distribution(X, Y, B=B, seed=seed + 1)
    gamma_fit = gamma_moment_match(null_stats)
    alpha = gamma_fit['shape']
    theta = gamma_fit['scale']
    # Visualization
    # =========================================================
    x_grid = np.linspace(0, np.quantile(null_stats, 0.995), 500)
    gamma_density = stats.gamma.pdf(x_grid, a=alpha, scale=theta)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ---- Plot 1: Histogram + Gamma density ----
    ax1 = axes[0]
    ax1.hist(null_stats, bins=35, density=True, color='grey',
             edgecolor='black', alpha=0.7, label='Permutation null')
    ax1.plot(x_grid, gamma_density, linewidth=2, color='steelblue',
             label='Gamma approx')
    ax1.axvline(T_obs, linestyle='--', linewidth=2, color='red',
                label=f'$T_{{obs}} = {T_obs:.3f}$')
    ax1.set_xlabel(r'$n \cdot \widehat{BCov}_n^2$')
    ax1.set_ylabel('Density')
    ax1.set_title('Permutation Null vs Gamma Approximation')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ---- Plot 2: Empirical CDF vs Gamma CDF ----
    ax2 = axes[1]
    sorted_stats = np.sort(null_stats)
    ecdf_vals = np.arange(1, len(sorted_stats) + 1) / len(sorted_stats)
    gamma_cdf_vals = stats.gamma.cdf(sorted_stats, a=alpha, scale=theta)
    
    ax2.plot(sorted_stats, ecdf_vals, linewidth=2, label='Permutation ECDF')
    ax2.plot(sorted_stats, gamma_cdf_vals, linestyle='--', linewidth=2,
             label='Gamma CDF')
    ax2.set_xlabel(r'$n \cdot \widehat{BCov}_n^2$')
    ax2.set_ylabel('CDF')
    ax2.set_title('Empirical CDF vs Gamma CDF')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ---- Plot 3: QQ plot ----
    ax3 = axes[2]
    probs = (np.arange(1, B + 1) - 0.5) / B
    gamma_quantiles = stats.gamma.ppf(probs, a=alpha, scale=theta)
    
    ax3.scatter(gamma_quantiles, sorted_stats, s=15, alpha=0.6, color='steelblue')
    max_val = max(np.max(gamma_quantiles), np.max(sorted_stats))
    ax3.plot([0, max_val], [0, max_val], linestyle='--', color='red', linewidth=1.5)
    ax3.set_xlabel('Gamma theoretical quantiles')
    ax3.set_ylabel('Permutation empirical quantiles')
    ax3.set_title('QQ Plot')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()



















