"""
Data generation functions for RWSPM simulation.

Implements the simulation settings from the RWSPM paper:
- Setting 1:      a=0, sx=sy=0,  sr=0   -> Type I error (X _|_ Y)
- Settings 2-4:   a=1, sx=sy=1-3, sr=1  -> Linear model
- Settings 5-7:   a=1, sx=sy=1-3, sr=1  -> Non-linear model

Grid:  M1 x M2 voxels over [-50,100] x [-50,50]
Disease region A = [-10, 10] x [-10, 10]
"""

import numpy as np
from typing import Dict, TypedDict


class _Setting(TypedDict):
    """One simulation setting: numeric parameters plus the model name."""

    a: float
    sigma_xy: float
    sigma_r: float
    model: str


SETTINGS: Dict[int, _Setting] = {
    1: {"a": 0.0, "sigma_xy": 0.0, "sigma_r": 0.0, "model": "independent"},
    2: {"a": 1.0, "sigma_xy": 1.0, "sigma_r": 1.0, "model": "linear"},
    3: {"a": 1.0, "sigma_xy": 2.0, "sigma_r": 1.0, "model": "linear"},
    4: {"a": 1.0, "sigma_xy": 3.0, "sigma_r": 1.0, "model": "linear"},
    5: {"a": 1.0, "sigma_xy": 1.0, "sigma_r": 1.0, "model": "nonlinear"},
    6: {"a": 1.0, "sigma_xy": 2.0, "sigma_r": 1.0, "model": "nonlinear"},
    7: {"a": 1.0, "sigma_xy": 3.0, "sigma_r": 1.0, "model": "nonlinear"},
}


def generate_data(setting, n=200, M1=150, M2=100, random_state=None):
    """
    Generate one simulation replicate.

    Parameters
    ----------
    setting : int
        Setting number 1-7.
    n : int
        Number of subjects (default 200).
    M1 : int
        Image height (rows).  Default 150.
    M2 : int
        Image width (columns).  Default 100.
    random_state : int or RandomState, optional

    Returns
    -------
    X : ndarray, shape (n, 1)
        Genotype dosage {0, 1, 2}.
    Y : ndarray, shape (n, M1*M2)
        Flattened image response.
    Z : ndarray, shape (n, 2)
        Covariates: [age/100, gender].
    """
    if setting not in SETTINGS:
        raise ValueError("setting must be 1-7, got %d" % setting)

    rng = np.random.RandomState(random_state)
    p = SETTINGS[setting]
    a = p["a"]
    sigma_xy = p["sigma_xy"]
    sigma_r = p["sigma_r"]
    model = p["model"]

    n_pix = M1 * M2

    # ---- Coordinate grid ----
    s1 = np.linspace(-50, 100, M1)
    s2 = np.linspace(-50, 50, M2)
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")
    S1f = S1.ravel()
    S2f = S2.ravel()
    mask_a = (S1 >= -10) & (S1 <= 10) & (S2 >= -10) & (S2 <= 10)
    mask_a = mask_a.ravel()
    a_idx = np.where(mask_a)[0]

    # ---- X : genotype Binomial(2, 0.3) ----
    X = rng.binomial(2, 0.3, size=n).astype(float)

    # ---- Z : covariates ----
    Z_age = rng.uniform(50, 100, size=n)
    Z_gender = rng.binomial(1, 0.5, size=n).astype(float)
    Z = np.column_stack([Z_age / 100.0, Z_gender])

    # ---- Non-genetic fixed effects ----
    gamma0 = np.zeros(n_pix)
    gamma1 = np.zeros(n_pix)
    gamma0[mask_a] = a
    gamma1[mask_a] = 0.25 * a

    # ---- Noise parameters ----
    V = rng.randn(n)
    xi1 = np.sign(V) * (np.abs(V) - np.floor(np.abs(V)))
    lam = rng.choice([-1, 1], size=n)
    xi2 = lam * np.sqrt(1.0 - xi1 ** 2)

    sin_part = np.sin(2 * np.pi * S1f[None, :] / 150.0)
    cos_part = np.cos(2 * np.pi * S2f[None, :] / 100.0)
    noise = xi1[:, None] * sin_part + xi2[:, None] * cos_part

    # ---- Build Y ----
    Y = np.zeros((n, n_pix))

    for i in range(n):
        # Individual affected subregion A_i
        if sigma_xy > 0:
            b_i1 = rng.normal(0, sigma_xy)
            b_i2 = rng.normal(0, sigma_xy)
        else:
            b_i1 = b_i2 = 0.0

        r_tilde = rng.normal(5, sigma_r)
        r_i = r_tilde * X[i] + Z_age[i] / 100.0 + Z_gender[i]

        # beta_i(s)
        beta_s = np.zeros(n_pix)
        if a > 0.0 and sigma_xy >= 0.0:
            dist2 = (S1f[a_idx] - b_i1)**2 + (S2f[a_idx] - b_i2)**2
            in_ai = dist2 <= r_i**2
            affected = a_idx[in_ai]
            if len(affected) > 0:
                d = np.sqrt(dist2[in_ai])
                beta_s[affected] = 0.8 * a * np.exp(-0.1 * d)

        if model == "independent":
            Y[i, :] = noise[i, :]
        elif model == "linear":
            Y[i, :] = (Z[i, 0] * gamma0 + Z[i, 1] * gamma1
                       + X[i] * beta_s + noise[i, :])
        else:
            Y[i, :] = (Z[i, 0] * gamma0 + Z[i, 1] * gamma1
                       + (X[i] * beta_s) ** 2 + noise[i, :])

    return X.reshape(-1, 1), Y, Z
