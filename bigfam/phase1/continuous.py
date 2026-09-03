"""Continuous phenotype: residualize on covariates, then OLS on the residuals."""
from __future__ import annotations

import numpy as np

from .sandwich import cluster_meat, sandwich_cov


def estimate_gamma(p, pheno, cov, cov_cols):
    """Covariate coefficients from an OLS over *individuals*, not pairs.

    Fitting on the pair table would weight an individual by how many pairs they
    appear in, so individuals are deduplicated first.
    """
    import pandas as pd
    unique_ids = pd.unique(pd.concat([p["id1"], p["id2"]], ignore_index=True))
    valid = np.intersect1d(unique_ids, pheno.index.intersection(cov.index))

    y = pheno.loc[valid, "phenotype"].values
    X = cov.loc[valid, cov_cols].values
    W = np.column_stack([np.ones(len(valid)), X])   # (I, 1+p)
    return np.linalg.lstsq(W, y, rcond=None)[0]      # (1+p,)


def residualize(data, gamma_hat, cov_cols):
    """Subtract the covariate fit: Y_tilde_k = Y_k - W_k @ gamma_hat."""
    def W(prefix):
        X = data[[f"{prefix}_{c}" for c in cov_cols]].values
        return np.column_stack([np.ones(len(data)), X])

    y_tilde_1 = data["y1"].values - W("x1") @ gamma_hat
    y_tilde_2 = data["y2"].values - W("x2") @ gamma_hat
    return y_tilde_1, y_tilde_2


def fit_rho_ols(y_tilde_1, y_tilde_2, g, D=3):
    """Residual OLS: rho_hat = (Z^T Z)^-1 Z^T y_tilde_1, Z = diag(Y~2) A.

    A is the DOR indicator matrix, so rho_hat has one entry per DOR level.
    """
    N = len(g)
    A = np.zeros((N, D))
    for d in range(1, D + 1):
        A[g == d, d - 1] = 1.0
    Z = y_tilde_2[:, None] * A                          # (N, D) = diag(Y~2) @ A
    rho_hat = np.linalg.lstsq(Z, y_tilde_1, rcond=None)[0]
    return rho_hat, Z


def estimate_rho_sigma(data, gamma_hat, cov_cols, D=3):
    """Full continuous path on a flipped & concatenated pair table.

    -> (rho_hat (D,), Sigma_hat (D, D)).
    """
    y_tilde_1, y_tilde_2 = residualize(data, gamma_hat, cov_cols)
    g = data["dor"].values
    rho_hat, Z = fit_rho_ols(y_tilde_1, y_tilde_2, g, D)

    e_hat = y_tilde_1 - Z @ rho_hat        # (N,)
    scores = Z * e_hat[:, None]            # (N, D): s_n = Z_n * e_n
    B = Z.T @ Z
    M = cluster_meat(scores, data["id1"].values, data["id2"].values)
    Sigma_hat = sandwich_cov(B, M)
    return rho_hat, Sigma_hat
