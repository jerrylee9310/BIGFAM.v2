"""Phase 1 continuous estimator: residual-OLS recovery + the Z broadcast fix.

fit_rho_ols used to build Z as np.diag(y_tilde_2) @ A, an (N, N) dense matrix
(7.2 GB at N=30k, OOM for real data). It is now the equivalent broadcast
y_tilde_2[:, None] * A. These tests lock both the equivalence and that the
estimator still recovers rho.
"""
import numpy as np
import pandas as pd

from bigfam.config import COV_COLS
from bigfam.phase1.continuous import fit_rho_ols, estimate_rho_sigma


def _onehot(g, D=3):
    A = np.zeros((len(g), D))
    for d in range(1, D + 1):
        A[g == d, d - 1] = 1.0
    return A


def test_fit_rho_ols_Z_is_broadcast():
    """Z must equal diag(y2) @ A without materializing the N x N matrix."""
    rng = np.random.default_rng(0)
    n = 500
    y2 = rng.normal(size=n)
    g = rng.integers(1, 4, n)
    A = _onehot(g)
    _, Z = fit_rho_ols(rng.normal(size=n), y2, g, D=3)
    np.testing.assert_allclose(Z, y2[:, None] * A)
    np.testing.assert_allclose(Z, np.diag(y2) @ A)        # same as the old form


def test_fit_rho_ols_recovers_rho():
    rng = np.random.default_rng(1)
    n = 60000
    rho = np.array([0.30, 0.12, 0.05])
    g = rng.integers(1, 4, n)
    y2 = rng.normal(size=n)
    y1 = rho[g - 1] * y2 + rng.normal(scale=0.1, size=n)   # rho_g * y2 + noise
    rho_hat, _ = fit_rho_ols(y1, y2, g, D=3)
    np.testing.assert_allclose(rho_hat, rho, atol=0.02)


def test_estimate_rho_sigma_end_to_end():
    """Full continuous path (gamma=0 residuals) recovers rho; Sigma is sane."""
    rng = np.random.default_rng(2)
    n = 40000
    rho = np.array([0.30, 0.12, 0.05])
    g = rng.integers(1, 4, n)
    y2 = rng.normal(size=n)
    y1 = rho[g - 1] * y2 + rng.normal(scale=0.1, size=n)

    data = pd.DataFrame({"y1": y1, "y2": y2, "dor": g,
                         "id1": np.arange(n), "id2": np.arange(n, 2 * n)})
    for c in COV_COLS:                                     # zero covariates -> gamma=0 residuals
        data[f"x1_{c}"] = 0.0
        data[f"x2_{c}"] = 0.0
    gamma0 = np.zeros(1 + len(COV_COLS))

    rho_hat, Sigma = estimate_rho_sigma(data, gamma0, COV_COLS, D=3)
    np.testing.assert_allclose(rho_hat, rho, atol=0.02)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)   # symmetric
    assert np.all(np.diag(Sigma) > 0)
