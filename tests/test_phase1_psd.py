"""Phase 1 Sigma_hat PSD guarantee.

The cluster sandwich can emit an indefinite Sigma_hat (M = M1+M2-M12 is a
difference of PSD matrices). estimate_rho projects it to the nearest PSD at the
phase boundary so phase2 cholesky/inv consumers stay valid. Tested at two
levels: the nearest_psd kernel, and the estimate_rho boundary (binary estimator
monkeypatched to return an indefinite Sigma).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

COV_COLS = ["age", "sex", "age_x_sex", "age2_x_sex"]   # fixture covariates
from bigfam.core.linalg import nearest_psd
from bigfam.phase1 import api as _api


# ── nearest_psd kernel ─────────────────────────────────────────────────────────
def test_nearest_psd_repairs_indefinite():
    M = np.diag([1e-4, 1e-4, -1e-5])          # one negative eigenvalue
    P = nearest_psd(M)
    assert np.linalg.eigvalsh(P).min() > 0
    np.testing.assert_allclose(P, P.T)


def test_nearest_psd_leaves_pd_unchanged():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(3, 3))
    M = B @ B.T + 1e-3 * np.eye(3)            # PD
    np.testing.assert_allclose(nearest_psd(M), 0.5 * (M + M.T))


# ── estimate_rho boundary ──────────────────────────────────────────────────────
def _mini_binary():
    """Smallest table that covers all three dor levels (estimate_rho requires it)."""
    ids = [1, 2, 3, 4]
    cov = pd.DataFrame({c: np.arange(4, dtype=float) for c in COV_COLS},
                       index=pd.Index(ids, name="id"))
    pheno = pd.DataFrame({"phenotype": [0, 1, 1, 0]}, index=pd.Index(ids, name="id"))
    pairs = pd.DataFrame({"id1": [1, 1, 2, 1], "id2": [2, 3, 3, 4],
                          "dor": [1, 2, 1, 3]})
    return pairs, cov, pheno


def test_estimate_rho_projects_indefinite_sigma(monkeypatch):
    indef = np.diag([1e-4, 1e-4, -1e-5])
    rho = np.array([0.30, 0.12, 0.05])
    monkeypatch.setattr(_api._bin, "estimate_rho_sigma",
                        lambda data, cov_cols, D: (rho, indef, None))

    pairs, cov, pheno = _mini_binary()
    with pytest.warns(UserWarning, match="not PSD"):
        est = _api.estimate_rho(pairs, cov, pheno, "binary", COV_COLS)
    assert np.linalg.eigvalsh(est.Sigma_hat).min() > 0
    assert np.all(np.isfinite(est.sigma_hat))


def test_estimate_rho_keeps_pd_sigma(monkeypatch):
    pd_sigma = np.diag([1e-4, 2e-4, 3e-4])
    rho = np.array([0.30, 0.12, 0.05])
    monkeypatch.setattr(_api._bin, "estimate_rho_sigma",
                        lambda data, cov_cols, D: (rho, pd_sigma, None))

    pairs, cov, pheno = _mini_binary()
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # no PSD warning expected
        est = _api.estimate_rho(pairs, cov, pheno, "binary", COV_COLS)
    np.testing.assert_allclose(est.Sigma_hat, pd_sigma)
