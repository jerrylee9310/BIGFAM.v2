"""Phase 3 refit: finite output, SE grows toward the w_S=0.5 singular point."""
import numpy as np

from bigfam.phase3.refit import refit_fixed_ws


def _data():
    rho_hat = np.array([0.30, 0.12, 0.05])
    Sigma = np.diag([1e-4, 1.5e-4, 2e-4])
    return rho_hat, Sigma


def test_refit_finite_and_nonneg():
    rho, Sig = _data()
    beta, se, Ainv, dbeta = refit_fixed_ws(rho, Sig, 0.2)
    assert np.all(np.isfinite(beta)) and np.all(beta >= 0)
    assert np.all(np.isfinite(se)) and np.all(se >= 0)


def test_se_increases_toward_half():
    rho, Sig = _data()
    _, se_far, _, _ = refit_fixed_ws(rho, Sig, 0.10)    # far from 0.5
    _, se_near, _, _ = refit_fixed_ws(rho, Sig, 0.45)   # near 0.5
    # decomposition becomes ill-conditioned approaching 0.5 -> larger SE
    assert se_near[0] > se_far[0]
