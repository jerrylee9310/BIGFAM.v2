"""Phase 3: refit conditioning and what the profile CI says about identifiability."""
import numpy as np

from bigfam.phase3.refit import refit_fixed_ws
from bigfam.phase3.robust import profile_ci


def _data():
    rho_hat = np.array([0.30, 0.12, 0.05])
    Sigma = np.diag([1e-4, 1.5e-4, 2e-4])
    return rho_hat, Sigma


def test_refit_finite_and_nonneg():
    rho, Sig = _data()
    beta, se = refit_fixed_ws(rho, Sig, 0.2)
    assert np.all(np.isfinite(beta)) and np.all(beta >= 0)
    assert np.all(np.isfinite(se)) and np.all(se >= 0)


def test_se_increases_toward_half():
    rho, Sig = _data()
    _, se_far = refit_fixed_ws(rho, Sig, 0.10)    # far from 0.5
    _, se_near = refit_fixed_ws(rho, Sig, 0.45)   # near 0.5
    # decomposition becomes ill-conditioned approaching 0.5 -> larger SE
    assert se_near[0] > se_far[0]


def _curve(v_g, v_s, w=0.7):
    """Noise-free rho at (V_G, V_S) with shared-env decay w."""
    X = np.array([[0.5, 1.0], [0.25, w], [0.125, w * w]])
    return X @ np.array([v_g, v_s]), np.diag([1.6e-5, 1.6e-5, 1.6e-5])


def test_profile_ci_excludes_half_when_identified():
    # clean shared-env signal decaying at w=0.7, clearly not 0.5
    lo, hi = profile_ci(*_curve(0.5, 0.2))
    assert 0.01 <= lo <= hi <= 0.99
    assert not (lo <= 0.5 <= hi)


def test_profile_ci_covers_half_without_shared_env():
    # V_S = 0 -> w_S is unplaceable, so the CI must not claim identification
    lo, hi = profile_ci(*_curve(0.5, 0.0))
    assert lo <= 0.5 <= hi
