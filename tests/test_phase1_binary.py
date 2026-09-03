"""Phase 1 binary estimator recovery (bivariate probit MLE).

Simulate pairs from the liability model with known per-DOR rho, run the full
binary path (flip & concat -> probit MLE -> sandwich), and assert rho is
recovered. test_phase1_binary_fast checks the pieces; this checks the whole.
"""
import numpy as np
import pandas as pd

COV_COLS = ["age", "sex", "age_x_sex", "age2_x_sex"]   # fixture covariates
from bigfam.phase1.binary import estimate_rho_sigma
from bigfam.phase1.pairs import flip_concat


def test_binary_probit_recovers_rho():
    rng = np.random.default_rng(0)
    n = 12000
    rho_true = np.array([0.50, 0.30, 0.10])
    g = rng.integers(1, 4, n)
    r = rho_true[g - 1]

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    L1 = z1                                        # latent liabilities, mean 0
    L2 = r * z1 + np.sqrt(1.0 - r ** 2) * z2       # corr(L1, L2) = r
    y1 = (L1 > 0).astype(float)
    y2 = (L2 > 0).astype(float)

    data = pd.DataFrame({"y1": y1, "y2": y2, "dor": g,
                         "id1": np.arange(n), "id2": np.arange(n, 2 * n)})
    for c in COV_COLS:                             # full-rank design, true effect 0
        data[f"x1_{c}"] = rng.normal(size=n)
        data[f"x2_{c}"] = rng.normal(size=n)
    data = flip_concat(data, COV_COLS)

    rho_hat, Sigma, res = estimate_rho_sigma(data, COV_COLS, D=3)
    assert res.success or res.status == 0
    np.testing.assert_allclose(rho_hat, rho_true, atol=0.05)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)
    assert np.all(np.diag(Sigma) >= 0)
