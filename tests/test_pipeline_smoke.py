"""End-to-end smoke + Phase3/Phase2 NNLS consistency."""
from pathlib import Path

import numpy as np
import pytest

from bigfam.types import RhoEstimate
from bigfam.core.nnls import nnls_2d_batch
from bigfam.core.design import design_matrix
from bigfam.phase3.refit import refit_fixed_wc

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "bigfam" / "artifacts"
TOY_DATA = ROOT / "examples" / "toy_data"


def _rho():
    return RhoEstimate(np.array([0.30, 0.12, 0.05]),
                       np.diag([1e-4, 1.5e-4, 2e-4]), 3)


def test_phase3_equals_profile_nnls():
    """Phase 3 refit at w == direct profile NNLS at the same w (same solver)."""
    rho = _rho()
    w = 0.2                                  # away from 0.5; relative ridge negligible
    beta_refit, _ = refit_fixed_wc(rho.rho_hat, rho.Sigma_hat, w)

    X = design_matrix(w)                     # (3, 2)
    Sinv = np.linalg.inv(rho.Sigma_hat)
    ATA = X.T @ Sinv @ X
    ATz = X.T @ Sinv @ rho.rho_hat
    beta_profile = nnls_2d_batch(ATA[None], ATz[None])[0]

    np.testing.assert_allclose(beta_refit, beta_profile, atol=1e-6)


@pytest.mark.skipif(not (ARTIFACTS / "wc_calibration.json").exists(),
                    reason="artifacts not built; run scripts/train_phase2.py")
def test_full_pipeline_finite():
    import bigfam
    from bigfam.io import load_artifacts

    rho = _rho()
    calib = load_artifacts(ARTIFACTS)
    wc = bigfam.estimate_wc(rho, calib)
    dec = bigfam.decompose(rho, wc)

    assert 0.01 <= wc.w_c_cal <= 0.99
    for v in (dec.V_A, dec.V_C, dec.se_VA_cond, dec.se_VC_cond):
        assert np.isfinite(v)
    assert dec.V_A >= 0 and dec.V_C >= 0
    assert 0.01 <= dec.wci_lo <= dec.wci_hi <= 0.99


@pytest.mark.skipif(not (ARTIFACTS / "wc_calibration.json").exists(),
                    reason="artifacts not built; run scripts/train_phase2.py")
def test_toy_data_runs_end_to_end():
    """The committed example layout loads and the three phases run on it."""
    import bigfam
    from bigfam.io import load_pairs, load_artifacts

    pairs, cov, pheno = load_pairs(TOY_DATA, "height", "continuous")
    rho = bigfam.estimate_rho(pairs, cov, pheno, "continuous", cov_cols=["age", "sex"])
    dec = bigfam.decompose(rho, bigfam.estimate_wc(rho, load_artifacts(ARTIFACTS)))

    # generated at V_A=0.5, V_C=0.2 -- loose bounds, it is a small sample
    assert 0.3 < dec.V_A < 0.7
    assert 0.05 < dec.V_C < 0.35
