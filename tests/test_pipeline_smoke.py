"""End-to-end smoke + Phase3/Phase2 NNLS consistency.

The artifact-dependent smoke test is skipped if artifacts/ has not been built.
"""
from pathlib import Path

import numpy as np
import pytest

from bigfam.types import RhoEstimate
from bigfam.core.nnls import nnls_2d_batch
from bigfam.core.design import design_matrix
from bigfam.phase3.refit import refit_fixed_ws

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _rho():
    return RhoEstimate(np.array([0.30, 0.12, 0.05]),
                       np.diag([1e-4, 1.5e-4, 2e-4]), 3)


def test_phase3_equals_profile_nnls():
    """Phase 3 refit at w == direct profile NNLS at the same w (same solver)."""
    rho = _rho()
    w = 0.2                                  # away from 0.5; relative ridge negligible
    beta_refit, _, _, _ = refit_fixed_ws(rho.rho_hat, rho.Sigma_hat, w)

    X = design_matrix(w)                     # (3, 2)
    Sinv = np.linalg.inv(rho.Sigma_hat)
    ATA = X.T @ Sinv @ X
    ATz = X.T @ Sinv @ rho.rho_hat
    beta_profile = nnls_2d_batch(ATA[None], ATz[None])[0]

    np.testing.assert_allclose(beta_refit, beta_profile, atol=1e-6)


@pytest.mark.skipif(not (ARTIFACTS / "ws_calibration.json").exists(),
                    reason="artifacts not built; run scripts/train_phase2.py")
def test_full_pipeline_finite():
    import bigfam
    from bigfam.io import load_artifacts

    rho = _rho()
    calib = load_artifacts(ARTIFACTS)
    ws = bigfam.estimate_ws(rho, calib)
    dec = bigfam.decompose(rho, ws)

    assert 0.01 <= ws.w_s_cal <= 0.99
    for v in (dec.V_G, dec.V_S, dec.se_VG_cond, dec.se_VS_cond):
        assert np.isfinite(v)
    assert dec.V_G >= 0 and dec.V_S >= 0
    assert 0.01 <= dec.wci_lo <= dec.wci_hi <= 0.99
