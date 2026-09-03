"""Phase 2: (rho_hat, Sigma_hat) -> w_S, the shared-environment decay rate.

24 summary features of the Phase 1 estimate, fed to a ridge model whose
coefficients were fit offline by simulation (see train.py).
"""
from __future__ import annotations

from ..types import RhoEstimate, WsEstimate, CalibrationCoef
from .features import extract_features
from .calibrate import calibrate_ws


def estimate_ws(rho: RhoEstimate, calib: CalibrationCoef) -> WsEstimate:
    """Estimate w_S. calib comes from bigfam.io.load_artifacts().

    A point estimate only; the uncertainty in w_S is reported by Phase 3 as the
    profile CI, not as a per-estimate standard error.
    """
    return WsEstimate(w_s_cal=calibrate_ws(extract_features(rho), calib))
