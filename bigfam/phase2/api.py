"""Phase 2: (rho_hat, Sigma_hat) -> w_C, the shared-environment decay rate.

24 summary features of the Phase 1 estimate, fed to a ridge model whose
coefficients were fit offline by simulation (see train.py).
"""
from __future__ import annotations

from ..types import RhoEstimate, WcEstimate, CalibrationCoef
from .features import extract_features
from .calibrate import calibrate_wc


def estimate_wc(rho: RhoEstimate, calib: CalibrationCoef) -> WcEstimate:
    """Estimate w_C. calib comes from bigfam.io.load_artifacts().

    A point estimate only; the uncertainty in w_C is reported by Phase 3 as the
    profile CI, not as a per-estimate standard error.
    """
    return WcEstimate(w_c_cal=calibrate_wc(extract_features(rho), calib))
