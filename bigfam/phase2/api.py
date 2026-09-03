"""Phase 2 unified entry point.

    (rho_hat, Sigma_hat) -> w_s_cal

The reliability GBR / gate / sigma_w are gone: the reporting tier (Phase 3) does
not use them (see scratch v2-fieller-refactor step5). w_S uncertainty is carried
by the profile w-CI in Phase 3, not a per-point sigma_w.
"""
from __future__ import annotations

from ..types import RhoEstimate, WsEstimate, CalibrationCoef
from .features import extract_features
from .calibrate import calibrate_ws


def estimate_ws(rho: RhoEstimate, calib: CalibrationCoef) -> WsEstimate:
    """Calibrate the environmental decay rate w_S."""
    x = extract_features(rho)                    # (24,) FEAT_ALL order
    return WsEstimate(w_s_cal=calibrate_ws(x, calib))
