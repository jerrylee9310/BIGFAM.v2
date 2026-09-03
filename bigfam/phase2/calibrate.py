"""Step 2: ridge calibration.

    w_S_cal = clip(ridge_intercept + standardized(x) @ ridge_coef)

x is the 24-feature vector in FEAT_ALL order; the scaler and ridge coefficients
come from the trained artifact (bigfam.io.load_artifacts).
"""
from __future__ import annotations

import numpy as np

from ..types import CalibrationCoef


def calibrate_ws(x: np.ndarray, calib: CalibrationCoef) -> float:
    """x: (24,) in FEAT_ALL order -> calibrated w_S (float)."""
    x = np.asarray(x, dtype=float)
    lo, hi = calib.clip
    x_std = (x - calib.scaler_mean) / calib.scaler_scale
    return float(np.clip(calib.ridge_intercept + x_std @ calib.ridge_coef, lo, hi))
