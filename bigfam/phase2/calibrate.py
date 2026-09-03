"""Step 2: ridge calibration.

    w_S_cal = clip(ridge_intercept + standardized(x) @ ridge_coef)

The ridge input is the StandardScaler-standardized 24-feature vector.

History: Step 2 used to blend a likelihood-only point estimate (w_map) into
w_ridge via a confidence gate alpha = sigmoid(a + b*c), c from profile_width.
That blend was removed -- it shipped inert (alpha ~ 0 over the whole realizable
profile_width range, so w_cal == w_ridge to <1e-3) and, measured out-of-
distribution, w_map was too noisy to hedge prior-pull while a fixed admixture
hurt under the realistic prior. The right OOD lever is retraining/estimating the
prior, not a w_map admixture. Measurement: research/phase2-drop-wmap/
(SQ1 headroom, SQ2 separability, SQ3 deployable gate, SQ4 OOD).
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
