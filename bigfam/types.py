"""Phase-boundary data objects. The type *is* the I/O contract.

Phase 1 knows the phenotype kind (continuous/binary); Phase 2 and 3 see only
(rho_hat, Sigma_hat). These small frozen dataclasses are the spine of the
design: they are the only things that cross a phase boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ── phase outputs ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RhoEstimate:
    """Phase 1 output. The phenotype kind disappears here."""
    rho_hat: np.ndarray      # (D,)    per-DOR similarity
    Sigma_hat: np.ndarray    # (D, D)  cluster sandwich covariance
    D: int = 3               # number of DOR levels (3)

    @property
    def sigma_hat(self) -> np.ndarray:   # (D,) standard errors
        return np.sqrt(np.diag(self.Sigma_hat))


@dataclass(frozen=True)
class WsEstimate:
    """Phase 2 output: the calibrated w_S point estimate (features -> ridge)."""
    w_s_cal: float           # calibrated environmental decay rate


@dataclass(frozen=True)
class Decomposition:
    """Phase 3 output = final result.

    (V_G, V_S) refit at the fixed w = w_s_cal, their conditional SE and z, and
    w_s_cal with its profile-likelihood 95% CI. Whether the decomposition is
    identified is read off that CI: w_S = 0.5 makes X(w) rank 1, so a CI covering
    0.5 means (V_G, V_S) are not separable. No trust label is emitted -- the CI
    is the report, and any cut on it belongs to the analysis, not the estimator.
    """
    V_G: float
    V_S: float
    se_VG_cond: float        # conditional SE at fixed w_s_cal = sqrt(diag A^-1)
    se_VS_cond: float
    z_VG: float              # V_G / se_VG_cond
    z_VS: float              # V_S / se_VS_cond
    w_s_cal: float           # decomposition point (clipped)
    wci_lo: float            # profile-likelihood 95% w-CI
    wci_hi: float


# ── learned-coefficient containers (artifacts) ───────────────────────────────
@dataclass(frozen=True)
class CalibrationCoef:
    """Phase 2 Step-2 calibration coefficients (artifacts/ws_calibration.json).

    Step 2 is a single ridge regression: w_S_cal = clip(ridge(standardized x)).
    The w_map confidence-blend (gate a,b + profile_width mean/std) was removed --
    it was inert in-distribution and a net-harmful instrument out-of-distribution;
    see research/phase2-drop-wmap/ (SQ1-4) for the measurement.
    """
    feature_order: List[str]     # = FEAT_ALL (single source of truth)
    scaler_mean: np.ndarray      # (24,) StandardScaler mean_
    scaler_scale: np.ndarray     # (24,) StandardScaler scale_
    ridge_coef: np.ndarray       # (24,) ridge coefficients (on standardized features)
    ridge_intercept: float
    ridge_alpha: float
    clip: tuple = (0.01, 0.99)
