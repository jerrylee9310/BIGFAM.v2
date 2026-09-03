"""Step 1: 24 features. FEAT_ALL order is the contract (single source of truth).

StandardScaler / Ridge / GBR are all positional, so the training order
(FEAT_ALL) and the inference order (extract_features) MUST match exactly or you
get silently wrong values. calibrate.py's i_w_map / i_profile_width are derived
from FEAT_ALL.index(...).
"""
from __future__ import annotations

import numpy as np

from ...config import RHO_FLOOR
from .slope import slope_feats_batch
from .profile import profile_feats_batch
from .contrast import contrast_feats_batch
from .raw import raw_feats_batch

FEAT_BASE = [                                  # 15
    "slope_hat", "slope_se", "slope_z",
    "w_map", "profile_width", "middle_mass",
    "D_2_hat", "D_3_hat", "I_D2", "I_D3", "ratio_naive",
    "rho_hat_1", "rho_hat_2", "rho_hat_3", "se_max",
]
FEAT_NEW = [                                   # 9
    "w_mean", "w_median", "effective_count", "map_mean_gap",
    "fieller_bounded", "fieller_width", "se_mean", "signal_rms_z", "any_nonpos",
]
FEAT_ALL = FEAT_BASE + FEAT_NEW                # 24

# group output column -> feature name (batch concat order, NOT FEAT_ALL order)
_SLOPE_COLS = ["slope_hat", "slope_se", "slope_z"]
_PROFILE_COLS = ["w_map", "profile_width", "middle_mass",
                 "w_mean", "w_median", "effective_count", "map_mean_gap"]
_CONTRAST_COLS = ["D_2_hat", "D_3_hat", "I_D2", "I_D3",
                  "ratio_naive", "fieller_bounded", "fieller_width"]
_RAW_COLS = ["rho_hat_1", "rho_hat_2", "rho_hat_3", "se_max", "se_mean", "signal_rms_z"]


def compute_feature_dict(rho_all, Sigmas):
    """Batch feature computation -> dict name -> (N,) array, all 24 names.

    any_nonpos is computed from the *pre-floor* rho_hat; rho_hat is then floored
    to RHO_FLOOR before the (log-using) features are computed.
    """
    rho_all = np.asarray(rho_all, dtype=float)
    Sigmas = np.asarray(Sigmas, dtype=float)

    any_nonpos = (rho_all <= 0).any(axis=1).astype(float)
    rho_floored = np.maximum(rho_all, RHO_FLOOR)
    Sigma_invs = np.linalg.inv(Sigmas)

    sf = slope_feats_batch(rho_floored, Sigmas)
    pf = profile_feats_batch(rho_floored, Sigma_invs)
    cf = contrast_feats_batch(rho_floored, Sigmas)
    rf = raw_feats_batch(rho_floored, Sigmas)

    out = {}
    for j, name in enumerate(_SLOPE_COLS):
        out[name] = sf[:, j]
    for j, name in enumerate(_PROFILE_COLS):
        out[name] = pf[:, j]
    for j, name in enumerate(_CONTRAST_COLS):
        out[name] = cf[:, j]
    for j, name in enumerate(_RAW_COLS):
        out[name] = rf[:, j]
    out["any_nonpos"] = any_nonpos
    return out


def extract_features(rho) -> np.ndarray:
    """Single-family: RhoEstimate -> (24,) vector in FEAT_ALL order."""
    d = compute_feature_dict(rho.rho_hat[None], rho.Sigma_hat[None])
    return np.array([d[name][0] for name in FEAT_ALL], dtype=float)


__all__ = [
    "FEAT_BASE", "FEAT_NEW", "FEAT_ALL",
    "slope_feats_batch", "profile_feats_batch", "contrast_feats_batch", "raw_feats_batch",
    "compute_feature_dict", "extract_features",
]
