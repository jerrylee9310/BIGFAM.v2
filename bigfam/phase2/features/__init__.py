"""Step 1: the 24 features summarising (rho_hat, Sigma_hat).

FEAT_ALL is the single source of truth for their order. The scaler and ridge
are positional, so training and inference MUST build the vector in this exact
order or the calibration silently returns wrong values.
"""
from __future__ import annotations

import numpy as np

from ...config import RHO_FLOOR
from .slope import slope_feats_batch
from .profile import profile_feats_batch
from .contrast import contrast_feats_batch
from .raw import raw_feats_batch

# The 24 features, in the order the trained artifact expects. Changing this
# order (or the names) invalidates every existing ws_calibration.json.
FEAT_ALL = [
    "slope_hat", "slope_se", "slope_z",                    # slope.py
    "w_map", "profile_width", "middle_mass",               # profile.py
    "D_2_hat", "D_3_hat", "I_D2", "I_D3", "ratio_naive",   # contrast.py
    "rho_hat_1", "rho_hat_2", "rho_hat_3", "se_max",       # raw.py
    "w_mean", "w_median", "effective_count", "map_mean_gap",
    "fieller_bounded", "fieller_width",
    "se_mean", "signal_rms_z",
    "any_nonpos",
]

# each batch function returns its columns in this order (not FEAT_ALL order)
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
    "FEAT_ALL", "compute_feature_dict", "extract_features",
    "slope_feats_batch", "profile_feats_batch", "contrast_feats_batch", "raw_feats_batch",
]
