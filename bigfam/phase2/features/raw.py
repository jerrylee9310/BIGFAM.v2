"""Step 1d: raw / SE features (6).

Returns [rho_hat_1, rho_hat_2, rho_hat_3, se_max, se_mean, signal_rms_z].
"""
from __future__ import annotations

import numpy as np


def raw_feats_batch(rho_all, Sigmas):
    """rho_all: (N, 3), Sigmas: (N, 3, 3) -> (N, 6)."""
    sd = Sigmas[:, np.arange(3), np.arange(3)]
    se = np.sqrt(np.maximum(sd, 1e-24))
    se_max = se.max(1)
    se_mean = se.mean(1)
    signal_rms_z = np.sqrt(np.mean(rho_all ** 2 / np.maximum(sd, 1e-24), axis=1))
    return np.column_stack([rho_all, se_max, se_mean, signal_rms_z])
