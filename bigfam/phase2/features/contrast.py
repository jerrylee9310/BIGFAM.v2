"""Step 1c: D2/D3 contrasts + Fieller features (7).

    D2 = rho2 - rho1/2,   D3 = rho3 - rho2/2
    I_D2 = |D2|/sqrt(var D2), ratio_naive = clip(D3/D2, -2, 3)
    Fieller CI for theta = D3/D2: g = Z95^2 var(D2)/D2^2;
    bounded = (g<1) & (I_D2>Z95); width capped at 3.0 when unbounded.

Returns [D2, D3, I_D2, I_D3, ratio_naive, fieller_bounded, fieller_width].
"""
from __future__ import annotations

import numpy as np

from ...config import Z95


def contrast_feats_batch(rho_all, Sigmas):
    """rho_all: (N, 3), Sigmas: (N, 3, 3) -> (N, 7)."""
    D2 = rho_all[:, 1] - rho_all[:, 0] / 2.0
    D3 = rho_all[:, 2] - rho_all[:, 1] / 2.0
    var_D2 = np.maximum(0.25 * Sigmas[:, 0, 0] + Sigmas[:, 1, 1] - Sigmas[:, 0, 1], 1e-12)
    var_D3 = np.maximum(0.25 * Sigmas[:, 1, 1] + Sigmas[:, 2, 2] - Sigmas[:, 1, 2], 1e-12)
    cov_23 = (Sigmas[:, 1, 2] - 0.5 * Sigmas[:, 1, 1] - 0.5 * Sigmas[:, 0, 2]
              + 0.25 * Sigmas[:, 0, 1])
    I_D2 = np.abs(D2) / np.sqrt(var_D2)
    I_D3 = np.abs(D3) / np.sqrt(var_D3)
    D2_ok = np.abs(D2) > 1e-9
    D2_safe = np.where(D2_ok, D2, 1e-9)              # avoid 0/0 warning; masked out below
    ratio = np.clip(np.where(D2_ok, D3 / D2_safe, 0.5), -2.0, 3.0)

    g = (Z95 ** 2) * var_D2 / np.maximum(D2 ** 2, 1e-24)
    bounded = ((g < 1.0) & (I_D2 > Z95)).astype(float)
    rat = D3 / D2_safe
    disc = var_D3 - 2.0 * rat * cov_23 + rat ** 2 * var_D2 - g * (var_D3 - cov_23 ** 2 / var_D2)
    disc = np.maximum(disc, 0.0)
    Bspr = (Z95 / np.maximum(np.abs(D2), 1e-9)) * np.sqrt(disc)
    denom = np.where(np.abs(1.0 - g) > 1e-9, 1.0 - g, 1e-9)
    width = np.abs(2.0 * Bspr / denom)
    fieller_width = np.where(bounded > 0, np.minimum(width, 3.0), 3.0)

    return np.column_stack([D2, D3, I_D2, I_D3, ratio, bounded, fieller_width])
