"""Step 1a: Jensen-corrected GLS slope features (3).

    y* = log2(rho_hat) + Sigma_dd / (2 rho_hat^2 ln2)        (Jensen correction)
    J_dd = (1 - Sigma_dd / rho_hat^2) / (rho_hat ln2)         (delta-method Jacobian)
    V* = J Sigma J
    beta* = (H^T V*^-1 H)^-1 H^T V*^-1 y*                     (GLS, ridge 1e-8 on A)

slope_hat = beta*_2, slope_se = sqrt[A^-1]_22, slope_z = (slope_hat - 1)/slope_se.
"""
from __future__ import annotations

import numpy as np

from ...config import H, RIDGE_SLOPE


def slope_feats_batch(rho_all, Sigmas):
    """rho_all: (N, 3), Sigmas: (N, 3, 3) -> (N, 3) [slope_hat, slope_se, slope_z]."""
    sd = Sigmas[:, np.arange(3), np.arange(3)]
    y = np.log2(rho_all) + sd / (2.0 * rho_all ** 2 * np.log(2))
    J_d = (1.0 - sd / rho_all ** 2) / (rho_all * np.log(2))
    V = J_d[:, :, None] * Sigmas * J_d[:, None, :]
    V_inv = np.linalg.inv(V)
    A = np.einsum("kp,nkl,lq->npq", H, V_inv, H)
    A_inv = np.linalg.inv(A + RIDGE_SLOPE * np.eye(2)[None])
    Viy = np.einsum("nkl,nl->nk", V_inv, y)
    HtViy = np.einsum("kp,nk->np", H, Viy)
    b = np.einsum("npq,nq->np", A_inv, HtViy)
    sh = b[:, 1]
    se = np.sqrt(np.maximum(A_inv[:, 1, 1], 1e-12))
    return np.column_stack([sh, se, (sh - 1.0) / se])
