"""Step 1b: NNLS profile-out features (7).

For each w on WS_PROFILE: NNLS GLS fit beta>=0 at X(w), loss ell = e^T Sigma^-1 e.
w_map = argmin ell. CI width via {w: ell <= ell_min + 3.84}. Softmax weights
p ~ exp(-0.5(ell - ell_min)) give middle_mass, w_mean, w_median, effective_count.

Returns [w_map, profile_width, middle_mass, w_mean, w_median, effective_count, map_mean_gap].
"""
from __future__ import annotations

import numpy as np

from ...config import WS_PROFILE, CHI2_95
from ...core.design import design_matrix
from ...core.nnls import nnls_2d_batch


def profile_feats_batch(rho_all, Sigma_invs):
    """rho_all: (N, 3), Sigma_invs: (N, 3, 3) -> (N, 7)."""
    N, K = len(rho_all), len(WS_PROFILE)
    ell = np.zeros((N, K))
    for k, ws in enumerate(WS_PROFILE):
        X_k = design_matrix(ws)
        ATA = np.einsum("ia,nij,jb->nab", X_k, Sigma_invs, X_k)
        ATz = np.einsum("ia,nij,nj->na", X_k, Sigma_invs, rho_all)
        beta = nnls_2d_batch(ATA, ATz)
        e = rho_all - np.einsum("na,da->nd", beta, X_k)
        ell[:, k] = np.einsum("ni,nij,nj->n", e, Sigma_invs, e)

    ell_min = ell.min(1, keepdims=True)
    w_map = WS_PROFILE[ell.argmin(1)]

    in_ci = ell <= ell_min + CHI2_95
    first = np.argmax(in_ci, 1)
    last = (K - 1) - np.argmax(in_ci[:, ::-1], 1)
    profile_width = WS_PROFILE[last] - WS_PROFILE[first]

    p = np.exp(-0.5 * (ell - ell_min))
    p /= p.sum(1, keepdims=True)
    mid = (WS_PROFILE >= 0.35) & (WS_PROFILE <= 0.65)
    middle_mass = p[:, mid].sum(1)

    w_mean = p @ WS_PROFILE
    cdf = np.cumsum(p, axis=1)
    med_idx = np.argmax(cdf >= 0.5, axis=1)
    w_median = WS_PROFILE[med_idx]
    entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
    eff_count = np.exp(entropy)
    map_mean_gap = np.abs(w_map - w_mean)

    return np.column_stack([w_map, profile_width, middle_mass,
                            w_mean, w_median, eff_count, map_mean_gap])
