"""Fixed-w_C refit: (rho_hat, Sigma_hat, w_C) -> (V_A, V_C) and their conditional SE.

With w_C held fixed the model is linear in (V_A, V_C):

    rho = X(w_C) [V_A, V_C]^T,   beta = NNLS( X^T Sigma^-1 X, X^T Sigma^-1 rho )

so the conditional SE is sqrt(diag(A^-1)) with A = X^T Sigma^-1 X. A relative
ridge (RIDGE_REFIT * tr A) keeps A invertible at w_C = 0.5, where the two
columns of X become proportional; elsewhere it is negligible.
"""
from __future__ import annotations

import numpy as np

from ..config import RIDGE_REFIT
from ..core.design import design_matrix_batch
from ..core.nnls import nnls_2d_batch


def refit_batch(w_vec, rho, Sigmas):
    """Batch refit. w_vec: (N,), rho: (N, 3), Sigmas: (N, 3, 3) -> beta, se each (N, 2)."""
    rho = np.asarray(rho, dtype=float)
    Sinv = np.linalg.inv(np.asarray(Sigmas, dtype=float))
    X = design_matrix_batch(w_vec)                       # (N, 3, 2)

    A = np.einsum("nia,nij,njb->nab", X, Sinv, X)        # X^T Sigma^-1 X
    ATz = np.einsum("nia,nij,nj->na", X, Sinv, rho)      # X^T Sigma^-1 rho
    A = A + (RIDGE_REFIT * np.einsum("nii->n", A))[:, None, None] * np.eye(2)[None]

    beta = nnls_2d_batch(A, ATz)
    A_inv = np.linalg.inv(A)
    se = np.sqrt(np.maximum(np.stack([A_inv[:, 0, 0], A_inv[:, 1, 1]], 1), 0.0))
    return beta, se


def refit_fixed_wc(rho_hat, Sigma_hat, w_c):
    """Single family. rho_hat: (3,), Sigma_hat: (3, 3), w_c: float -> beta, se each (2,)."""
    beta, se = refit_batch(np.array([float(w_c)]), rho_hat[None], Sigma_hat[None])
    return beta[0], se[0]
