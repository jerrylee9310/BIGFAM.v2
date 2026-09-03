"""Fixed-w_S refit: (rho_hat, Sigma_hat, w_S) -> (V_G, V_S) and their conditional SE.

With w_S held fixed the model is linear in (V_G, V_S):

    rho = X(w_S) [V_G, V_S]^T,   beta = NNLS( X^T Sigma^-1 X, X^T Sigma^-1 rho )

so the conditional SE is sqrt(diag(A^-1)) with A = X^T Sigma^-1 X. A relative
ridge (RIDGE_REFIT * tr A) keeps A invertible at w_S = 0.5, where the two
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


def refit_fixed_ws(rho_hat, Sigma_hat, w_s):
    """Single family. rho_hat: (3,), Sigma_hat: (3, 3), w_s: float -> beta, se each (2,)."""
    beta, se = refit_batch(np.array([float(w_s)]), rho_hat[None], Sigma_hat[None])
    return beta[0], se[0]
