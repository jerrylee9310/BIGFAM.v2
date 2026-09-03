"""Phase 3 Step 1-2: fixed-w_S GLS/NNLS refit + A^-1 (rho-noise SE).

With w_S held fixed the model is linear in (V_G, V_S):

    rho = X(w_S) [V_G, V_S]^T,   beta = NNLS( X^T Sigma^-1 X, X^T Sigma^-1 rho )

A relative ridge 1e-9*tr(A)*I stabilises the w_S=0.5 singular case (where the
two columns of X become proportional, c_S = 2 c_G) and is negligible elsewhere.
"""
from __future__ import annotations

import numpy as np

from ..config import RIDGE_REFIT
from ..core.design import design_matrix_batch
from ..core.nnls import nnls_2d_batch


def refit_batch(w_vec, rho, Sigmas):
    """Batch fixed-w_S refit.

    w_vec: (N,), rho: (N, 3), Sigmas: (N, 3, 3)
    -> beta (N, 2), se (N, 2) [A^-1 rho-noise SE], Ainv (N, 2, 2), dbeta (N, 2).
    """
    w_vec = np.asarray(w_vec, dtype=float)
    rho = np.asarray(rho, dtype=float)
    Sigmas = np.asarray(Sigmas, dtype=float)
    Sinv = np.linalg.inv(Sigmas)
    X = design_matrix_batch(w_vec)                       # (N, 3, 2)

    ATA = np.einsum("nia,nij,njb->nab", X, Sinv, X)      # X^T S^-1 X
    ATz = np.einsum("nia,nij,nj->na", X, Sinv, rho)      # X^T S^-1 rho
    tr = np.einsum("nii->n", ATA)
    ATAr = ATA + (RIDGE_REFIT * tr)[:, None, None] * np.eye(2)[None]
    beta = nnls_2d_batch(ATAr, ATz)
    Ainv = np.linalg.inv(ATAr)
    se = np.sqrt(np.maximum(np.stack([Ainv[:, 0, 0], Ainv[:, 1, 1]], 1), 0.0))

    # analytical Jacobian d beta / d w_S (diagnostic only; NOT used for SE)
    Xp = np.zeros_like(X); Xp[:, 1, 1] = 1.0; Xp[:, 2, 1] = 2.0 * w_vec
    resid = rho - np.einsum("nia,na->ni", X, beta)
    term1 = np.einsum("nia,nij,nj->na", Xp, Sinv, resid)
    XtSXp = np.einsum("nia,nij,njb->nab", X, Sinv, Xp)
    term2 = np.einsum("nab,nb->na", XtSXp, beta)
    dbeta = np.einsum("nab,nb->na", Ainv, term1 - term2)
    return beta, se, Ainv, dbeta


def refit_fixed_ws(rho_hat, Sigma_hat, w_s):
    """Single-family refit. rho_hat: (3,), Sigma_hat: (3, 3), w_s: float.

    -> beta (2,), se (2,), Ainv (2, 2), dbeta (2,).
    """
    beta, se, Ainv, dbeta = refit_batch(
        np.array([float(w_s)]), rho_hat[None], Sigma_hat[None]
    )
    return beta[0], se[0], Ainv[0], dbeta[0]
