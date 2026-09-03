"""Profile-likelihood 95% CI for w_S.

For each w on the WS_PROFILE grid, fit the same fixed-w NNLS as refit.py and read
the loss ell = e^T Sigma^-1 e. The CI is {w : ell <= ell_min + CHI2_95}.

Read identifiability off this interval: at w_S = 0.5 the shared-environment column
of X(w) is exactly twice the genetic one, so the design is rank 1 and (V_G, V_S)
cannot be separated. A CI excluding 0.5 is therefore the likelihood-ratio test of
H0: w_S = 0.5. No pass/fail label is emitted -- where to cut the interval is the
analysis's call.
"""
from __future__ import annotations

import numpy as np

from ..config import WS_PROFILE, CHI2_95
from ..core.nnls import nnls_2d


def profile_ci(rho_hat, Sigma_hat):
    """rho_hat: (3,), Sigma_hat: (3, 3) -> (wci_lo, wci_hi)."""
    rho_hat = np.asarray(rho_hat, dtype=float)
    Sinv = np.linalg.inv(np.asarray(Sigma_hat, dtype=float))
    K = len(WS_PROFILE)
    ell = np.zeros(K)
    for k, ws in enumerate(WS_PROFILE):
        X = np.array([[0.5, 1.0], [0.25, ws], [0.125, ws * ws]])
        beta = nnls_2d(X.T @ Sinv @ X, X.T @ Sinv @ rho_hat)
        e = rho_hat - X @ beta
        ell[k] = e @ Sinv @ e

    in_ci = ell <= ell.min() + CHI2_95
    lo = WS_PROFILE[in_ci.argmax()]
    hi = WS_PROFILE[K - 1 - in_ci[::-1].argmax()]
    return float(lo), float(hi)

