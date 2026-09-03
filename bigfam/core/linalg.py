"""Safe linear-algebra helpers for the (near-singular at w_S=0.5) GLS solves."""
from __future__ import annotations

import numpy as np


def safe_inv(M: np.ndarray) -> np.ndarray:
    """Inverse that tolerates near-singular matrices via pinv fallback."""
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def nearest_psd(M: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """Nearest symmetric positive-definite matrix by eigenvalue clipping.

    The cluster sandwich M = M_1 + M_2 - M_12 is a difference of PSD matrices,
    so Sigma_hat = B^-1 M B^-1 can be indefinite (negative eigenvalues) on
    high-overlap / low-signal data. Downstream consumers assume it is PD:
    np.linalg.cholesky (per-case sampling) hard-crashes, and np.linalg.inv (the
    feature GLS solves) silently returns an indefinite inverse. Clipping the
    eigenvalues to a small positive floor is the Frobenius-nearest PD repair
    (Higham): floor = rtol * max_eig keeps the condition number bounded so the
    matrix stays invertible. A PD input is returned unchanged.
    """
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)                       # symmetrize away fp asymmetry
    w, V = np.linalg.eigh(M)
    floor = max(float(w[-1]), 0.0) * rtol
    if floor <= 0.0:                          # fully degenerate (all eig <= 0)
        floor = np.finfo(float).tiny
    if w[0] >= floor:
        return M                              # already PD at this tolerance
    w = np.clip(w, floor, None)
    return (V * w) @ V.T
