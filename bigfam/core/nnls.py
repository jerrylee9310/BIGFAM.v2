"""Proper 2-variable NNLS via KKT active-set.

Shared by Phase 2 profile loss and Phase 3 refit: Phase 3's decomposition at
w_S = w_hat must equal Phase 2's profile curve read at the same point, which
holds by construction because both call the *same* solver. Inputs are in
normal-equation form (ATA, ATz) so batch and single-family share one kernel.
"""
from __future__ import annotations

import numpy as np


def nnls_2d_batch(ATA: np.ndarray, ATz: np.ndarray) -> np.ndarray:
    """Solve min_{b>=0} b^T ATA b - 2 ATz^T b for a batch.

    ATA: (N, 2, 2)  = X^T Sigma^-1 X
    ATz: (N, 2)     = X^T Sigma^-1 rho
    returns beta: (N, 2), non-negative.
    """
    reg = ATA + 1e-12 * np.eye(2)[None]
    beta = np.linalg.solve(reg, ATz[:, :, None])[:, :, 0]
    needs = (beta[:, 0] < 0) | (beta[:, 1] < 0)
    for n in np.where(needs)[0]:
        ata, atz = ATA[n], ATz[n]
        b1 = max(atz[1] / max(ata[1, 1], 1e-12), 0.0)
        b0 = max(atz[0] / max(ata[0, 0], 1e-12), 0.0)
        cand = [np.array([0.0, b1]), np.array([b0, 0.0]), np.zeros(2)]
        cost = [c @ ata @ c - 2.0 * atz @ c for c in cand]
        beta[n] = cand[int(np.argmin(cost))]
    return np.maximum(beta, 0.0)


def nnls_2d(ATA: np.ndarray, ATz: np.ndarray) -> np.ndarray:
    """Single-family NNLS. ATA: (2, 2), ATz: (2,) -> beta: (2,)."""
    return nnls_2d_batch(ATA[None], ATz[None])[0]
