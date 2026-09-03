"""Design matrix X(w_S) for the fixed-w_S decomposition (D = 3, fixed).

    rho_d = 0.5^d * V_G + w_S^{d-1} * V_S,  d = 1, 2, 3

        X(w_S) = [[0.5,   1   ],
                  [0.25,  w_S ],
                  [0.125, w_S^2]]
"""
from __future__ import annotations

import numpy as np


def design_matrix_batch(w_vec: np.ndarray) -> np.ndarray:
    """w_vec: (N,) -> X: (N, 3, 2)."""
    w_vec = np.asarray(w_vec, dtype=float)
    N = len(w_vec)
    X = np.zeros((N, 3, 2))
    X[:, 0, 0] = 0.5
    X[:, 0, 1] = 1.0
    X[:, 1, 0] = 0.25
    X[:, 1, 1] = w_vec
    X[:, 2, 0] = 0.125
    X[:, 2, 1] = w_vec ** 2
    return X


def design_matrix(w_s: float) -> np.ndarray:
    """Single w_S -> X: (3, 2)."""
    return design_matrix_batch(np.array([float(w_s)]))[0]
