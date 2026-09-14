"""Design matrix X(w_C) for the fixed-w_C decomposition (D = 3, fixed).

    rho_d = 0.5^d * V_A + w_C^{d-1} * V_C,  d = 1, 2, 3

        X(w_C) = [[0.5,   1   ],
                  [0.25,  w_C ],
                  [0.125, w_C^2]]
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


def design_matrix(w_c: float) -> np.ndarray:
    """Single w_C -> X: (3, 2)."""
    return design_matrix_batch(np.array([float(w_c)]))[0]
