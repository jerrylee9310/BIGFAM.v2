"""Cluster sandwich covariance (shared by continuous and binary Phase 1).

An individual appears in many pairs; ignoring that repetition pretends there is
more data than there is and shrinks the SE. The cluster meat corrects it:

    M = M_1 + M_2 - M_12,   Sigma_hat = B^-1 M B^-1

M_1 clusters on id1, M_2 on id2, M_12 on the (id1, id2) pair. Only the score
definition and the bread B differ between continuous/binary; this is common.

Reference: phase1_continuous.py / phase1_binary.py cluster_meat (verbatim).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _meat_by_codes(scores: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """sum_c (sum_{n in c} s_n)(.)^T = S^T S, S = per-cluster score sums.

    codes: contiguous group labels 0..G-1. Replaces a per-group Python loop
    (groupby + np.outer) with one bincount per score column, then a single
    matmul -- the cluster-meat groupby was the Phase 1 bottleneck after the
    likelihood was vectorized.
    """
    dim = scores.shape[1]
    G = int(codes.max()) + 1 if codes.size else 0
    S = np.empty((G, dim))
    for j in range(dim):
        S[:, j] = np.bincount(codes, weights=scores[:, j], minlength=G)
    return S.T @ S


def cluster_meat(scores: np.ndarray, id1: np.ndarray, id2: np.ndarray) -> np.ndarray:
    """M = M_1 + M_2 - M_12. scores: (N, dim) -> M: (dim, dim)."""
    c1, _ = pd.factorize(id1)
    c2, _ = pd.factorize(id2)
    pair = c1.astype(np.int64) * (int(c2.max()) + 1) + c2 if c2.size else c1
    cpair, _ = pd.factorize(pair)
    return (_meat_by_codes(scores, c1)
            + _meat_by_codes(scores, c2)
            - _meat_by_codes(scores, cpair))


def sandwich_cov(B: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Sigma = B^-1 M B^-1."""
    B_inv = np.linalg.inv(B)
    return B_inv @ M @ B_inv
