"""Phase 1 binary internals, each checked against a slower reference:

  - bvn_cdf (vectorized Gauss-Legendre) vs scipy's mvn CDF
  - neg_log_lik_grad (analytical) vs a finite-difference gradient
  - cluster_meat (factorize + bincount) vs a naive groupby loop
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.stats import multivariate_normal as mvn

from bigfam.phase1 import binary as B
from bigfam.phase1.sandwich import cluster_meat

_P1 = 5  # intercept + 4 covariates — toy-data shape for grad/meat tests


# ── bvn_cdf vs scipy ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("r", [-0.95, -0.5, -0.05, 0.0, 0.05, 0.5, 0.9, 0.95])
def test_bvn_cdf_matches_scipy(r):
    rng = np.random.default_rng(0)
    h = rng.uniform(-3, 3, 500)
    k = rng.uniform(-3, 3, 500)
    mine = B.bvn_cdf(h, k, np.full(500, r))
    if r == 0.0:
        ref = mvn(mean=[0, 0], cov=[[1, 0], [0, 1]]).cdf(np.column_stack([h, k]))
    else:
        ref = mvn(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf(np.column_stack([h, k]))
    assert np.abs(mine - ref).max() < 1e-8


# ── analytical jac vs finite difference ────────────────────────────────────────
def _toy_binary(n=400, seed=1):
    rng = np.random.default_rng(seed)
    W1 = np.column_stack([np.ones(n), rng.normal(size=(n, _P1 - 1))])
    W2 = np.column_stack([np.ones(n), rng.normal(size=(n, _P1 - 1))])
    y1 = rng.integers(0, 2, n).astype(float)
    y2 = rng.integers(0, 2, n).astype(float)
    g = rng.integers(1, 4, n)
    return W1, W2, y1, y2, g


def test_neg_log_lik_grad_matches_numeric():
    W1, W2, y1, y2, g = _toy_binary()
    theta = np.concatenate([np.zeros(_P1), [0.3, 0.15, 0.05]])
    err = check_grad(
        lambda t: B.neg_log_lik(t, W1, W2, y1, y2, g, 3),
        lambda t: B.neg_log_lik_grad(t, W1, W2, y1, y2, g, 3),
        theta,
    )
    assert err < 1e-4


# ── cluster_meat: vectorized == naive groupby loop ─────────────────────────────
def _naive_meat(scores, id1, id2):
    dim = scores.shape[1]
    df = pd.DataFrame(scores, columns=[f"s{i}" for i in range(dim)])
    df["id1"], df["id2"] = id1, id2
    cols = [f"s{i}" for i in range(dim)]

    def meat(key):
        M = np.zeros((dim, dim))
        for _, grp in df.groupby(key):
            s = grp[cols].values.sum(axis=0)
            M += np.outer(s, s)
        return M

    return meat("id1") + meat("id2") - meat(["id1", "id2"])


def test_cluster_meat_matches_naive():
    rng = np.random.default_rng(2)
    n, dim = 3000, 8
    scores = rng.normal(size=(n, dim))
    id1 = rng.integers(0, 400, n)
    id2 = rng.integers(0, 400, n)
    np.testing.assert_allclose(
        cluster_meat(scores, id1, id2), _naive_meat(scores, id1, id2), atol=1e-8)
