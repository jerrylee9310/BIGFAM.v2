"""Simulated data for offline Phase 2 training (not used at inference time).

    rho_true_d = 0.5^d * V_G + w_S^{d-1} * V_S,   d = 1, 2, 3
    rho_hat    ~ N(rho_true, Sigma)

The shipped calibration is trained on a deliberately wide draw so it stays
usable across traits: w_S ~ U(0.01, 0.99), (V_G, V_S) ~ Dirichlet(1, 1, 1),
per-DOR sd ~ U(0.001, SIGMA_HI), and Sigma = diag(sd) R diag(sd) with R a free
positive correlation matrix (resampled until PSD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED, N_SIM, SIGMA_HI
from .features import compute_feature_dict, FEAT_ALL


def sample_rho_hats(rho_true, Sigmas, rng):
    """Draw rho_hat ~ N(rho_true, Sigma) per row. rho_true: (N, 3) -> (N, 3)."""
    L = np.linalg.cholesky(Sigmas)
    z = rng.standard_normal((len(rho_true), 3))
    return rho_true + np.einsum("nij,nj->ni", L, z)


def rho_true_batch(V_G, V_S, w_S):
    """Noise-free similarity curve. V_G, V_S, w_S each (N,) -> (N, 3).

    rho_d = 0.5^d * V_G + w_S^(d-1) * V_S,  d = 1, 2, 3.
    """
    V_G = np.asarray(V_G, float)
    V_S = np.asarray(V_S, float)
    w_S = np.asarray(w_S, float)
    d = np.array([1, 2, 3])
    return (0.5 ** d)[None, :] * V_G[:, None] + (w_S[:, None] ** (d - 1)[None, :]) * V_S[:, None]


# ── scenario draw ────────────────────────────────────────────────────────────
def _psd_ok(r):
    """r: (N, 3) = [r12, r13, r23]. True where the correlation matrix is PSD."""
    a, b, c = r[:, 0], r[:, 1], r[:, 2]
    return (1.0 + 2.0 * a * b * c - a * a - b * b - c * c) > 1e-6


def _free_corr(rng, N):
    """3 free positive off-diagonals [r12,r13,r23] ~ U(0,0.9), rejection to PSD."""
    r = rng.uniform(0.0, 0.9, (N, 3))
    bad = ~_psd_ok(r)
    while bad.any():
        r[bad] = rng.uniform(0.0, 0.9, (int(bad.sum()), 3))
        bad = ~_psd_ok(r)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1.0
    R[:, 0, 1] = R[:, 1, 0] = r[:, 0]
    R[:, 0, 2] = R[:, 2, 0] = r[:, 1]
    R[:, 1, 2] = R[:, 2, 1] = r[:, 2]
    return R


def sample_scenarios(rng, N):
    """Draw N scenarios -> (w_S, V_G, V_S, Sigmas).

    The draw ORDER (w, Dirichlet fold, correlation, sd) fixes what a given seed
    produces -- reordering it changes the shipped artifact.
    """
    w = rng.uniform(0.01, 0.99, N)

    u = rng.uniform(0.0, 1.0, N)                        # Dirichlet(1,1,1) on (V_G,V_S)
    v = rng.uniform(0.0, 1.0, N)
    m = (u + v) > 1.0
    u[m], v[m] = 1.0 - u[m], 1.0 - v[m]
    V_G, V_S = u, v

    R = _free_corr(rng, N)
    sd = rng.uniform(0.001, SIGMA_HI, (N, 3))           # unordered
    Sigmas = R * sd[:, :, None] * sd[:, None, :]
    return w, V_G, V_S, Sigmas


def generate_training_frame(seed=SEED, n=N_SIM) -> pd.DataFrame:
    """The training frame ws_calibration.json is fit on.

    Columns: the 24 features, plus w_S_true (the ridge target) and V_G / V_S.
    """
    rng = np.random.default_rng(seed)
    w, V_G, V_S, Sigmas = sample_scenarios(rng, n)
    rho_true = rho_true_batch(V_G, V_S, w)
    rho_hat = sample_rho_hats(rho_true, Sigmas, rng)
    feats = compute_feature_dict(rho_hat, Sigmas)

    df = pd.DataFrame({name: feats[name] for name in FEAT_ALL})
    df["w_S_true"] = w
    df["V_G"], df["V_S"] = V_G, V_S
    return df
