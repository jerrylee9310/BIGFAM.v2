"""Phase 2 training simulation DGP (offline only).

    rho_true_d = 0.5^d * V_G + w_S^{d-1} * V_S,   d = 1, 2, 3
    rho_hat    ~ N(rho_true, Sigma)   (cholesky)

The WIDE step1 space (generate_training_frame_wide) is what ws_calibration.json
is trained on: w_S ~ U(0.01, 0.99) continuous, (V_G,V_S) ~ Dirichlet(1,1,1),
sd ~ U(0.001, 0.10) unordered, Sigma = diag(sd) R diag(sd) with R a free 3-param
positive correlation (rejection-sampled to PSD). See scratch step2a/2b. The mean/
noise primitives (rho_true_batch, m_batch, sample_rho_hats) are DGP-agnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED, N_SIM, SIGMA_HI
from .features import compute_feature_dict, FEAT_ALL


def sample_rho_hats(rho_true, Sigmas, rng):
    L = np.linalg.cholesky(Sigmas)
    z = rng.standard_normal((len(rho_true), 3))
    return rho_true + np.einsum("nij,nj->ni", L, z)


def rho_true_batch(w_S, V_G, V_S):
    d = np.arange(1, 4)
    return 0.5 ** d[None, :] * V_G[:, None] + w_S ** (d - 1)[None, :] * V_S[:, None]


def m_batch(V_G, V_S, w_S):
    """Per-sample mean signal. V_G, V_S, w_S each (N,) -> (N, 3).

    m[d] = 0.5^d * V_G + w_S^(d-1) * V_S,  d = 1, 2, 3.
    Vector-w_S twin of rho_true_batch (which takes a scalar w_S).
    """
    V_G = np.asarray(V_G, float)
    V_S = np.asarray(V_S, float)
    w_S = np.asarray(w_S, float)
    d = np.array([1, 2, 3])
    return (0.5 ** d)[None, :] * V_G[:, None] + (w_S[:, None] ** (d - 1)[None, :]) * V_S[:, None]


# ── WIDE DGP (shipped artifact) ──────────────────────────────────────────────
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


def sample_scenarios_wide(rng, N):
    """Wide step1-style draw. Returns (w, V_G, V_S, Sigmas). Draw ORDER is the
    frozen contract (w, Dirichlet fold, free corr, sd) -- do not reorder."""
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


def generate_training_frame_wide(seed=SEED, n=N_SIM) -> pd.DataFrame:
    """Wide-DGP training frame (24 features + w_S_true + V_G/V_S, all 'train').

    This is the frame ws_calibration.json is fit on. Reproduces the scratch
    step2 prototype exactly (seed 42, N 40k) -- same rng order, same features.
    """
    rng = np.random.default_rng(seed)
    w, V_G, V_S, Sigmas = sample_scenarios_wide(rng, n)
    rho_true = m_batch(V_G, V_S, w)
    rho_hat = sample_rho_hats(rho_true, Sigmas, rng)
    feats = compute_feature_dict(rho_hat, Sigmas)

    df = pd.DataFrame({name: feats[name] for name in FEAT_ALL})
    df["w_S_true"] = w
    df["V_G"], df["V_S"] = V_G, V_S
    df["split"] = "train"
    return df
