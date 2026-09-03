"""Binary phenotype: joint bivariate probit MLE -> (rho_hat, Sigma_hat).

theta = (gamma [1+p covariate coefficients], rho [D]). Fit by L-BFGS-B with an
analytical gradient, then a cluster sandwich covariance whose rho block is
Sigma_hat.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.stats import norm

from ..config import RHO_CLIP, PROB_FLOOR, HESS_EPS, BVN_GL_NODES
from .sandwich import cluster_meat

# Gauss-Legendre nodes/weights for the vectorized bivariate-normal CDF below.
_GL_X, _GL_W = leggauss(BVN_GL_NODES)


def make_arrays(data, cov_cols):
    def W(prefix):
        X = data[[f"{prefix}_{c}" for c in cov_cols]].values
        return np.column_stack([np.ones(len(data)), X])
    return (
        W("x1"), W("x2"),
        data["y1"].values, data["y2"].values, data["dor"].values,
        data["id1"].values, data["id2"].values,
    )


def bvn_cdf(a, b, r):
    """Vectorized bivariate-normal CDF Phi_2(a, b; r), r per-row, in [-1, 1].

    Uses the integral identity
        Phi_2(a,b;r) = Phi(a)Phi(b)
                     + 1/(2pi) integral_0^r exp(-(a^2-2t a b+b^2)/(2(1-t^2)))
                       / sqrt(1-t^2) dt
    with Gauss-Legendre quadrature on [0, r]. Matches scipy's mvn CDF to ~2e-10
    over |r| <= 0.95 (config.BVN_GL_NODES) and, unlike scipy's, is vectorized
    over rows -- the likelihood is evaluated on every pair at every iteration.
    """
    a = np.asarray(a, float); b = np.asarray(b, float); r = np.asarray(r, float)
    base = norm.cdf(a) * norm.cdf(b)
    t = r[:, None] / 2.0 * (_GL_X + 1.0)[None, :]        # (n, nodes) on [0, r]
    one_minus = 1.0 - t * t
    integ = np.exp(
        -(a[:, None] ** 2 - 2.0 * t * a[:, None] * b[:, None] + b[:, None] ** 2)
        / (2.0 * one_minus)
    ) / np.sqrt(one_minus)
    return base + (r / 2.0) * (integ @ _GL_W) / (2.0 * np.pi)


def bvn_log_prob_by_group(a, b, q12, rho_d):
    """Per-row log Phi_2(a, b; r), r = q1*q2*rho_d (= q12*rho_d), clipped."""
    r = np.clip(q12 * rho_d, *RHO_CLIP)
    probs = bvn_cdf(a, b, r)
    return np.log(np.clip(probs, PROB_FLOOR, None))


def log_lik_rows(W1, W2, y1, y2, g, gamma, rho, D=3):
    """Row-level log-likelihood for the bivariate probit."""
    q1 = 2 * y1 - 1
    q2 = 2 * y2 - 1
    mu1 = W1 @ gamma
    mu2 = W2 @ gamma
    a = q1 * mu1
    b = q2 * mu2
    q12 = (q1 * q2).astype(int)

    log_probs = np.empty(len(y1))
    for d in range(1, D + 1):
        mask_d = (g == d)
        if not mask_d.any():
            continue
        log_probs[mask_d] = bvn_log_prob_by_group(
            a[mask_d], b[mask_d], q12[mask_d], rho[d - 1]
        )
    return log_probs


def neg_log_lik(theta, W1, W2, y1, y2, g, D=3):
    p1 = W1.shape[1]
    gamma = theta[:p1]
    rho = theta[p1:]
    return -log_lik_rows(W1, W2, y1, y2, g, gamma, rho, D).sum()


def neg_log_lik_grad(theta, W1, W2, y1, y2, g, D=3):
    """Analytical gradient of neg_log_lik = -sum row_scores.

    Passed to the optimizer as jac, so it skips finite differences (~8x fewer
    likelihood evaluations).
    """
    p1 = W1.shape[1]
    return -row_scores(W1, W2, y1, y2, g, theta[:p1], theta[p1:], D).sum(axis=0)


def row_scores(W1, W2, y1, y2, g, gamma, rho, D=3):
    """Analytical row score s_n = d(log P_n)/d(theta), shape (N, p1+D)."""
    p1 = W1.shape[1]
    q1 = 2 * y1 - 1
    q2 = 2 * y2 - 1
    mu1 = W1 @ gamma
    mu2 = W2 @ gamma
    a = q1 * mu1
    b = q2 * mu2
    q12 = (q1 * q2).astype(int)

    log_probs = log_lik_rows(W1, W2, y1, y2, g, gamma, rho, D)
    probs = np.exp(log_probs)

    scores = np.zeros((len(y1), p1 + D))
    for d in range(1, D + 1):
        for s in [1, -1]:
            mask = (g == d) & (q12 == s)
            if not mask.any():
                continue
            r = np.clip(float(s) * rho[d - 1], *RHO_CLIP)
            sqrt_1mr2 = np.sqrt(1 - r ** 2)

            a_m = a[mask]; b_m = b[mask]
            q1_m = q1[mask]; q2_m = q2[mask]
            p_m = probs[mask]

            d_da = norm.pdf(a_m) * norm.cdf((b_m - r * a_m) / sqrt_1mr2) / p_m
            d_db = norm.pdf(b_m) * norm.cdf((a_m - r * b_m) / sqrt_1mr2) / p_m

            phi2 = (1 / (2 * np.pi * sqrt_1mr2)) * np.exp(
                -(a_m ** 2 - 2 * r * a_m * b_m + b_m ** 2) / (2 * (1 - r ** 2))
            )
            d_drho_d = phi2 / p_m * float(s)

            scores[mask, :p1] += (d_da * q1_m)[:, None] * W1[mask] \
                + (d_db * q2_m)[:, None] * W2[mask]
            scores[mask, p1 + d - 1] += d_drho_d
    return scores


def numerical_hessian(theta_hat, W1, W2, y1, y2, g, D=3, eps=HESS_EPS):
    """Negative Hessian of total log-lik via central differences of the gradient."""
    p1 = W1.shape[1]
    dim = len(theta_hat)
    H = np.zeros((dim, dim))

    def total_score(t):
        return row_scores(W1, W2, y1, y2, g, t[:p1], t[p1:], D).sum(axis=0)

    for i in range(dim):
        e = np.zeros(dim); e[i] = eps
        H[:, i] = -(total_score(theta_hat + e) - total_score(theta_hat - e)) / (2 * eps)
    return (H + H.T) / 2


def estimate_rho_sigma(data, cov_cols, D=3):
    """Full binary path on a flipped&concat table -> (rho_hat, Sigma_hat)."""
    W1, W2, y1, y2, g, id1, id2 = make_arrays(data, cov_cols)
    p1 = W1.shape[1]

    theta0 = np.zeros(p1 + D)
    theta0[p1:] = [0.40, 0.15, 0.05][:D]
    bounds = [(None, None)] * p1 + [RHO_CLIP] * D

    res = minimize(
        neg_log_lik, theta0,
        args=(W1, W2, y1, y2, g, D),
        jac=neg_log_lik_grad,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
    )
    theta_hat = res.x
    rho_hat = theta_hat[p1:]

    scores = row_scores(W1, W2, y1, y2, g, theta_hat[:p1], rho_hat, D)
    B = numerical_hessian(theta_hat, W1, W2, y1, y2, g, D)
    M = cluster_meat(scores, id1, id2)
    B_inv = np.linalg.inv(B)
    Var_hat = B_inv @ M @ B_inv          # (p1+D, p1+D)
    Sigma_hat = Var_hat[p1:, p1:]        # rho block (D, D)
    return rho_hat, Sigma_hat, res
