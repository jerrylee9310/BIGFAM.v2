"""BIGFAM.v1 -- faithful obj1.py band machine (decay class, the earlier estimator).

Same decay class as BIGFAM.v2 (estimates its own w_S) but the old approach:
per-DOR resample -> a slope test classifies the decay shape -> that class fixes
the w search BAND (_setInitialRange) -> nested CV inside the band picks w by
held-out mse, and (V_G,V_S) are fit by equal-weight log2-SSE -> median over
folds. The fit never uses Sigma (off-diagonal unused); only the per-DOR SE
(the diagonal) enters, via the resample + slope test.

v1 fidelity (band machine vs. the original obj1.py) is verified in
pjt-bf's research/bigfam-benchmark/scripts/verify_v1_band.py (not included
here -- this folder ships the estimator, not its independent verification).

Two entry points:
  v1_batch(rho_obs, Sig, rng)  batch -- callers plant the full Sigma and only
                                the diagonal is read (signature kept as-is)
  make(seed) -> estimate(rho)  single-trait, for this folder's run.py. rho:
                                anything exposing .rho_hat and .sigma_hat
                                (per-DOR SE). v1 is stochastic, so seed is fixed.
"""
from __future__ import annotations

import numpy as np

DOR = np.array([1, 2, 3])
LN2 = np.log(2.0)
V1_SEED = 20260704                       # default seed for this folder's driver


# ── equal-weight log2-OLS fit at fixed w (obj1 `_lossFunc`) ────────────────────
def _fit_at_w(logrho, w, iters=12):
    """min over (V_G,V_S) of  sum_d (log2 rho_d - log2[0.5^d V_G + w^(d-1) V_S])^2  (Gauss-Newton).
    Reproduces obj1.py `_lossFunc` (log2 SSE, Sigma ignored); GN solves the identical objective."""
    N = len(logrho)
    cG = 0.5 ** DOR
    cS = w ** (DOR - 1)
    G = np.full(N, 0.5); S = np.full(N, 0.1)
    for _ in range(iters):
        m = np.maximum(cG[None] * G[:, None] + cS[None] * S[:, None], 1e-9)
        r = logrho - np.log2(m)
        JG = cG[None] / (m * LN2); JS = cS[None] / (m * LN2)
        a = (JG * JG).sum(1); b = (JG * JS).sum(1); c = (JS * JS).sum(1)
        gG = (JG * r).sum(1); gS = (JS * r).sum(1)
        det = a * c - b * b + 1e-12
        G = np.clip(G + (c * gG - b * gS) / det, 1e-6, 1.0)
        S = np.clip(S + (a * gS - b * gG) / det, 1e-6, 1.0)
    m = np.maximum(cG[None] * G[:, None] + cS[None] * S[:, None], 1e-9)
    return G, S, ((logrho - np.log2(m)) ** 2).sum(1)


WS_STEP = 0.01
BANDS = {                                               # obj1 `_setInitialRange`: slope-class -> band
    "None": np.arange(0.40, 0.60 + WS_STEP, WS_STEP),   # similar decay (slope CI brackets 1)
    "High": np.arange(0.01, 0.45 + WS_STEP, WS_STEP),   # fast decay    (slope CI > 1)
    "Low":  np.arange(0.55, 0.95 + WS_STEP, WS_STEP),   # slow decay    (slope CI < 1)
}
CLASS_NAME = {"High": "fast", "Low": "slow", "None": "similar"}   # slope-class -> report name


def _slope_test(L):
    """obj1 `_slopeSig`: per resample, decay-slope of log2(rho) on -DOR; classify by 95% CI vs 1.
    OLS slope on the 3 equally-spaced DORs uses only the endpoints -> slope = (log2 s_1 - log2 s_3)/2.
    slope = 1 <=> pure genetic halving (rho_d ~ 0.5^d). lower>1 -> 'High' (fast), upper<1 -> 'Low'
    (slow), else 'None' (similar)."""
    slopes = (L[:, :, 0] - L[:, :, 2]) / 2.0                            # (N, n_resample)
    lower = np.percentile(slopes, 2.5, axis=1)
    upper = np.percentile(slopes, 97.5, axis=1)
    return np.where(lower > 1, "High", np.where(upper < 1, "Low", "None"))


def _nested_cv(L, band, rng, n_repeat, n_block):
    """obj1 `prediction`: n_repeat x n_block CV over the resamples; per fold fit (G,S) on train,
    pick w by min TEST mse, median (G,S,w) over folds. log2-SSE over resample rows collapses to
    per-DOR means, so a fold reduces to (train_mean_d, test_mean_d) and the w-independent SS-within
    drops out of the argmin."""
    Ng, R, _ = L.shape
    per = R // n_block
    overall = L.mean(1)                                                # (Ng, 3) mean over all resamples
    trains, tests = [], []
    for _ in range(n_repeat):
        perm = np.argsort(rng.standard_normal((Ng, R, 3)), axis=1)     # random block partition per DOR
        blk = np.take_along_axis(L, perm, axis=1).reshape(Ng, n_block, per, 3)
        bm = blk.mean(2)                                               # (Ng, n_block, 3) test means
        tests.append(bm)
        trains.append((R * overall[:, None, :] - per * bm) / (R - per))  # leave-block-out train mean
    train = np.concatenate(trains, 1).reshape(-1, 3)
    test = np.concatenate(tests, 1).reshape(-1, 3)
    M = len(train)
    best = np.full(M, np.inf); bG = np.zeros(M); bS = np.zeros(M); bW = np.zeros(M)
    for w in band:
        g, s, _ = _fit_at_w(train, w)
        m = np.maximum((0.5 ** DOR)[None] * g[:, None] + (w ** (DOR - 1))[None] * s[:, None], 1e-9)
        tl = ((test - np.log2(m)) ** 2).sum(1)
        upd = tl < best
        best = np.where(upd, tl, best)
        bG = np.where(upd, g, bG); bS = np.where(upd, s, bS); bW = np.where(upd, w, bW)
    nf = n_repeat * n_block
    return (np.median(bW.reshape(Ng, nf), 1),
            np.median(bG.reshape(Ng, nf), 1),
            np.median(bS.reshape(Ng, nf), 1))


def v1_batch(rho_obs, Sig, rng, n_resample=100, n_repeat=10, n_block=10, return_cls=False):
    """Faithful BIGFAM.v1: resample -> slope-test classify -> w BAND -> nested CV -> median.
    The fit never uses Sigma (off-diagonal unused); only the per-DOR SE (diagonal) enters, via the
    resample + slope test. Stage A skipped: real v1 estimates sd_d by a frreg data-bootstrap; the sim
    plants Sigma, so we read sd_d off its diagonal.

    return_cls=True also returns the slope-test classification ('High' fast,
    'Low' slow, 'None' similar). The classification fixes the search band, so
    it must be read from the same call as the estimate -- v1 is stochastic, a
    separate call would resample differently and the class would not match."""
    N = len(rho_obs)
    sd = np.sqrt(np.diagonal(Sig, axis1=1, axis2=2))                   # per-DOR SE = DIAGONAL only
    res = np.abs(rho_obs[:, None, :] + sd[:, None, :] * rng.standard_normal((N, n_resample, 3)))
    L = np.log2(np.maximum(res, 1e-9))                                 # (N, n_resample, 3)
    cls = _slope_test(L)
    W = np.empty(N); G = np.empty(N); S = np.empty(N)
    for c, band in BANDS.items():
        idx = np.where(cls == c)[0]
        if len(idx):
            W[idx], G[idx], S[idx] = _nested_cv(L[idx], band, rng, n_repeat, n_block)
    return (W, G, S, cls) if return_cls else (W, G, S)


# ── cross-method driver adapter (single-trait, seeded) ────────────────────────
def make(seed=V1_SEED):
    """Close over seed -> estimate(rho) callable, same shape as bigfam.make().
    v1 is resample + nested-CV, so it is stochastic: one seed makes the
    rep/trait order deterministic. rho needs only .rho_hat and .sigma_hat
    (per-DOR SE); v1 uses only the diagonal, so sigma_hat is wrapped as a
    diagonal Sigma."""
    rng = np.random.default_rng(seed)

    def estimate(rho):
        sd = np.asarray(rho.sigma_hat, float)                         # per-DOR SE (RhoEstimate.sigma_hat = sqrt(diag))
        Sig = np.zeros((1, 3, 3)); Sig[0][np.diag_indices(3)] = sd ** 2
        W, G, S, cls = v1_batch(np.asarray(rho.rho_hat, float)[None, :], Sig, rng, return_cls=True)
        return {"V_G": float(G[0]), "V_S": float(S[0]), "w_s_cal": float(W[0]),
                "decay_class": CLASS_NAME[str(cls[0])]}

    return estimate


def _demo():
    """Self-check: clean rho should recover the truth (band machine intact)."""
    VG, VS, wS = 0.5, 0.2, 0.7
    rho_hat = np.array([0.5 ** d * VG + wS ** (d - 1) * VS for d in (1, 2, 3)])
    sd = np.array([0.003, 0.006, 0.009])
    Sig = np.zeros((1, 3, 3)); Sig[0][np.diag_indices(3)] = sd ** 2
    W, G, S = v1_batch(rho_hat[None, :], Sig, np.random.default_rng(0))
    assert 0.3 < G[0] < 0.7, f"V_G off: {G[0]}"
    assert 0.0 < S[0] < 0.5, f"V_S off: {S[0]}"
    est = make(0)(type("R", (), {"rho_hat": rho_hat, "sigma_hat": sd})())
    assert 0.3 < est["V_G"] < 0.7, f"make V_G off: {est}"
    print(f"ok: v1_batch V_G={G[0]:.3f} V_S={S[0]:.3f} w_S={W[0]:.3f}  make={est}")


if __name__ == "__main__":
    _demo()
