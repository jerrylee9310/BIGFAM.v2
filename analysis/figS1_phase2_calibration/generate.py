"""Figure S1 — DATA: Phase 2 w_hat_C calibration replicates.

Forward-simulates the shipped Phase 2 estimator over the paper's Supplement
S2.4 prior and records, per replicate, the true w_C and the ridge point
estimate w_hat_C. plot.py reads the CSV this writes.

Prior (identical to S2.4):
    w_C        ~ U(0.01, 0.99)
    (V_G, V_S) ~ Dirichlet(1, 1, 1)           # V_E is the remainder
    sigma_d    ~ U(0.001, 0.10), unordered
    Sigma corr : 3 free positive off-diagonals ~U(0, 0.9), rejected to PSD

For each draw: rho_d = 0.5^d V_G + w_C^{d-1} V_S, one noisy rho_hat ~ N(rho, Sigma),
then features -> shipped ridge -> w_hat_C. Seed 123 = the calibration eval seed; at
N=40k this reproduces the validated eval draw byte-identically. N is enlarged here
so the Figure S1B retention heatmap fills its sparse high-signal / low-noise corner.

Run:  .venv/bin/python analysis/figS1_phase2_calibration/generate.py
Out:  analysis/figS1_phase2_calibration/figS1.csv  (200,000 rows)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from bigfam.phase2.dgm import sample_rho_hats
from bigfam.phase2.features import compute_feature_dict, FEAT_ALL
from bigfam.io.load import load_artifacts


def calibrate_predict(df, calib):
    """Vectorised calibrated w_C over a DataFrame (matches bigfam.phase2.calibrate.calibrate_ws
    row-wise; that function is single-row only, so this batches the same formula)."""
    lo, hi = calib.clip
    x_std = (df[FEAT_ALL].values - calib.scaler_mean) / calib.scaler_scale
    return np.clip(calib.ridge_intercept + x_std @ calib.ridge_coef, lo, hi)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figS1.csv")

SEED, N = 123, 200_000        # visualization eval draw (independent of the 40k ridge fit)
DOMAIN, SIGMA_HI = (0.01, 0.99), 0.10


def _psd_ok(r):
    a, b, c = r[:, 0], r[:, 1], r[:, 2]
    return (1.0 + 2.0 * a * b * c - a * a - b * b - c * c) > 1e-6


def _free_corr(rng, N):
    """3 free positive off-diagonals [r12, r13, r23], rejection to PSD."""
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
    """S2.4 prior draw. Returns w_C, V_G, V_S, Sigmas (same RNG order as calibration)."""
    w = rng.uniform(*DOMAIN, N)

    u = rng.uniform(0.0, 1.0, N)                    # Dirichlet(1,1,1) on (V_G, V_S)
    v = rng.uniform(0.0, 1.0, N)
    m = (u + v) > 1.0
    u[m], v[m] = 1.0 - u[m], 1.0 - v[m]
    V_G, V_S = u, v

    R = _free_corr(rng, N)
    sd = rng.uniform(0.001, SIGMA_HI, (N, 3))       # unordered
    Sigmas = R * sd[:, :, None] * sd[:, None, :]
    return w, V_G, V_S, Sigmas


def main():
    rng = np.random.default_rng(SEED)
    w, V_G, V_S, Sigmas = sample_scenarios(rng, N)

    d = np.arange(1, 4)
    rho_true = 0.5 ** d[None, :] * V_G[:, None] + w[:, None] ** (d - 1)[None, :] * V_S[:, None]
    rho_hat = sample_rho_hats(rho_true, Sigmas, rng)

    feats = compute_feature_dict(rho_hat, Sigmas)
    df = pd.DataFrame({name: feats[name] for name in FEAT_ALL})

    calib = load_artifacts()                  # shipped Phase 2 ridge (packaged artifact)
    w_hat = calibrate_predict(df, calib)

    sd = np.sqrt(np.stack([Sigmas[:, k, k] for k in range(3)], axis=1))
    out = pd.DataFrame({
        "w_S_true": w,
        "V_G": V_G,
        "V_S": V_S,
        "sigma_max": sd.max(axis=1),               # raw correlation measurement sd (noise)
        "w_hat": w_hat,                            # w_hat_C, shipped ridge point estimate
    })
    out.to_csv(OUT, index=False)
    print(f"saved: {OUT}  ({len(out)} rows)")
    print(out.describe().loc[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
