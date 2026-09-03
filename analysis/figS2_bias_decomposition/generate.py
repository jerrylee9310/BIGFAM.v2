"""Figure S2 -- DATA: decomposing the Fig. 2 bias into decay-rate estimation vs. NNLS.

Fig. 2 shows BIGFAM.v2 underestimating h^2 under slow/fast common-environmental
decay. This isolates where that bias comes from, on the same simulated data
(rho_hat, Sigma_hat), by comparing three variants of the Phase 2->3 pipeline:

    pipeline    w_hat_C (ridge) -> NNLS refit     the reported BIGFAM.v2 estimator
    truew_nnls  true w_C -> NNLS refit             removes the decay-rate estimation error
    truew_gls   true w_C -> unconstrained GLS      also removes the NNLS non-negativity step

At w_C=0.5 the design is exactly singular, so truew_gls is undefined (dropped
via divide-by-zero -> NaN) and truew_nnls does not recover the truth either.

Self-contained: only depends on the installed bigfam package (no other
analysis/ folder). plot.py reads the parquet this writes.

Run:  .venv/bin/python analysis/figS2_bias_decomposition/generate.py
Out:  figS2.parquet  (R=1,000 replicates x w_C in {0.2, 0.5, 0.8} x 3 variants)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import bigfam
from bigfam.io.load import load_artifacts
from bigfam.phase2.features import compute_feature_dict, FEAT_ALL
from bigfam.phase3.refit import refit_batch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figS2.parquet"

SEED_BASE = 20_260_821_800_000
V_A, V_C, R = 0.5, 0.2, 1_000
WS = [0.2, 0.5, 0.8]                  # fast, degenerate, slow -- matches plot.py's column order
W_G = 0.5
X_G = W_G ** np.arange(1, 4)


def generate(v_a, v_c, w_c, seed, n_d=10_000):
    """Pair-level DGM at DOR in {1,2,3}, identical setup to Fig. 2."""
    rng = np.random.default_rng(seed)
    id1, id2, dor, pids, pvals = [], [], [], [], []
    nxt = 0
    for d in (1, 2, 3):
        rho = W_G ** d * v_a + w_c ** (d - 1) * v_c
        za, zb = rng.standard_normal(n_d), rng.standard_normal(n_d)
        y1, y2 = za, rho * za + np.sqrt(1.0 - rho ** 2) * zb
        i1 = np.arange(nxt, nxt + n_d); nxt += n_d
        i2 = np.arange(nxt, nxt + n_d); nxt += n_d
        id1.append(i1); id2.append(i2); dor.append(np.full(n_d, d))
        pids += [i1, i2]; pvals += [y1, y2]
    pairs = pd.DataFrame({"id1": np.concatenate(id1), "id2": np.concatenate(id2),
                          "dor": np.concatenate(dor)})
    pheno = pd.DataFrame({"phenotype": np.concatenate(pvals)},
                         index=np.concatenate(pids))
    pheno.index.name = "id"
    return pairs, pheno


def evidence(pairs, pheno):
    """pairs, pheno -> RhoEstimate(rho_hat, Sigma_hat). No covariates in this sim."""
    cov = pheno[[]]                          # 0-col, same index (intercept-only)
    return bigfam.estimate_rho(pairs, cov, pheno, "continuous", cov_cols=[])


def calibrate_predict_batch(df, calib):
    """Batch form of bigfam.phase2.calibrate.calibrate_ws (that one is single-vector)."""
    X = df[FEAT_ALL].to_numpy(dtype=float)
    X_std = (X - calib.scaler_mean) / calib.scaler_scale
    lo, hi = calib.clip
    return np.clip(calib.ridge_intercept + X_std @ calib.ridge_coef, lo, hi)


def gls_unconstrained(rho, Sinv, w_vec):
    """Unconstrained GLS on design [0.5^d, w^(d-1)] -- V_A_hat without the NNLS step."""
    c2 = w_vec[:, None] ** np.arange(3)[None, :]
    xg = np.broadcast_to(X_G, rho.shape)
    quad = lambda a, b: np.einsum("ni,nij,nj->n", a, Sinv, b)   # noqa: E731
    a11, a12, a22 = quad(xg, xg), quad(xg, c2), quad(c2, c2)
    b1, b2 = quad(xg, rho), quad(c2, rho)
    det = a11 * a22 - a12 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a22 * b1 - a12 * b2) / det


def main():
    calib = load_artifacts()          # packaged ws_calibration.json
    rows = []
    for wi, w in enumerate(WS):
        rhos, Sigs = [], []
        for rep in range(R):
            pairs, pheno = generate(V_A, V_C, w, SEED_BASE + wi * 100_000 + rep)
            r = evidence(pairs, pheno)
            rhos.append(r.rho_hat); Sigs.append(r.Sigma_hat)
        rho, Sig = np.stack(rhos), np.stack(Sigs)
        Sinv = np.linalg.inv(Sig)

        feat = compute_feature_dict(rho, Sig)
        w_hat = calibrate_predict_batch(pd.DataFrame({k: feat[k] for k in FEAT_ALL}), calib)
        w_true = np.full(R, float(w))

        variants = {
            "pipeline": refit_batch(w_hat, rho, Sig)[0][:, 0],
            "truew_nnls": refit_batch(w_true, rho, Sig)[0][:, 0],
            "truew_gls": gls_unconstrained(rho, Sinv, w_true),
        }
        for name, v in variants.items():
            rows.append(pd.DataFrame({"w_true": w, "variant": name, "V_A_hat": v}))
            print(f"  w={w} {name}: mean={v.mean():.4f} bias={v.mean() - V_A:+.4f}")
        print(f"w={w} done", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
