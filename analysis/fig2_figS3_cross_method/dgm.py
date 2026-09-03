"""Individual-level relative-pair generator (continuous + binary).

For each DOR d, draws N_d independent pairs from a bivariate normal:
    (y1, y2) ~ N(0, [[1, rho_d], [rho_d, 1]]),  rho_d = w_G^d V_G + w_S^{d-1} V_S
binary thresholds the same liability at tau = Phi^{-1}(1-K).

Pairs are independent of each other (no within-family structure). Each
individual appears in exactly one pair, so ids are assigned globally unique.
There are no covariates in this simulation (config.COV_COLS = []) -- BIGFAM
Phase 1 sees only the intercept column, so no covariate table is built here;
an empty cov frame is filled in on the fly by evidence.py.

Returns: (pairs, pheno)
    pairs : DataFrame[id1, id2, dor]
    pheno : DataFrame indexed by id, 'phenotype'
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import DORS, K_PREV, N_D, true_rho


def generate(scale: str, w_s: float, seed: int, n_d: int = N_D, K: float = K_PREV):
    rng = np.random.default_rng(seed)
    tau = norm.ppf(1.0 - K)

    id1_all, id2_all, dor_all = [], [], []
    pheno_ids, pheno_vals = [], []
    next_id = 0

    for d in DORS:
        rho = true_rho(w_s, d)
        za = rng.standard_normal(n_d)
        zb = rng.standard_normal(n_d)
        L1 = za
        L2 = rho * za + np.sqrt(1.0 - rho ** 2) * zb

        if scale == "continuous":
            y1, y2 = L1, L2
        elif scale == "binary":
            y1 = (L1 > tau).astype(float)
            y2 = (L2 > tau).astype(float)
        else:
            raise ValueError(f"scale must be continuous|binary, got {scale!r}")

        ids1 = np.arange(next_id, next_id + n_d); next_id += n_d
        ids2 = np.arange(next_id, next_id + n_d); next_id += n_d

        id1_all.append(ids1); id2_all.append(ids2)
        dor_all.append(np.full(n_d, d))
        pheno_ids.append(ids1); pheno_ids.append(ids2)
        pheno_vals.append(y1); pheno_vals.append(y2)

    pairs = pd.DataFrame({
        "id1": np.concatenate(id1_all),
        "id2": np.concatenate(id2_all),
        "dor": np.concatenate(dor_all),
    })

    pheno = pd.DataFrame(
        {"phenotype": np.concatenate(pheno_vals)},
        index=np.concatenate(pheno_ids),
    )
    pheno.index.name = "id"

    return pairs, pheno
