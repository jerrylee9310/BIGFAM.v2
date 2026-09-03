"""End-to-end demo on synthetic data — no real phenotype data required.

Generates relative pairs at DOR 1/2/3 from a known (V_G, V_S, w_S), then runs
the full Phase 1 -> 2 -> 3 pipeline and compares the estimate to the truth.

    python examples/quickstart.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bigfam
from bigfam.io import load_artifacts

V_G_TRUE, V_S_TRUE, W_S_TRUE = 0.5, 0.2, 0.2   # ground truth this demo recovers
N_PER_DOR = 10_000
SEED = 0


def make_synthetic_pairs(v_g=V_G_TRUE, v_s=V_S_TRUE, w_s=W_S_TRUE,
                          n_per_dor=N_PER_DOR, seed=SEED):
    """Synthetic continuous-trait relative pairs (no covariates).

    rho_d = 0.5^d * V_G + w_S^(d-1) * V_S  (docs/README.md), sampled as a
    bivariate normal per DOR level. Returns (pairs, cov, pheno) ready for
    bigfam.estimate_rho.
    """
    rng = np.random.default_rng(seed)
    id1_all, id2_all, dor_all, y1_all, y2_all = [], [], [], [], []
    next_id = 0
    for d in (1, 2, 3):
        rho = 0.5 ** d * v_g + w_s ** (d - 1) * v_s
        z1 = rng.standard_normal(n_per_dor)
        z2 = rng.standard_normal(n_per_dor)
        y1 = z1
        y2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * z2

        ids1 = np.arange(next_id, next_id + n_per_dor); next_id += n_per_dor
        ids2 = np.arange(next_id, next_id + n_per_dor); next_id += n_per_dor
        id1_all.append(ids1); id2_all.append(ids2); dor_all.append(np.full(n_per_dor, d))
        y1_all.append(y1); y2_all.append(y2)

    pairs = pd.DataFrame({
        "id1": np.concatenate(id1_all),
        "id2": np.concatenate(id2_all),
        "dor": np.concatenate(dor_all),
    })
    ids = np.concatenate([*id1_all, *id2_all])
    pheno = pd.DataFrame({"phenotype": np.concatenate([*y1_all, *y2_all])}, index=ids)
    cov = pd.DataFrame(index=pheno.index)   # no covariates in this demo
    return pairs, cov, pheno


def main():
    pairs, cov, pheno = make_synthetic_pairs()
    calib = load_artifacts("artifacts/")

    rho = bigfam.estimate_rho(pairs, cov, pheno, "continuous", cov_cols=[])   # Phase 1
    ws = bigfam.estimate_ws(rho, calib)                                       # Phase 2
    result = bigfam.decompose(rho, ws)                                        # Phase 3

    print(f"rho_hat = {rho.rho_hat}")
    print(f"w_S:  true={W_S_TRUE:.3f}  est={result.w_s_cal:.3f}  "
          f"CI=[{result.wci_lo:.3f}, {result.wci_hi:.3f}]")
    print(f"V_G:  true={V_G_TRUE:.3f}  est={result.V_G:.3f}")
    print(f"V_S:  true={V_S_TRUE:.3f}  est={result.V_S:.3f}")


if __name__ == "__main__":
    main()
