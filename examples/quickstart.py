"""End-to-end demo on synthetic data — no real phenotype data required.

Generates relative pairs at dor 1/2/3 from a known (V_A, V_C, w_C), runs the
full Phase 1 -> 2 -> 3 pipeline, and compares the estimates to the truth. The
three DataFrames built in make_synthetic_pairs() are exactly the input shape
bigfam.estimate_rho expects, so copy that part for your own data.

    python examples/quickstart.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bigfam
from bigfam.io import load_artifacts

V_A_TRUE, V_C_TRUE, W_S_TRUE = 0.5, 0.2, 0.2   # ground truth this demo recovers
N_PER_DOR = 10_000
SEED = 0


def make_synthetic_pairs(v_g=V_A_TRUE, v_s=V_C_TRUE, w_c=W_S_TRUE,
                          n_per_dor=N_PER_DOR, seed=SEED):
    """Synthetic continuous-trait relative pairs (no covariates).

    Each pair is a draw from a bivariate normal whose correlation is the model's
    rho_d = 0.5^d * V_A + w_C^(d-1) * V_C. Every individual appears in exactly
    one pair here; real data reuses individuals, which is what the cluster
    sandwich in Phase 1 corrects for.

    Returns (pairs, cov, pheno) ready for bigfam.estimate_rho.
    """
    rng = np.random.default_rng(seed)
    id1_all, id2_all, dor_all, y1_all, y2_all = [], [], [], [], []
    next_id = 0
    for d in (1, 2, 3):
        rho = 0.5 ** d * v_g + w_c ** (d - 1) * v_s
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
    calib = load_artifacts()

    rho = bigfam.estimate_rho(pairs, cov, pheno, "continuous", cov_cols=[])   # Phase 1
    wc = bigfam.estimate_wc(rho, calib)                                       # Phase 2
    result = bigfam.decompose(rho, wc)                                        # Phase 3

    print(f"rho_hat = {rho.rho_hat}")
    print(f"w_C:  true={W_S_TRUE:.3f}  est={result.w_c_cal:.3f}  "
          f"CI=[{result.wci_lo:.3f}, {result.wci_hi:.3f}]")
    print(f"V_A:  true={V_A_TRUE:.3f}  est={result.V_A:.3f}  se={result.se_VA_cond:.3f}")
    print(f"V_C:  true={V_C_TRUE:.3f}  est={result.V_C:.3f}  se={result.se_VC_cond:.3f}")
    print("(se is conditional on w_C)")


if __name__ == "__main__":
    main()
