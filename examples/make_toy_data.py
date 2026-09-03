"""Write a small toy dataset in the layout bigfam.io.load_pairs reads.

The output is committed under examples/toy_data/, so run this only to change
the size or the truth it is generated from:

    python examples/make_toy_data.py --out examples/toy_data

Both phenotypes come from one simulation at V_G = 0.5, V_S = 0.2, w_S = 0.2:
'height' is the continuous liability, 'disease' is that liability thresholded
at the top 20%. Age and sex have a real effect on both, which is what Phase 1
residualizes away; age is standardized (mean 0, sd 1), as the model expects,
not raw years.

Every individual sits in exactly one pair here, so 4,000 pairs per degree of
relatedness is not much information: expect the estimates to land near the
truth, not on it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V_G_TRUE, V_S_TRUE, W_S_TRUE = 0.5, 0.2, 0.2
N_PER_DOR = 4_000
PREVALENCE = 0.2
SEED = 0

# covariate -> effect on the phenotype (what estimate_rho has to remove)
COV_EFFECT = {"age": 0.30, "sex": 0.20}


def make_tables(n_per_dor: int = N_PER_DOR, seed: int = SEED):
    """-> (pairs, cov, height, disease) DataFrames, ready to write out."""
    rng = np.random.default_rng(seed)

    pair_blocks, ids, liability = [], [], []
    next_id = 0
    for d in (1, 2, 3):
        rho = 0.5 ** d * V_G_TRUE + W_S_TRUE ** (d - 1) * V_S_TRUE
        z1 = rng.standard_normal(n_per_dor)
        z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n_per_dor)

        first = np.arange(next_id, next_id + n_per_dor)
        second = first + n_per_dor
        next_id += 2 * n_per_dor

        pair_blocks.append(pd.DataFrame({"id1": first, "id2": second, "dor": d}))
        ids += [first, second]
        liability += [z1, z2]

    pairs = pd.concat(pair_blocks, ignore_index=True)

    order = np.argsort(np.concatenate(ids))          # tidy, id-ordered files
    ids = np.concatenate(ids)[order]
    liability = np.concatenate(liability)[order]

    cov = pd.DataFrame({"id": ids,
                        "age": rng.standard_normal(len(ids)),
                        "sex": rng.integers(0, 2, len(ids))})

    y = liability + sum(beta * cov[name].values for name, beta in COV_EFFECT.items())
    height = pd.DataFrame({"id": ids, "height": y})
    disease = pd.DataFrame({"id": ids,
                            "disease": (y > np.quantile(y, 1 - PREVALENCE)).astype(int)})
    return pairs, cov, height, disease


def write_toy_data(out_dir, n_per_dor: int = N_PER_DOR) -> None:
    """Write the four CSVs under out_dir in the load_pairs layout."""
    out = Path(out_dir)
    pairs, cov, height, disease = make_tables(n_per_dor)

    (out / "phenotypes" / "continuous").mkdir(parents=True, exist_ok=True)
    (out / "phenotypes" / "binary").mkdir(parents=True, exist_ok=True)

    pairs.to_csv(out / "kinpairs_dor1_3.csv", index=False)
    cov.to_csv(out / "covariates.csv", index=False, float_format="%.4f")
    height.to_csv(out / "phenotypes" / "continuous" / "height.csv",
                  index=False, float_format="%.4f")
    disease.to_csv(out / "phenotypes" / "binary" / "disease.csv", index=False)

    print(f"wrote {len(pairs)} pairs over {len(cov)} individuals to {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "toy_data"),
                    help="output directory (default: examples/toy_data)")
    ap.add_argument("--n-per-dor", type=int, default=N_PER_DOR,
                    help=f"pairs per degree of relatedness (default: {N_PER_DOR})")
    args = ap.parse_args()
    write_toy_data(args.out, args.n_per_dor)


if __name__ == "__main__":
    main()
