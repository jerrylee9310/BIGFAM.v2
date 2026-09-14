"""CLI: relative pairs -> V_A, V_C (Phase 1 -> 2 -> 3).

    python scripts/run_pipeline.py examples/toy_data height continuous \
        --cov-cols age sex --out outputs/

The data directory must hold the CSV (or parquet) layout bigfam.io.load_pairs
documents; examples/toy_data/ is a working example of it. To run on tables you
build yourself, call the three phases directly -- see examples/quickstart.py.
"""
from __future__ import annotations

import argparse

import bigfam
from bigfam.io import load_pairs, load_artifacts, save_rho, save_decomposition
from bigfam.io.load import DEFAULT_ARTIFACTS


def run(data_dir, name, pheno_type, artifacts=DEFAULT_ARTIFACTS,
        cov_cols=(), out=None):
    pairs, cov, pheno = load_pairs(data_dir, name, pheno_type)
    calib = load_artifacts(artifacts)

    rho = bigfam.estimate_rho(pairs, cov, pheno, pheno_type, cov_cols)   # Phase 1
    wc = bigfam.estimate_wc(rho, calib)                                  # Phase 2
    result = bigfam.decompose(rho, wc)                                   # Phase 3

    if out:
        save_rho(rho, out)
        save_decomposition(result, out)
    return rho, wc, result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", help="directory holding the input tables (CSV or parquet)")
    ap.add_argument("name", help="phenotype name, e.g. height")
    ap.add_argument("pheno_type", choices=["continuous", "binary"])
    ap.add_argument("--cov-cols", nargs="*", default=[],
                    help="covariate columns to adjust for (default: none)")
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS),
                    help="directory holding wc_calibration.json (default: the packaged one)")
    ap.add_argument("--out", default=None, help="write result TSVs here")
    args = ap.parse_args()

    rho, wc, result = run(args.data_dir, args.name, args.pheno_type,
                          args.artifacts, args.cov_cols, args.out)
    print(f"covariates: {', '.join(args.cov_cols) if args.cov_cols else 'none'}")
    print(f"rho_hat   = {rho.rho_hat}")
    print(f"w_c_cal   = {result.w_c_cal:.4f}  w-CI=[{result.wci_lo:.3f}, {result.wci_hi:.3f}]")
    print(f"V_A={result.V_A:.4f} (z {result.z_VA:.2f})  "
          f"V_C={result.V_C:.4f} (z {result.z_VC:.2f})")


if __name__ == "__main__":
    main()
