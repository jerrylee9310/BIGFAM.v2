"""CLI: raw relative pairs -> V_G, V_S (Phase 1 -> 2 -> 3).

    python scripts/run_pipeline.py height continuous --artifacts artifacts/ --out outputs/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bigfam
from bigfam.config import COV_COLS
from bigfam.io import load_pairs, load_artifacts, save_result

DEFAULT_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def run(name, pheno_type, artifacts=DEFAULT_ARTIFACTS, dataset="GS23471", out=None):
    pairs, cov, pheno = load_pairs(name, pheno_type, dataset=dataset)
    calib = load_artifacts(artifacts)

    rho = bigfam.estimate_rho(pairs, cov, pheno, pheno_type, COV_COLS)   # Phase 1
    ws = bigfam.estimate_ws(rho, calib)                                  # Phase 2
    result = bigfam.decompose(rho, ws)                                   # Phase 3

    if out:
        save_result(rho, ws, result, out)
    return rho, ws, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="phenotype name (e.g. height, diabetes_G)")
    ap.add_argument("pheno_type", choices=["continuous", "binary"])
    ap.add_argument("--dataset", default="GS23471")
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rho, ws, result = run(args.name, args.pheno_type, args.artifacts, args.dataset, args.out)
    print(f"rho_hat   = {rho.rho_hat}")
    print(f"w_s_cal   = {result.w_s_cal:.4f}  w-CI=[{result.wci_lo:.3f}, {result.wci_hi:.3f}]")
    print(f"V_G={result.V_G:.4f} (z {result.z_VG:.2f})  "
          f"V_S={result.V_S:.4f} (z {result.z_VS:.2f})")


if __name__ == "__main__":
    main()
