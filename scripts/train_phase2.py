"""CLI: run the offline Phase 2 simulation and rewrite the shipped calibration.

    python scripts/train_phase2.py [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bigfam.config import N_SIM, SEED
from bigfam.phase2.train import train_all

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "bigfam" / "artifacts"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory (default: bigfam/artifacts/)")
    args = ap.parse_args()

    print(f"training Phase 2 calibration ({N_SIM:,} simulated draws, seed {SEED}) ...")
    calib = train_all(args.out)
    print(f"  ridge_alpha={calib.ridge_alpha}  intercept={calib.ridge_intercept:.4f}")
    print(f"wrote ws_calibration.json to {args.out}")


if __name__ == "__main__":
    main()
