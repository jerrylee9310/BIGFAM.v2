"""CLI: run the offline Phase 2 simulation and write artifacts/.

    python scripts/train_phase2.py [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bigfam.phase2.train import train_all

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--objective", choices=["ws", "downstream_hybrid"], default="ws")
    ap.add_argument("--lambda-w", type=float, default=0.05)
    args = ap.parse_args()

    print(f"training Phase 2 calibration ({args.objective}, wide DGP 40k, seed 42) ...")
    calib = train_all(
        args.out,
        objective=args.objective,
        lambda_w=args.lambda_w,
    )
    print(f"  ridge_alpha={calib.ridge_alpha}  intercept={calib.ridge_intercept:.4f}  (ridge-only)")
    print(f"wrote ws_calibration.json to {args.out}")


if __name__ == "__main__":
    main()
