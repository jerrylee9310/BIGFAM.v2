"""File input: relative pairs / covariates / phenotypes and learned artifacts.

Reads the frozen db/01_processed layout (parquet-only; phenotypes split into
continuous/ and binary/ subdirs -- see scratch slug bigfam-input-freeze):
  - kinpairs column DOR (uppercase) -> dor
  - phenotype value column is named after the trait -> renamed to 'phenotype'
  - continuous phenotypes are z-scored here (model rho_d assumes standardized y;
    values are already irnt-frozen, so this z-score is ~identity)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import COV_COLS
from ..types import CalibrationCoef

# default location of the processed GS23471 tables (relative to project root)
DEFAULT_DB_ROOT = Path("db/01_processed")


def load_pairs(name: str, pheno_type: str, dataset: str = "GS23471",
               base_dir=None, cov_cols=COV_COLS):
    """Load (pairs, cov, pheno) for a phenotype.

    pairs: id1, id2, dor.  cov: indexed by id with cov_cols.
    pheno: indexed by id with a single 'phenotype' column (z-scored if continuous).
    """
    root = Path(base_dir) if base_dir is not None else (DEFAULT_DB_ROOT / dataset)

    pairs = pd.read_parquet(root / "kinpairs_dor1_3.parquet")
    pairs = pairs.rename(columns={c: "dor" for c in pairs.columns if c.lower() == "dor"})

    cov = pd.read_parquet(root / "covariates.parquet").set_index("id")

    pheno = pd.read_parquet(root / "phenotypes" / pheno_type / f"{name}.parquet").set_index("id")
    value_col = [c for c in pheno.columns][0]            # trait-named value column
    pheno = pheno.rename(columns={value_col: "phenotype"})[["phenotype"]]

    if pheno_type == "continuous":
        y = pheno["phenotype"].astype(float)
        std = y.std()
        pheno["phenotype"] = (y - y.mean()) / (std if std > 0 else 1.0)

    return pairs, cov, pheno


def load_artifacts(artifacts_dir):
    """Load the Phase 2 calibration (artifacts/ws_calibration.json).

    Returns a CalibrationCoef. This used to return a 3-tuple whose second and
    third slots held a learned reliability gate (ws_gate.pkl / ws_gate_cuts.json);
    the gate was removed and the slots always came back empty.
    """
    with open(Path(artifacts_dir) / "ws_calibration.json") as f:
        c = json.load(f)
    return CalibrationCoef(
        feature_order=c["feature_order"],
        scaler_mean=np.asarray(c["scaler_mean"], dtype=float),
        scaler_scale=np.asarray(c["scaler_scale"], dtype=float),
        ridge_coef=np.asarray(c["ridge_coef"], dtype=float),
        ridge_intercept=float(c["ridge_intercept"]),
        ridge_alpha=float(c["ridge_alpha"]),
        clip=tuple(c["clip"]),
    )
