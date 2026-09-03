"""File input: relative-pair tables and the trained calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import CalibrationCoef


def _read_table(stem: Path) -> pd.DataFrame:
    """Read <stem>.parquet if it exists, else <stem>.csv."""
    parquet = stem.parent / f"{stem.name}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)          # needs the 'parquet' extra
    return pd.read_csv(stem.parent / f"{stem.name}.csv")


def load_pairs(data_dir, name: str, pheno_type: str):
    """Read (pairs, cov, pheno) from a directory laid out as

        <data_dir>/kinpairs_dor1_3.csv                 id1, id2, dor
        <data_dir>/covariates.csv                      id + one column per covariate
        <data_dir>/phenotypes/<pheno_type>/<name>.csv  id + one value column

    .parquet is read in preference to .csv where present. A runnable example of
    this layout is examples/toy_data/ (see examples/make_toy_data.py).

    Convenience only -- estimate_rho takes the three DataFrames directly, so any
    loader that produces them works (see examples/quickstart.py).
    """
    root = Path(data_dir)

    pairs = _read_table(root / "kinpairs_dor1_3")
    pairs = pairs.rename(columns={c: "dor" for c in pairs.columns if c.lower() == "dor"})

    cov = _read_table(root / "covariates").set_index("id")

    pheno = _read_table(root / "phenotypes" / pheno_type / name).set_index("id")
    pheno = pheno.rename(columns={pheno.columns[0]: "phenotype"})[["phenotype"]]

    return pairs, cov, pheno


DEFAULT_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def load_artifacts(artifacts_dir=DEFAULT_ARTIFACTS) -> CalibrationCoef:
    """Read the trained Phase 2 calibration from <artifacts_dir>/ws_calibration.json.

    Defaults to the one shipped inside the package, so `load_artifacts()` works
    from an installed wheel with no repository checkout.
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
