"""Offline training: simulated data -> the w_S ridge calibrator.

Not part of the inference path. Inference only loads the artifact this writes
(bigfam/artifacts/ws_calibration.json); run it with `scripts/train_phase2.py`.
"""
from __future__ import annotations

import numpy as np

from ..config import ALPHAS_CV, CLIP, SEED
from ..types import CalibrationCoef
from .features import FEAT_ALL


def train_calibration(train_df, seed: int = SEED) -> CalibrationCoef:
    """Fit StandardScaler + RidgeCV on the 24 features -> CalibrationCoef.

    train_df: a frame with the FEAT_ALL columns plus w_S_true (the target),
    as produced by dgm.generate_training_frame().
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    X = train_df[FEAT_ALL].values
    y = train_df["w_S_true"].values

    scaler = StandardScaler().fit(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    ridge = RidgeCV(alphas=ALPHAS_CV, cv=cv).fit(scaler.transform(X), y)

    return CalibrationCoef(
        feature_order=list(FEAT_ALL),
        scaler_mean=np.asarray(scaler.mean_, dtype=float),
        scaler_scale=np.asarray(scaler.scale_, dtype=float),
        ridge_coef=np.asarray(ridge.coef_, dtype=float),
        ridge_intercept=float(ridge.intercept_),
        ridge_alpha=float(ridge.alpha_),
        clip=tuple(CLIP),
    )


def train_all(out_dir) -> CalibrationCoef:
    """Simulate -> features -> ridge -> write ws_calibration.json. Returns the fit."""
    from ..io.save import save_artifacts
    from .dgm import generate_training_frame

    calib = train_calibration(generate_training_frame())
    save_artifacts(calib, out_dir)
    return calib
