"""Calibration serialization round-trip + downstream-loss metric (no full training)."""
import numpy as np
import pandas as pd

from bigfam.types import CalibrationCoef
from bigfam.phase2.features import FEAT_ALL
from bigfam.phase2.train import (
    calibrate_predict,
    downstream_hybrid_loss,
    train_calibration_downstream,
)
from bigfam.io.save import save_artifacts
from bigfam.io.load import load_artifacts


def _make_calib():
    return CalibrationCoef(
        feature_order=list(FEAT_ALL),
        scaler_mean=np.linspace(0, 1, 24),
        scaler_scale=np.linspace(1, 2, 24),
        ridge_coef=np.linspace(-1, 1, 24),
        ridge_intercept=0.3, ridge_alpha=1.0,
        clip=(0.01, 0.99),
    )


def test_round_trip(tmp_path):
    calib = _make_calib()
    save_artifacts(calib, tmp_path)                       # only ws_calibration.json
    calib2 = load_artifacts(tmp_path)

    assert calib2.feature_order == list(FEAT_ALL)
    np.testing.assert_allclose(calib2.scaler_mean, calib.scaler_mean)
    np.testing.assert_allclose(calib2.ridge_coef, calib.ridge_coef)
    np.testing.assert_allclose(calib2.ridge_intercept, calib.ridge_intercept)


def _make_downstream_frame(n=48):
    rs = np.random.RandomState(1)
    w = np.resize(np.array([0.20, 0.30, 0.70, 0.80]), n)
    V_G = rs.uniform(0.10, 0.60, n)
    V_S = rs.uniform(0.05, 0.35, n)

    d = np.arange(1, 4)
    rho = 0.5 ** d[None, :] * V_G[:, None] + w[:, None] ** (d - 1)[None, :] * V_S[:, None]
    Sigma = np.diag([1e-4, 1.5e-4, 2e-4])

    data = {name: rs.normal(size=n) for name in FEAT_ALL}
    data["w_map"] = np.clip(w + rs.normal(scale=0.03, size=n), 0.01, 0.99)
    data["profile_width"] = rs.uniform(0.02, 0.40, n)
    data["w_S_true"] = w
    data["V_G"] = V_G
    data["V_S"] = V_S
    for j in range(3):
        data[f"rho_obs_{j + 1}"] = rho[:, j]
    for r in range(3):
        for c in range(3):
            data[f"Sigma_{r + 1}{c + 1}"] = Sigma[r, c]
    return pd.DataFrame(data)


def test_downstream_hybrid_loss_prefers_true_w():
    df = _make_downstream_frame()
    w_true = df["w_S_true"].values
    w_bad = np.clip(1.0 - w_true, 0.01, 0.99)

    assert downstream_hybrid_loss(w_true, df) < downstream_hybrid_loss(w_bad, df)


def test_train_calibration_downstream_smoke():
    df = _make_downstream_frame()
    calib = train_calibration_downstream(df, lambda_w=0.05)
    pred = calibrate_predict(df, calib)

    assert isinstance(calib, CalibrationCoef)
    assert calib.feature_order == list(FEAT_ALL)
    assert np.all(np.isfinite(pred))
    assert np.all((0.01 <= pred) & (pred <= 0.99))
