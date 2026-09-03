"""Calibration artifact: save/load round-trip (no full training)."""
import numpy as np

from bigfam.types import CalibrationCoef
from bigfam.phase2.features import FEAT_ALL
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
