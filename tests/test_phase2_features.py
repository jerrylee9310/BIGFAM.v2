"""extract_features: length 24, FEAT_ALL order, finite on a known rho."""
import numpy as np

from bigfam.types import RhoEstimate
from bigfam.phase2.features import (
    FEAT_ALL, FEAT_BASE, FEAT_NEW, extract_features, compute_feature_dict,
)


def _rho():
    rho_hat = np.array([0.30, 0.12, 0.05])
    Sigma = np.diag([1e-4, 2e-4, 3e-4])
    return RhoEstimate(rho_hat, Sigma, 3)


def test_feat_contract_lengths():
    assert len(FEAT_BASE) == 15
    assert len(FEAT_NEW) == 9
    assert len(FEAT_ALL) == 24
    assert FEAT_ALL == FEAT_BASE + FEAT_NEW
    assert len(set(FEAT_ALL)) == 24          # no duplicate names


def test_extract_shape_and_order():
    x = extract_features(_rho())
    assert x.shape == (24,)
    assert np.all(np.isfinite(x))

    # value at each position must equal the named entry in the batch dict
    d = compute_feature_dict(_rho().rho_hat[None], _rho().Sigma_hat[None])
    for i, name in enumerate(FEAT_ALL):
        assert x[i] == d[name][0], f"order mismatch at {i}:{name}"


def test_any_nonpos_flag():
    # negative rho -> any_nonpos = 1
    rho = RhoEstimate(np.array([-0.01, 0.1, 0.05]), np.diag([1e-4, 1e-4, 1e-4]), 3)
    x = extract_features(rho)
    assert x[FEAT_ALL.index("any_nonpos")] == 1.0
