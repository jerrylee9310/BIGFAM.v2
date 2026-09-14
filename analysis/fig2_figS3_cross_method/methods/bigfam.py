"""BIGFAM -- decay class, estimates its own w_C.

Phase 2 predicts w_C from (rho_hat, Sigma_hat) (a trained predictor); Phase 3
fits (V_A, V_C) by GLS/NNLS at that fixed w_C. No condition -- the s_d =
w_C^{d-1} shape is set by the data, not chosen in advance.
"""
from __future__ import annotations

import bigfam as _bf


def make(calib):
    """Close over the artifacts to make an estimate(rho) callable."""
    def estimate(rho):
        wc = _bf.estimate_wc(rho, calib)
        dec = _bf.decompose(rho, wc)
        return {
            "V_A": dec.V_A, "V_C": dec.V_C,
            "w_c_cal": wc.w_c_cal,
        }
    return estimate
