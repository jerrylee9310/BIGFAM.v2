"""BIGFAM -- decay class, estimates its own w_S.

Phase 2 predicts w_S from (rho_hat, Sigma_hat) (a trained predictor); Phase 3
fits (V_G, V_S) by GLS/NNLS at that fixed w_S. No condition -- the s_d =
w_S^{d-1} shape is set by the data, not chosen in advance.
"""
from __future__ import annotations

import bigfam as _bf


def make(calib):
    """Close over the artifacts to make an estimate(rho) callable."""
    def estimate(rho):
        ws = _bf.estimate_ws(rho, calib)
        dec = _bf.decompose(rho, ws)
        return {
            "V_G": dec.V_G, "V_S": dec.V_S,
            "w_s_cal": ws.w_s_cal,
        }
    return estimate
