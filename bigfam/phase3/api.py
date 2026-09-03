"""Phase 3 unified entry point.

    (rho_hat, Sigma_hat, w_hat) -> (V_G, V_S, conditional SE, w-CI)

Fixed-w conditional decomposition (refit) + profile w-CI (robust.py).
Reference: docs/method/phase3.md, scratch step3/step4.
"""
from __future__ import annotations

import numpy as np

from ..config import CLIP
from ..types import RhoEstimate, WsEstimate, Decomposition
from .refit import refit_fixed_ws
from .robust import profile_ci


def decompose(rho: RhoEstimate, ws: WsEstimate) -> Decomposition:
    """Decompose into (V_G, V_S) at fixed w_S = w_s_cal, with conditional SE and
    the profile-likelihood 95% CI for w_S."""
    lo, hi = CLIP
    w_cal = float(np.clip(ws.w_s_cal, lo, hi))

    beta, se, _Ainv, _dbeta = refit_fixed_ws(rho.rho_hat, rho.Sigma_hat, w_cal)
    V_G, V_S = float(beta[0]), float(beta[1])
    se_VG, se_VS = float(se[0]), float(se[1])
    z_VG = V_G / se_VG if se_VG > 0 else float("nan")
    z_VS = V_S / se_VS if se_VS > 0 else float("nan")

    wci_lo, wci_hi = profile_ci(rho.rho_hat, rho.Sigma_hat)

    return Decomposition(
        V_G=V_G, V_S=V_S, se_VG_cond=se_VG, se_VS_cond=se_VS,
        z_VG=z_VG, z_VS=z_VS, w_s_cal=w_cal,
        wci_lo=wci_lo, wci_hi=wci_hi,
    )
