"""Phase 3: (rho_hat, Sigma_hat, w_C) -> (V_A, V_C, conditional SE, w-CI).

Conditional decomposition at the fixed w_C from Phase 2 (refit.py) plus the
profile-likelihood CI for w_C itself (robust.py).
"""
from __future__ import annotations

import numpy as np

from ..config import CLIP
from ..types import RhoEstimate, WcEstimate, Decomposition
from .refit import refit_fixed_wc
from .robust import profile_ci


def decompose(rho: RhoEstimate, wc: WcEstimate) -> Decomposition:
    """Split the similarity curve into genetic and shared-environment variance.

    rho: Phase 1 output.  wc: Phase 2 output.
    Returns (V_A, V_C) at fixed w_C = wc.w_c_cal with their conditional SE, plus
    the 95% profile CI for w_C (a CI covering 0.5 means the split is not identified).
    """
    w_cal = float(np.clip(wc.w_c_cal, *CLIP))

    beta, se = refit_fixed_wc(rho.rho_hat, rho.Sigma_hat, w_cal)
    V_A, V_C = float(beta[0]), float(beta[1])
    se_VA, se_VC = float(se[0]), float(se[1])
    z_VA = V_A / se_VA if se_VA > 0 else float("nan")
    z_VC = V_C / se_VC if se_VC > 0 else float("nan")

    wci_lo, wci_hi = profile_ci(rho.rho_hat, rho.Sigma_hat)

    return Decomposition(
        V_A=V_A, V_C=V_C, se_VA_cond=se_VA, se_VC_cond=se_VC,
        z_VA=z_VA, z_VC=z_VC, w_c_cal=w_cal,
        wci_lo=wci_lo, wci_hi=wci_hi,
    )
