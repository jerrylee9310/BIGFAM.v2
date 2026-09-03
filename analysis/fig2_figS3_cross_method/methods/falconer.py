"""Falconer -- relative correlation / relatedness coefficient, inverse-variance WLS.

zero-class model (s_d == 0): no shared-environment term, so V_S>0 gets absorbed
into h^2 if present.
    h2_d = rho_d / w_G^d,   V_G = sum_d w_d h2_d / sum_d w_d,   w_d = w_G^{2d} / se_d^2
For binary, the same formula runs on the liability-scale (tetrachoric) rho ->
Tet-Falconer.
"""
from __future__ import annotations

import numpy as np

from relatedness import W_G

_WGD = W_G ** np.arange(1, 4)              # [0.5, 0.25, 0.125]


def estimate(rho):
    r = rho.rho_hat
    se = rho.sigma_hat
    h2_d = r / _WGD
    w = _WGD ** 2 / se ** 2
    return {"V_G": float((w * h2_d).sum() / w.sum())}
