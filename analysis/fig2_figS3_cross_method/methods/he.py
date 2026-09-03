"""Haseman-Elston -- original form: regress individual-pair (y1-y2)^2 on relatedness.

Continuous only (HE-SD, Haseman & Elston 1972); imposes the AE model outright:
    E[(y1-y2)^2] = 2*sigma_P^2 - 2*w_G^d*sigma_A^2  -> fix the intercept at the
    AE value 2*sigma_P^2, origin-OLS on the rest.
    V_G = -slope / (2 * sigma_P^2)
Leaving the intercept free would let a DOR-constant shared-environment term
(-2*sigma_P^2*w_S(d)*V_S with w_S(d) === const) get absorbed into it, giving the
same V_G as ACE-const (not AE). Fixing the intercept instead pushes all of the
shared-environment signal into the slope, so V_G absorbs it -- the same
AE-class bias as Falconer/PCGC (origin regression).
Binary can't observe liability (y1-y2)^2, so PCGC replaces HE there
(-> methods/pcgc.py).
"""
from __future__ import annotations

import numpy as np

from relatedness import W_G

_WGD = W_G ** np.arange(1, 4)              # [0.5, 0.25, 0.125]
_REL = {1: _WGD[0], 2: _WGD[1], 3: _WGD[2]}


def estimate(rho, pairs, pheno, scale="continuous"):
    y = pheno["phenotype"]
    y1 = y.loc[pairs["id1"].values].to_numpy()
    y2 = y.loc[pairs["id2"].values].to_numpy()
    d2 = (y1 - y2) ** 2
    x = pairs["dor"].map(_REL).to_numpy()
    sigma2 = float(y.var(ddof=1))          # phenotypic variance sigma_P^2
    # fix the intercept at the AE value 2*sigma_P^2 -> origin regression of
    # (d2 - 2*sigma_P^2) on w_G^d
    slope = float(((d2 - 2.0 * sigma2) * x).sum() / (x * x).sum())
    return {"V_G": -slope / (2.0 * sigma2)}
