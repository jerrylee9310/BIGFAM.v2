"""Shared evidence step: relative pairs -> (rho_hat, Sigma_hat).

Every method in this folder consumes this single object -- all methods differ
only in what they do with the same relative-correlation summary statistic.
continuous uses observed-scale rho; binary uses bivariate-probit liability
(tetrachoric) rho.

Called once per replicate; Falconer/HE/BIGFAM all reuse the result (the
binary probit MLE is not re-solved per method). There are no covariates in
this simulation (COV_COLS=[]), so BIGFAM solves the intercept-only design
(W=[1]) -- cov is a 0-column frame aligned to pheno's index, built on the fly.
"""
from __future__ import annotations

import bigfam

from config import COV_COLS


def compute(pairs, pheno, scale: str):
    """scale in {continuous, binary} -> RhoEstimate(rho_hat, Sigma_hat, D=3)."""
    cov = pheno[[]]                         # 0-col, same index as pheno (intercept-only)
    return bigfam.estimate_rho(pairs, cov, pheno, scale, COV_COLS)
