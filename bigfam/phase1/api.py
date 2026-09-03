"""Phase 1 unified entry point.

    relative pairs -> RhoEstimate (rho_hat, Sigma_hat)

Branches on phenotype kind, then the output carries no trace of it.
Reference: docs/method/phase1.md 'Unified Evidence Object'.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..config import COV_COLS, D as D_DEFAULT
from ..core.linalg import nearest_psd
from ..types import RhoEstimate
from . import pairs as _pairs
from . import continuous as _cont
from . import binary as _bin


def estimate_rho(pairs, cov, pheno, pheno_type: str,
                 cov_cols=COV_COLS, D: int = D_DEFAULT) -> RhoEstimate:
    """Estimate (rho_hat, Sigma_hat) from relative pairs.

    pairs: DataFrame with columns id1, id2, dor.
    cov:   DataFrame indexed by id with cov_cols.
    pheno: DataFrame indexed by id with a 'phenotype' column.
    pheno_type: "continuous" | "binary".
    """
    binary = pheno_type == "binary"
    table = _pairs.build_pair_table(pairs, cov, pheno, cov_cols, binary=binary)
    data = _pairs.flip_concat(table, cov_cols)

    if binary:
        rho_hat, Sigma_hat, _ = _bin.estimate_rho_sigma(data, cov_cols, D)
    elif pheno_type == "continuous":
        gamma_hat = _cont.estimate_gamma(table, pheno, cov, cov_cols)
        rho_hat, Sigma_hat = _cont.estimate_rho_sigma(data, gamma_hat, cov_cols, D)
    else:
        raise ValueError(
            f"pheno_type must be 'continuous' or 'binary', got {pheno_type!r}")

    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    min_eig = float(np.linalg.eigvalsh(0.5 * (Sigma_hat + Sigma_hat.T))[0])
    if min_eig < 0.0:
        warnings.warn(
            f"Sigma_hat not PSD (min eigenvalue {min_eig:.2e}); projecting to "
            f"nearest PSD. Indicates high pair-overlap / low-signal data.",
            stacklevel=2)
        Sigma_hat = nearest_psd(Sigma_hat)

    return RhoEstimate(np.asarray(rho_hat, dtype=float), Sigma_hat, D)
