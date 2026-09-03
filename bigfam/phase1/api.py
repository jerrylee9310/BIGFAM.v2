"""Phase 1: relative pairs -> (rho_hat, Sigma_hat).

The only layer that knows whether the phenotype is continuous or binary; the
output carries no trace of it.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..config import D as D_DEFAULT
from ..core.linalg import nearest_psd
from ..types import RhoEstimate
from . import pairs as _pairs
from . import continuous as _cont
from . import binary as _bin


def estimate_rho(pairs, cov, pheno, pheno_type: str,
                 cov_cols=(), D: int = D_DEFAULT) -> RhoEstimate:
    """Estimate the per-DOR similarity rho_hat and its covariance Sigma_hat.

    pairs: DataFrame with columns id1, id2, dor (dor = 1, 2, 3). Direction does
           not matter and repeated pairs are dropped (with a warning).
    cov:   DataFrame indexed by unique id, one column per name in cov_cols.
    pheno: DataFrame indexed by unique id with a 'phenotype' column. Binary
           phenotypes must be coded 0/1; the scale of a continuous phenotype
           does not matter (rho_hat is invariant to shifting and rescaling it).
    pheno_type: "continuous" or "binary".
    cov_cols: covariate columns to adjust for. Empty by default: nothing is
           adjusted away except an intercept.
    """
    cov_cols = list(cov_cols)
    binary = pheno_type == "binary"
    if binary:
        seen = set(np.unique(pheno["phenotype"].dropna().values).tolist())
        if not seen <= {0, 1}:
            raise ValueError(
                f"binary phenotype must be coded 0/1, found {sorted(seen)[:5]}. "
                "Other codings run without error and return meaningless rho")
    table = _pairs.build_pair_table(pairs, cov, pheno, cov_cols, binary=binary)

    missing = sorted(set(range(1, D + 1)) - set(table["dor"].astype(int)))
    if missing:
        raise ValueError(
            f"no usable pairs left at dor {missing}; all of 1..{D} are needed. "
            "Pairs are kept only when both members appear in cov and pheno")

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
