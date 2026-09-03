"""Phase 3 profile-likelihood CI for w_S.

For each w on WS_PROFILE, fit the fixed-w NNLS GLS (same solver as refit) and read the
loss ell = e^T Sigma^-1 e. The profile 95% w-CI is {w : ell <= ell_min + 3.84}.

Whether that CI excludes 0.5 is the identifiability read-out: at w_S = 0.5 the shared-env
column is exactly twice the genetic one, so X(w) is rank 1 and (V_G, V_S) are not
separable. Excluding 0.5 is the LR test of H0: w_S = 0.5 at chi^2_{1,.95} -- the
threshold comes from the test, not from tuning. No label is emitted; where to cut the
interval belongs to the analysis, not the estimator.

Grid uses UN-ridged A (no refit ridge) so the loss matches the validated reference.
"""
from __future__ import annotations

import numpy as np

from ..config import WS_PROFILE, CHI2_95
from ..core.nnls import nnls_2d


def profile_ci(rho_hat, Sigma_hat):
    """rho_hat: (3,), Sigma_hat: (3, 3) -> (wci_lo, wci_hi)."""
    rho_hat = np.asarray(rho_hat, dtype=float)
    Sinv = np.linalg.inv(np.asarray(Sigma_hat, dtype=float))
    K = len(WS_PROFILE)
    ell = np.zeros(K)
    for k, ws in enumerate(WS_PROFILE):
        X = np.array([[0.5, 1.0], [0.25, ws], [0.125, ws * ws]])
        beta = nnls_2d(X.T @ Sinv @ X, X.T @ Sinv @ rho_hat)
        e = rho_hat - X @ beta
        ell[k] = e @ Sinv @ e

    in_ci = ell <= ell.min() + CHI2_95
    lo = WS_PROFILE[in_ci.argmax()]
    hi = WS_PROFILE[K - 1 - in_ci[::-1].argmax()]
    return float(lo), float(hi)


# ── self-check ────────────────────────────────────────────────────────────────
def _demo():
    # clean shared-env signal at w=0.7: CI should sit around 0.7 and exclude 0.5
    X = np.array([[0.5, 1.0], [0.25, 0.7], [0.125, 0.49]])
    Sig = np.diag([0.004, 0.004, 0.004]) ** 2
    lo, hi = profile_ci(X @ np.array([0.5, 0.2]), Sig)
    assert 0.01 <= lo <= hi <= 0.99, (lo, hi)
    assert not (lo <= 0.5 <= hi), f"identified case should exclude 0.5: {(lo, hi)}"

    # no shared env (V_S=0): w_S is unplaceable, so the CI must cover 0.5
    lo0, hi0 = profile_ci(X @ np.array([0.5, 0.0]), Sig)
    assert lo0 <= 0.5 <= hi0, f"V_S=0 should cover 0.5: {(lo0, hi0)}"
    print("profile_ci self-checks passed", (lo, hi), (lo0, hi0))


if __name__ == "__main__":
    _demo()
