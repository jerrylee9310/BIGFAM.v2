"""Single source of truth for every constant in the package.

D = 3 is fixed: design matrix X(w_S), slope design H, and the contrast
features D2/D3 are all hard-coded for three DOR levels. Generalising to D != 3
requires re-deriving those (plus the profile design). This package supports
D = 3 only.
"""
from __future__ import annotations

import numpy as np

# ── model structure (fixed) ──────────────────────────────────────────────────
D = 3                                   # number of DOR levels (fixed)

# slope GLS design (Phase 2 1a): rows = DOR 1,2,3
H = np.column_stack([np.ones(3), [-1.0, -2.0, -3.0]])  # (3, 2), cols [1, -d]

# ── grid / statistical constants ─────────────────────────────────────────────
_WS_FULL = np.linspace(0.01, 0.99, 99)
WS_PROFILE = _WS_FULL[np.abs(_WS_FULL - 0.5) > 1e-9]   # drop 0.5 (singular)
CHI2_95 = 3.84                          # chi^2_{1,0.95}, profile CI cut
Z95 = 1.96
CLIP = (0.01, 0.99)                     # w_S clip
RHO_CLIP = (-0.9999, 0.9999)            # bivariate probit rho bound

# ── numerical-stability eps ──────────────────────────────────────────────────
RIDGE_SLOPE = 1e-8                      # slope GLS A
RIDGE_REFIT = 1e-9                      # phase3 refit relative ridge (x tr(A))
RIDGE_NNLS = 1e-12                      # nnls unconstrained solve
RHO_FLOOR = 1e-6                        # rho_hat <= 0 clip (protect log features)
PROB_FLOOR = 1e-300                     # bivariate normal cdf floor
HESS_EPS = 1e-4                         # binary numerical Hessian central diff
BVN_GL_NODES = 20                       # Gauss-Legendre nodes, vectorized bvn CDF (err ~2e-10 @ |r|=0.95)

# ── training (offline, wide DGP) ─────────────────────────────────────────────
SEED = 42
N_SIM = 40_000                          # wide-DGP training draws (continuous w_S)
SIGMA_HI = 0.10                         # wide-DGP sd upper bound
ALPHAS_CV = [0.01, 0.1, 1.0, 10.0, 100.0]
