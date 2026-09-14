"""BIGFAM: split relative-pair similarity into genetic (V_A) and shared-environment
(V_C) variance, from phenotypes and family relationships alone -- no genotypes.

    import bigfam
    from bigfam.io import load_artifacts

    calib  = load_artifacts()
    rho    = bigfam.estimate_rho(pairs, cov, pheno, "continuous")  # Phase 1
    wc     = bigfam.estimate_wc(rho, calib)                        # Phase 2
    result = bigfam.decompose(rho, wc)                             # Phase 3

result.V_A, result.V_C with conditional SEs, plus w_c_cal and its 95% CI.
Runnable end-to-end example: examples/quickstart.py.
"""
from __future__ import annotations

from .types import RhoEstimate, WcEstimate, Decomposition, CalibrationCoef
from .phase1 import estimate_rho
from .phase2 import estimate_wc
from .phase3 import decompose
from . import io

__version__ = "0.1.0"

__all__ = [
    "estimate_rho", "estimate_wc", "decompose",
    "RhoEstimate", "WcEstimate", "Decomposition", "CalibrationCoef",
    "io",
]
