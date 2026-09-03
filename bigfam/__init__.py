"""BIGFAM: split relative-pair similarity into genetic (V_G) and shared-environment
(V_S) variance, from phenotypes and family relationships alone -- no genotypes.

    import bigfam
    from bigfam.io import load_artifacts

    calib  = load_artifacts("artifacts/")
    rho    = bigfam.estimate_rho(pairs, cov, pheno, "continuous")  # Phase 1
    ws     = bigfam.estimate_ws(rho, calib)                        # Phase 2
    result = bigfam.decompose(rho, ws)                             # Phase 3

result.V_G, result.V_S with conditional SEs, plus w_s_cal and its 95% CI.
Runnable end-to-end example: examples/quickstart.py.
"""
from __future__ import annotations

from .types import RhoEstimate, WsEstimate, Decomposition, CalibrationCoef
from .phase1 import estimate_rho
from .phase2 import estimate_ws
from .phase3 import decompose
from . import io

__version__ = "0.1.0"

__all__ = [
    "estimate_rho", "estimate_ws", "decompose",
    "RhoEstimate", "WsEstimate", "Decomposition", "CalibrationCoef",
    "io",
]
