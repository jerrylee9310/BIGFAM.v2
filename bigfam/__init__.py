"""bigfam: relative-pair V_G/V_S variance decomposition (Phase 1->2->3).

    rho   = estimate_rho(pairs, cov, pheno, pheno_type)   # Phase 1
    ws    = estimate_ws(rho, calib)                       # Phase 2
    result = decompose(rho, ws)                           # Phase 3  (V_G, V_S, w-CI)
"""
from __future__ import annotations

from .types import (
    RhoEstimate, WsEstimate, Decomposition, CalibrationCoef,
)
from .phase1 import estimate_rho
from .phase2 import estimate_ws
from .phase3 import decompose
from . import io

__all__ = [
    "estimate_rho", "estimate_ws", "decompose",
    "RhoEstimate", "WsEstimate", "Decomposition", "CalibrationCoef",
    "io",
]
