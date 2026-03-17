"""
FR-reg (Familial Relationship Regression) 모듈

연속형/이진형 표현형에 대한 FR-regression을 수행합니다.
"""

from .continuous import fit_continuous_frreg, fit_continuous_frreg_volsummary
from .binary import (
    fit_binary_frreg,
    fit_binary_frreg_liability,
    fit_binary_frreg_robust
)

__all__ = [
    # Continuous
    'fit_continuous_frreg',
    'fit_continuous_frreg_volsummary',
    # Binary
    'fit_binary_frreg',
    'fit_binary_frreg_liability',  # Posterior Mean Liability + Vol-Bootstrap
    'fit_binary_frreg_robust',     # Bivariate Probit + Cluster-Robust SE
]


