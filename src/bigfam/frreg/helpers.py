"""
이진형 FR-regression을 위한 공통 헬퍼 함수들

Bivariate Normal CDF, Tetrachoric Correlation, Bivariate Probit 관련 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional, List, Tuple


# =============================================================================
# Gauss-Legendre Quadrature Points (10-point)
# =============================================================================

GAUSS_X = np.array([
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312,
    0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
    0.8650633666889845, 0.9739065285171717
])

GAUSS_W = np.array([
    0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
    0.2692667193099963, 0.2955242247147529,
    0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
    0.1494513491505806, 0.0666713443086881
])


# =============================================================================
# Shared Utility
# =============================================================================

def normalize_covariate_prefixes(
    df: pd.DataFrame,
    covariate_cols: Optional[List[str]]
    ) -> List[str]:
    """
    Normalize covariate names to base prefixes used in pair columns.

    Accepts any of:
    - age
    - vol_age
    - rel_age

    and returns:
    - age  (if both vol_age and rel_age exist in df)
    """
    if not covariate_cols:
        return []

    normalized: List[str] = []
    seen = set()

    for col in covariate_cols:
        base = str(col)
        if base.startswith('vol_') or base.startswith('rel_'):
            base = base[4:]

        if base in seen:
            continue

        vol_col = f'vol_{base}'
        rel_col = f'rel_{base}'
        if vol_col in df.columns and rel_col in df.columns:
            normalized.append(base)
            seen.add(base)

    return normalized


# =============================================================================
# Bivariate Normal CDF Functions
# =============================================================================

def bivariate_normal_cdf(x1: float, x2: float, rho: float) -> float:
    """
    이변량 정규 누적분포함수: P(X1 <= x1, X2 <= x2)
    Gauss-Legendre 적분을 사용한 빠른 계산
    """
    from scipy.special import ndtr
    
    rho = np.clip(rho, -0.9999, 0.9999)
    
    if abs(rho) < 1e-10:
        return ndtr(x1) * ndtr(x2)
    
    asr = np.arcsin(rho)
    t_vals = asr * (1 + GAUSS_X) / 2
    sin_t = np.sin(t_vals)
    cos_t_sq = np.maximum(np.cos(t_vals) ** 2, 1e-10)
    
    exponent = -(x1*x1 + x2*x2 - 2*x1*x2*sin_t) / (2 * cos_t_sq)
    integrand = np.exp(exponent)
    integral = np.sum(integrand * GAUSS_W) * asr / 2
    
    result = integral / (2 * np.pi) + ndtr(x1) * ndtr(x2)
    return np.clip(result, 0, 1)


def bvn_cdf_vectorized(x1: np.ndarray, x2: np.ndarray, rho: float) -> np.ndarray:
    """이변량 정규 CDF 벡터화 버전"""
    from scipy.special import ndtr
    
    x1 = np.atleast_1d(np.asarray(x1, dtype=float))
    x2 = np.atleast_1d(np.asarray(x2, dtype=float))
    rho = np.clip(rho, -0.9999, 0.9999)
    
    if abs(rho) < 1e-10:
        return ndtr(x1) * ndtr(x2)
    
    asr = np.arcsin(rho)
    a = x1[:, np.newaxis]
    b = x2[:, np.newaxis]
    
    t_vals = asr * (1 + GAUSS_X) / 2
    sin_t = np.sin(t_vals)
    cos_t_sq = np.maximum(np.cos(t_vals) ** 2, 1e-10)
    
    exponent = -(a*a + b*b - 2*a*b*sin_t) / (2 * cos_t_sq)
    integrand = np.exp(exponent)
    integral = np.sum(integrand * GAUSS_W, axis=1) * asr / 2
    
    result = integral / (2 * np.pi) + ndtr(x1) * ndtr(x2)
    return np.clip(result, 0, 1)


# =============================================================================
# Tetrachoric Correlation
# =============================================================================

def tetrachoric_correlation(n00: int, n01: int, n10: int, n11: int) -> Tuple[float, int]:
    """2x2 분할표로부터 테트라코릭 상관계수 계산"""
    N = n00 + n01 + n10 + n11
    if N == 0:
        return np.nan, 0
    
    # 연속성 보정
    if min(n00, n01, n10, n11) < 1:
        n00, n01, n10, n11 = n00 + 0.5, n01 + 0.5, n10 + 0.5, n11 + 0.5
        N = n00 + n01 + n10 + n11
    
    p1_plus = (n10 + n11) / N  # P(Y1=1)
    p_plus1 = (n01 + n11) / N  # P(Y2=1)
    
    if not (0 < p1_plus < 1 and 0 < p_plus1 < 1):
        return np.nan, int(N)
    
    T1 = norm.ppf(1 - p1_plus)
    T2 = norm.ppf(1 - p_plus1)
    p11_obs = n11 / N
    
    def objective(rho):
        p_both_below = bivariate_normal_cdf(T1, T2, rho)
        p11_pred = 1 - (1 - p1_plus) - (1 - p_plus1) + p_both_below
        return p11_pred - p11_obs
    
    try:
        rho = brentq(objective, -0.999, 0.999, xtol=1e-8)
        return rho, int(N)
    except (ValueError, RuntimeError):
        return np.nan, int(N)


# =============================================================================
# Contingency Table & Bivariate Probit
# =============================================================================

def compute_contingency_table(df: pd.DataFrame, vol_col: str, rel_col: str) -> Tuple[int, int, int, int]:
    """2x2 분할표 계산"""
    n00 = ((df[vol_col] == 0) & (df[rel_col] == 0)).sum()
    n01 = ((df[vol_col] == 0) & (df[rel_col] == 1)).sum()
    n10 = ((df[vol_col] == 1) & (df[rel_col] == 0)).sum()
    n11 = ((df[vol_col] == 1) & (df[rel_col] == 1)).sum()
    return n00, n01, n10, n11


def bivariate_probit_negloglik(
    params: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray
    ) -> float:
    """Bivariate probit 모델의 음의 로그우도 (벡터화)"""
    beta = params[:-1]
    rho = np.clip(params[-1], -0.999, 0.999)
    
    eta1 = X1 @ beta
    eta2 = X2 @ beta
    
    eps = 1e-10
    
    # 단변량 CDF
    phi1 = norm.cdf(-eta1)  # P(Y1=0)
    phi2 = norm.cdf(-eta2)  # P(Y2=0)
    
    # 이변량 CDF: P(Y1=0, Y2=0)
    p00 = bvn_cdf_vectorized(-eta1, -eta2, rho)
    
    # 각 결과의 확률
    p01 = np.clip(phi1 - p00, eps, 1 - eps)
    p10 = np.clip(phi2 - p00, eps, 1 - eps)
    p11 = np.clip(1 - phi1 - phi2 + p00, eps, 1 - eps)
    p00 = np.clip(p00, eps, 1 - eps)
    
    # 관측된 결과에 따른 확률 선택
    probs = (
        p00 * ((Y1 == 0) & (Y2 == 0)) +
        p01 * ((Y1 == 0) & (Y2 == 1)) +
        p10 * ((Y1 == 1) & (Y2 == 0)) +
        p11 * ((Y1 == 1) & (Y2 == 1))
    )
    
    return -np.sum(np.log(np.clip(probs, eps, 1 - eps)))
