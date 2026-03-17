"""
Slope Test 모듈

w_S = 2 가설 검정을 위한 Slope Test 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Tuple, Dict, Optional, Literal
from ..utils.stats import ensure_log_columns


# =============================================================================
# Slope Test
# =============================================================================

def _coerce_rho_frame(df_frreg: pd.DataFrame) -> pd.DataFrame:
    """
    FR-reg 결과를 rho 기반 컬럼으로 정규화.

    내부 계산은 rho/log_rho를 사용하고, 외부 호환성을 위해 slope alias를 보존한다.
    """
    if "DOR" not in df_frreg.columns:
        raise ValueError("Missing required columns: ['DOR']")

    df = df_frreg.copy()

    has_rho_input = False
    has_log_rho_input = False

    if {"rho", "se"}.issubset(df.columns):
        df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
        df["se"] = pd.to_numeric(df["se"], errors="coerce")
        has_rho_input = True
    elif {"slope", "se"}.issubset(df.columns):
        df["rho"] = pd.to_numeric(df["slope"], errors="coerce")
        df["se"] = pd.to_numeric(df["se"], errors="coerce")
        has_rho_input = True
    elif {"log_rho", "log_rho_se"}.issubset(df.columns):
        df["log_rho"] = pd.to_numeric(df["log_rho"], errors="coerce")
        df["log_rho_se"] = pd.to_numeric(df["log_rho_se"], errors="coerce")
        has_log_rho_input = True
    elif {"log_rho", "log_se"}.issubset(df.columns):
        df["log_rho"] = pd.to_numeric(df["log_rho"], errors="coerce")
        df["log_rho_se"] = pd.to_numeric(df["log_se"], errors="coerce")
        has_log_rho_input = True
    elif {"log_slope", "log_se"}.issubset(df.columns):
        df["log_rho"] = pd.to_numeric(df["log_slope"], errors="coerce")
        df["log_rho_se"] = pd.to_numeric(df["log_se"], errors="coerce")
        has_log_rho_input = True
    else:
        raise ValueError(
            "Missing required columns: expected one of (rho/se), (slope/se), "
            "(log_rho/log_rho_se), (log_rho/log_se), or (log_slope/log_se)."
        )

    df["DOR"] = pd.to_numeric(df["DOR"], errors="coerce")

    if has_rho_input:
        df["log_rho"] = np.where(
            df["rho"] > 0,
            np.log2(df["rho"]),
            np.nan
        )
        df["log_rho_se"] = np.where(
            (df["rho"] > 0) & (df["se"] > 0),
            df["se"] / (df["rho"] * np.log(2)),
            np.nan
        )
    else:
        df["rho"] = 2 ** df["log_rho"]
        df["se"] = np.where(
            (df["rho"] > 0) & (df["log_rho_se"] > 0),
            df["log_rho_se"] * df["rho"] * np.log(2),
            np.nan
        )

    if "slope" not in df.columns:
        df["slope"] = df["rho"]
    if "log_slope" not in df.columns:
        df["log_slope"] = df["log_rho"]
    if "log_se" not in df.columns:
        df["log_se"] = df["log_rho_se"]

    return df


def run_slope_test(
    df_frreg: pd.DataFrame,
    method: Literal['direct', 'resample', 'lognormal', 'known_var'] = 'known_var',
    n_bootstrap: int = 100,
    verbose: bool = True
    ) -> Tuple[str, Dict]:
    """
    Slope Test를 수행하여 w_S = 2 가설을 검정합니다.
    
    이론적 배경:
    - H0 (w_S = 2) 하에서: log2(λ_d) = -d + const
    - 따라서 -DOR에 대한 회귀 기울기가 1이면 H0 채택
    
    Args:
        df_frreg: FR-reg 결과 DataFrame
            - 권장: `DOR`, `rho`, `se`
            - 또는: `DOR`, `log_rho`, `log_se`
            - 또는: `DOR`, `slope`, `se` (내부 호환성)
        method: 
            - 'direct': WLS + statsmodels (잔차로 σ² 추정)
            - 'resample': 정규분포 bootstrap (기존 방식, Jensen 편향 있음)
            - 'lognormal': Log-normal bootstrap (양수 제약 + Jensen 적절히 반영)
            - 'known_var': Jensen 보정 + Fixed-scale WLS (권장, 소수 데이터에서 안정적)
        n_bootstrap: Bootstrap 반복 횟수 (CI 계산용)
        verbose: 상세 출력 여부
        
    Returns:
        Tuple[str, Dict]: (significance, result_dict)
        - significance: "fast" (w_S > 2), "slow" (w_S < 2), "similar" (유의하지 않음)
        - result_dict: 상세 결과 (slope, slope_se, slope_lower, slope_upper, intercept, intercept_se, intercept_lower, intercept_upper)
    """
    df_work = _coerce_rho_frame(df_frreg)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Slope Test] Starting analysis")
        print(f"{'='*60}")
        print(f"  - Method: {method.upper()}")
        if method.upper() in ["RESAMPLE", "LOGNORMAL"]:
            print(f"  - Bootstrap: {n_bootstrap} iterations")
    
    if method == 'direct':
        return _slope_test_direct(df_work, n_bootstrap, verbose)
    elif method == 'resample':
        return _slope_test_resample(df_work, n_bootstrap, verbose)
    elif method == 'lognormal':
        return _slope_test_lognormal(df_work, n_bootstrap, verbose)
    elif method == 'known_var':
        return _slope_test_known_var(df_work, verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'direct', 'resample', 'lognormal', or 'known_var'.")


def _slope_test_direct(
    df_frreg: pd.DataFrame,
    n_bootstrap: int = 100,  # 이 파라미터는 호환성을 위해 유지하지만 사용하지 않음
    verbose: bool = True
    ) -> Tuple[str, Dict]:
    """
    Direct 방식 Slope Test (권장)
    
    log(E[λ])를 사용하여 WLS 회귀를 수행합니다.
    SE는 WLS 모델에서 직접 추정합니다 (Delta Method).
    
    Jensen's Inequality에 의한 편향이 없습니다.
    """
    if verbose:
        print(f"\n  [Step 1] Preparing data")
    
    # 필수 컬럼 확인
    required_cols = ['DOR', 'log_rho', 'log_rho_se']
    missing = [c for c in required_cols if c not in df_frreg.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # NA 제거
    df = df_frreg.dropna(subset=['log_rho', 'log_rho_se']).copy()
    
    if len(df) < 2:
        if verbose:
            print(f"    Warning: insufficient data (DOR groups: {len(df)} < minimum 2)")
        return "similar", {
            'slope': np.nan, 'slope_se': np.nan, 'slope_lower': np.nan, 'slope_upper': np.nan,
            'intercept': np.nan, 'intercept_se': np.nan, 'intercept_lower': np.nan, 'intercept_upper': np.nan
        }
    
    df['neg_dor'] = -df['DOR'].astype(float)
    
    if verbose:
        print(f"    - Valid DOR groups: {len(df)}")
        for _, row in df.iterrows():
            print(f"      • DOR {int(row['DOR'])}: log₂(ρ) = {row['log_rho']:.4f} ± {row['log_rho_se']:.4f}")
    
    # Step 2: Weighted Least Squares
    if verbose:
        print(f"\n  [Step 2] WLS regression (inverse-variance weighting)")
    
    try:
        # 가중치: 1 / SE^2 (inverse variance weighting)
        weights = 1 / (df['log_rho_se'] ** 2)
        weights = weights.replace([np.inf, -np.inf], 1.0)
        
        model = smf.wls('log_rho ~ 1 + neg_dor', data=df, weights=weights).fit()
        
        # Point estimates
        point_slope = model.params['neg_dor']
        point_intercept = model.params['Intercept']
        
        # Standard errors (from WLS model - Delta Method에 의해 자동 전파됨)
        se_slope = model.bse['neg_dor']
        se_intercept = model.bse['Intercept']
        
        # 95% CI (정규분포 가정)
        z = 1.96
        lower = point_slope - z * se_slope
        upper = point_slope + z * se_slope
        
        lower_intercept = point_intercept - z * se_intercept
        upper_intercept = point_intercept + z * se_intercept
        
        if verbose:
            print(f"    - Regression: log₂(ρ) = {point_intercept:.4f} + {point_slope:.4f} × (-DOR)")
            print(f"    - Slope: {point_slope:.4f}")
            print(f"    - Standard error: {se_slope:.4f}")
            print(f"    - 95% CI: [{lower:.4f}, {upper:.4f}]")
            print(f"    - Reference: expected slope under H0 (w_S=2) is 1.0")
            
    except Exception as e:
        if verbose:
            print(f"    Warning: WLS failed: {e}")
        return "similar", {
            'slope': np.nan, 'slope_se': np.nan, 'slope_lower': np.nan, 'slope_upper': np.nan,
            'intercept': np.nan, 'intercept_se': np.nan, 'intercept_lower': np.nan, 'intercept_upper': np.nan
        }
    
    # Step 3: Significance 판정
    if verbose:
        print(f"\n  [Step 3] Determining significance")
    
    sig = "similar"
    if lower > 1:
        sig = "fast"
        if verbose:
            print(f"    - CI lower bound ({lower:.4f}) > 1.0")
            print(f"    → Result: 'fast' (w_S > 2, faster environmental decay)")
    elif upper < 1:
        sig = "slow"
        if verbose:
            print(f"    - CI upper bound ({upper:.4f}) < 1.0")
            print(f"    → Result: 'slow' (w_S < 2, slower environmental decay)")
    else:
        if verbose:
            print(f"    - CI [{lower:.4f}, {upper:.4f}] includes 1.0")
            print(f"    → Result: 'similar' (no significant difference from w_S = 2)")
    
    result = {
        'slope': point_slope,
        'slope_se': se_slope,
        'slope_lower': lower,
        'slope_upper': upper,
        'intercept': point_intercept,
        'intercept_se': se_intercept,
        'intercept_lower': lower_intercept,
        'intercept_upper': upper_intercept
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Slope Test] Done ✓")
        print(f"  - Slope: {point_slope:.4f} ± {se_slope:.4f}")
        print(f"  - 95% CI: [{lower:.4f}, {upper:.4f}]")
        print(f"  - Significance: {sig}")
        print(f"{'='*60}")
    
    return sig, result


def _slope_test_known_var(
    df_frreg: pd.DataFrame,
    verbose: bool = True
    ) -> Tuple[str, Dict]:
    """
    Known-Variance WLS 방식 Slope Test (권장).
    
    특징:
    1. Jensen bias correction: log(λ̂) + SE²/(2λ̂²·ln2)
    2. Fixed-scale WLS: 잔차로 σ² 추정 안 함 (known variance 가정)
    3. z-기반 CI: 소수 데이터에서도 안정적
    
    장점:
    - d=1,2,3 같은 소수 데이터에서 CI가 과도하게 커지지 않음
    - Type I error가 안정적
    - Jensen's inequality 편향 보정
    """
    if verbose:
        print(f"\n  [Step 1] Preparing data (known-variance WLS)")
        print(f"    ✓ Jensen bias correction + fixed-scale WLS")
    
    # 필수 컬럼 확인: slope, se (λ-스케일)
    required_cols = ['DOR', 'rho', 'se']
    missing = [c for c in required_cols if c not in df_frreg.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df_frreg.dropna(subset=['rho', 'se']).copy()
    df = df[df['rho'] > 0].copy()  # log 때문에 양수만
    
    if len(df) < 2:
        if verbose:
            print(f"    Warning: insufficient data (DOR groups: {len(df)} < minimum 2)")
        return "similar", {
            'slope': np.nan, 'slope_se': np.nan, 'slope_lower': np.nan, 'slope_upper': np.nan,
            'intercept': np.nan, 'intercept_se': np.nan, 'intercept_lower': np.nan, 'intercept_upper': np.nan
        }
    
    df['neg_dor'] = -df['DOR'].astype(float)
    
    # λ-스케일에서 가져오기
    rho = df['rho'].values.astype(float)
    se_lambda = df['se'].values.astype(float)
    
    # SE가 너무 작거나 NaN인 경우 처리
    se_lambda = np.where((se_lambda <= 0) | np.isnan(se_lambda), rho * 0.1, se_lambda)
    
    # Step 1: log2 변환 + Jensen bias correction
    # E[log(λ̂)] ≈ log(λ) - SE²/(2λ²)
    # 보정: log_bc = log(λ̂) + SE²/(2λ̂²·ln2)  [log₂ 기준]
    log_rho = np.log2(rho)
    bias_correction = (se_lambda ** 2) / (2 * (rho ** 2) * np.log(2))
    log_rho_bc = log_rho + bias_correction
    
    # Step 2: log-스케일 SE (델타 방법)
    # SE(log₂(λ̂)) ≈ SE_λ / (λ̂ · ln2)
    log_rho_se = se_lambda / (rho * np.log(2))
    
    if verbose:
        print(f"    - Valid DOR groups: {len(df)}")
        print(f"\n  [Step 2] Jensen bias correction + log-scale SE")
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"      • DOR {int(row['DOR'])}: ρ={rho[i]:.4f} ± {se_lambda[i]:.4f}")
            print(f"        log₂(ρ)={log_rho[i]:.4f}, bias_corr={bias_correction[i]:.4f}, log_bc={log_rho_bc[i]:.4f}, log_se={log_rho_se[i]:.4f}")
    
    # Step 3: Fixed-scale WLS (known variance)
    # Cov(β̂) = (X'WX)⁻¹  (σ² 추정 없이!)
    if verbose:
        print(f"\n  [Step 3] Fixed-scale WLS (without residual variance estimation)")
    
    try:
        X = np.column_stack([np.ones(len(df)), df['neg_dor'].values])
        y = log_rho_bc
        w = 1.0 / (log_rho_se ** 2)
        W = np.diag(w)
        
        # WLS 추정: β̂ = (X'WX)⁻¹ X'Wy
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta = np.linalg.solve(XtWX, XtWy)
        
        # Known-variance: Cov(β̂) = (X'WX)⁻¹ (σ²=1 가정, 잔차 추정 없음)
        cov_beta = np.linalg.inv(XtWX)
        se_beta = np.sqrt(np.diag(cov_beta))
        
        point_intercept = beta[0]
        point_slope = beta[1]
        se_intercept = se_beta[0]
        se_slope = se_beta[1]
        
        # 95% CI (z-기반, t 아님!)
        z = 1.96
        lower = point_slope - z * se_slope
        upper = point_slope + z * se_slope
        lower_intercept = point_intercept - z * se_intercept
        upper_intercept = point_intercept + z * se_intercept
        
        if verbose:
            print(f"    - Regression: log₂(ρ)_bc = {point_intercept:.4f} + {point_slope:.4f} × (-DOR)")
            print(f"    - Slope: {point_slope:.4f}")
            print(f"    - Standard error: {se_slope:.4f}")
            print(f"    - 95% CI (z-based): [{lower:.4f}, {upper:.4f}]")
            print(f"    - Reference: expected slope under H0 (w_S=2) is 1.0")
            
    except Exception as e:
        if verbose:
            print(f"    Warning: WLS failed: {e}")
        return "similar", {
            'slope': np.nan, 'slope_se': np.nan, 'slope_lower': np.nan, 'slope_upper': np.nan,
            'intercept': np.nan, 'intercept_se': np.nan, 'intercept_lower': np.nan, 'intercept_upper': np.nan
        }
    
    # Step 4: Significance 판정
    if verbose:
        print(f"\n  [Step 4] Determining significance")
    
    sig = "similar"
    if lower > 1:
        sig = "fast"
        if verbose:
            print(f"    - CI lower bound ({lower:.4f}) > 1.0")
            print(f"    → Result: 'fast' (w_S > 2, faster environmental decay)")
    elif upper < 1:
        sig = "slow"
        if verbose:
            print(f"    - CI upper bound ({upper:.4f}) < 1.0")
            print(f"    → Result: 'slow' (w_S < 2, slower environmental decay)")
    else:
        if verbose:
            print(f"    - CI [{lower:.4f}, {upper:.4f}] includes 1.0")
            print(f"    → Result: 'similar' (no significant difference from w_S = 2)")
    
    result = {
        'slope': point_slope,
        'slope_se': se_slope,
        'slope_lower': lower,
        'slope_upper': upper,
        'intercept': point_intercept,
        'intercept_se': se_intercept,
        'intercept_lower': lower_intercept,
        'intercept_upper': upper_intercept
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Slope Test] Done ✓ (Known-Variance WLS)")
        print(f"  - Slope: {point_slope:.4f} ± {se_slope:.4f}")
        print(f"  - 95% CI: [{lower:.4f}, {upper:.4f}]")
        print(f"  - Significance: {sig}")
        print(f"{'='*60}")
    
    return sig, result


def _slope_test_resample(
    df_frreg: pd.DataFrame,
    n_bootstrap: int = 100,
    verbose: bool = True
    ) -> Tuple[str, Dict]:
    """
    Resample 방식 Slope Test (기존 방식)
    
    E[log(λ)]를 사용합니다.
    주의: Jensen's Inequality에 의한 편향이 있을 수 있습니다.
    """
    if verbose:
        print(f"\n  [Step 1] Preparing data (resample mode)")
        print(f"    Warning: this method may be biased by Jensen's inequality")
    
    # 필수 컬럼 확인
    required_cols = ['DOR', 'rho', 'se']
    missing = [c for c in required_cols if c not in df_frreg.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df_frreg.dropna(subset=['rho', 'se']).copy()
    
    if len(df) < 2:
        if verbose:
            print(f"    Warning: insufficient data (DOR groups: {len(df)} < minimum 2)")
        return "similar", {
            'slope': np.nan, 'slope_se': np.nan, 'slope_lower': np.nan, 'slope_upper': np.nan,
            'intercept': np.nan, 'intercept_se': np.nan, 'intercept_lower': np.nan, 'intercept_upper': np.nan
        }
    
    if verbose:
        print(f"    - Valid DOR groups: {len(df)}")
    
    # Step 2: Bootstrap resampling
    if verbose:
        print(f"\n  [Step 2] Bootstrap resampling ({n_bootstrap} iterations)")
    
    np.random.seed(42)
    boot_slopes = []
    boot_intercepts = []
    
    for _ in range(n_bootstrap):
        try:
            # 각 DOR에서 rho를 정규분포에서 재추출
            resampled_rhos = np.random.normal(
                df['rho'].values,
                df['se'].values
            )
            
            # 음수 필터링
            resampled_rhos = np.maximum(resampled_rhos, 1e-6)
            
            # log 변환
            log_rhos = np.log2(resampled_rhos)
            neg_dor = -df['DOR'].values.astype(float)
            
            # OLS 회귀
            X = np.column_stack([np.ones(len(neg_dor)), neg_dor])
            y = log_rhos
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            boot_slopes.append(beta[1])  # slope coefficient
            boot_intercepts.append(beta[0]) # intercept coefficient
        except Exception:
            continue
    
    boot_slopes = np.array(boot_slopes)
    boot_intercepts = np.array(boot_intercepts)
    
    if len(boot_slopes) > 0:
        mean_slope = np.mean(boot_slopes)
        se_slope = np.std(boot_slopes)
        lower = np.percentile(boot_slopes, 2.5)
        upper = np.percentile(boot_slopes, 97.5)
        
        mean_intercept = np.mean(boot_intercepts)
        se_intercept = np.std(boot_intercepts)
        lower_intercept = np.percentile(boot_intercepts, 2.5)
        upper_intercept = np.percentile(boot_intercepts, 97.5)
    else:
        mean_slope = np.nan
        se_slope = np.nan
        lower = np.nan
        upper = np.nan
        
        mean_intercept = np.nan
        se_intercept = np.nan
        lower_intercept = np.nan
        upper_intercept = np.nan
    
    if verbose:
        print(f"    - Successful iterations: {len(boot_slopes)}")
        print(f"    - Mean slope: {mean_slope:.4f}")
        print(f"    - 95% CI: [{lower:.4f}, {upper:.4f}]")
    
    # Step 3: Significance 판정
    if verbose:
        print(f"\n  [Step 3] Determining significance")
    
    sig = "similar"
    if not np.isnan(lower) and lower > 1:
        sig = "fast"
        if verbose:
            print(f"    → Result: 'fast' (w_S > 2)")
    elif not np.isnan(upper) and upper < 1:
        sig = "slow"
        if verbose:
            print(f"    → Result: 'slow' (w_S < 2)")
    else:
        if verbose:
            print(f"    → Result: 'similar' (no significant difference)")
    
    result = {
        'slope': mean_slope,
        'slope_se': se_slope,
        'slope_lower': lower,
        'slope_upper': upper,
        'intercept': mean_intercept,
        'intercept_se': se_intercept,
        'intercept_lower': lower_intercept,
        'intercept_upper': upper_intercept
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Slope Test] Done ✓")
        print(f"  - Slope: {mean_slope:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
        print(f"  - Significance: {sig}")
        print(f"{'='*60}")
    
    return sig, result


def _slope_test_lognormal(
    df_frreg: pd.DataFrame,
    n_bootstrap: int = 100,
    verbose: bool = True
    ) -> Tuple[str, Dict]:
    """
    Log-normal Parametric Bootstrap 방식 Slope Test.
    
    λ를 log-normal 분포로 모델링하여:
    1. 양수 제약을 자연스럽게 만족
    2. Jensen's inequality를 적절히 반영
    3. clip으로 인한 인위적 편향 제거
    
    방법:
    - mean(μ)과 SE로부터 log-normal 파라미터 (m, s²) 계산
    - s² = log(1 + SE²/μ²)
    - m = log(μ) - s²/2
    - LogNormal(m, s)에서 샘플링 → log 변환 → 회귀 → bootstrap CI
    """
    if verbose:
        print(f"\n  [Step 1] Preparing data (log-normal bootstrap)")
        print(f"    ✓ Log-normal distribution: preserves positivity and accounts for Jensen's inequality")
    
    # 필수 컬럼 확인
    required_cols = ['DOR', 'rho', 'se']
    missing = [c for c in required_cols if c not in df_frreg.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df_frreg.dropna(subset=['rho', 'se']).copy()
    
    if len(df) < 2:
        if verbose:
            print(f"    Warning: insufficient data (DOR groups: {len(df)} < minimum 2)")
        return "similar", {
            'slope': np.nan, 'slope_se': np.nan, 'slope_lower': np.nan, 'slope_upper': np.nan,
            'intercept': np.nan, 'intercept_se': np.nan, 'intercept_lower': np.nan, 'intercept_upper': np.nan
        }
    
    # 양수 확인 및 필터링
    df = df[df['rho'] > 0].copy()
    if len(df) < 2:
        if verbose:
            print(f"    Warning: insufficient positive slope values")
        return "similar", {
            'slope': np.nan, 'slope_se': np.nan, 'slope_lower': np.nan, 'slope_upper': np.nan,
            'intercept': np.nan, 'intercept_se': np.nan, 'intercept_lower': np.nan, 'intercept_upper': np.nan
        }
    
    if verbose:
        print(f"    - Valid DOR groups: {len(df)}")
    
    # Log-normal 파라미터 계산
    # μ = mean, SE = standard error of mean estimate
    # LogNormal(m, s²): E[X] = exp(m + s²/2), Var(X) = (exp(s²) - 1) * exp(2m + s²)
    # 역변환: s² = log(1 + Var/μ²), m = log(μ) - s²/2
    mu = df['rho'].values.astype(float)
    se = df['se'].values.astype(float)
    
    # SE가 너무 작거나 NaN인 경우 처리
    se = np.where((se <= 0) | np.isnan(se), mu * 0.1, se)  # 기본값: 10% of mean
    
    # Log-normal parameters
    s_sq = np.log(1 + (se / mu) ** 2)
    m = np.log(mu) - s_sq / 2
    s = np.sqrt(s_sq)
    
    if verbose:
        print(f"\n  [Step 2] Computing log-normal parameters")
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"      • DOR {int(row['DOR'])}: μ={mu[i]:.4f}, SE={se[i]:.4f} → m={m[i]:.4f}, s={s[i]:.4f}")
    
    # Step 3: Bootstrap resampling
    if verbose:
        print(f"\n  [Step 3] Log-normal bootstrap resampling ({n_bootstrap} iterations)")
    
    np.random.seed(42)
    boot_slopes = []
    boot_intercepts = []
    neg_dor = -df['DOR'].values.astype(float)
    
    for _ in range(n_bootstrap):
        try:
            # Log-normal 샘플링
            resampled_slopes = np.random.lognormal(mean=m, sigma=s)
            
            # log2 변환
            log_slopes = np.log2(resampled_slopes)
            
            # OLS 회귀: log2(λ) ~ 1 + (-DOR)
            X = np.column_stack([np.ones(len(neg_dor)), neg_dor])
            y = log_slopes
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            boot_intercepts.append(beta[0])  # intercept
            boot_slopes.append(beta[1])  # slope coefficient
        except Exception:
            continue
    
    boot_slopes = np.array(boot_slopes)
    boot_intercepts = np.array(boot_intercepts)
    
    if len(boot_slopes) > 0:
        mean_slope = np.mean(boot_slopes)
        se_slope = np.std(boot_slopes)
        lower = np.percentile(boot_slopes, 2.5)
        upper = np.percentile(boot_slopes, 97.5)
        
        mean_intercept = np.mean(boot_intercepts)
        se_intercept = np.std(boot_intercepts)
        lower_intercept = np.percentile(boot_intercepts, 2.5)
        upper_intercept = np.percentile(boot_intercepts, 97.5)
    else:
        mean_slope = se_slope = lower = upper = np.nan
        mean_intercept = se_intercept = lower_intercept = upper_intercept = np.nan
    
    if verbose:
        print(f"    - Successful iterations: {len(boot_slopes)}")
        print(f"    - Mean slope: {mean_slope:.4f} ± {se_slope:.4f}")
        print(f"    - 95% CI: [{lower:.4f}, {upper:.4f}]")
    
    # Step 4: Significance 판정
    if verbose:
        print(f"\n  [Step 4] Determining significance")
    
    sig = "similar"
    if not np.isnan(lower) and lower > 1:
        sig = "fast"
        if verbose:
            print(f"    - CI lower bound ({lower:.4f}) > 1.0")
            print(f"    → Result: 'fast' (w_S > 2, faster environmental decay)")
    elif not np.isnan(upper) and upper < 1:
        sig = "slow"
        if verbose:
            print(f"    - CI upper bound ({upper:.4f}) < 1.0")
            print(f"    → Result: 'slow' (w_S < 2, slower environmental decay)")
    else:
        if verbose:
            print(f"    - CI [{lower:.4f}, {upper:.4f}] includes 1.0")
            print(f"    → Result: 'similar' (no significant difference from w_S = 2)")
    
    result = {
        'slope': mean_slope,
        'slope_se': se_slope,
        'slope_lower': lower,
        'slope_upper': upper,
        'intercept': mean_intercept,
        'intercept_se': se_intercept,
        'intercept_lower': lower_intercept,
        'intercept_upper': upper_intercept
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Slope Test] Done ✓ (Log-normal Bootstrap)")
        print(f"  - Slope: {mean_slope:.4f} ± {se_slope:.4f}")
        print(f"  - 95% CI: [{lower:.4f}, {upper:.4f}]")
        print(f"  - Significance: {sig}")
        print(f"{'='*60}")
    
    return sig, result
