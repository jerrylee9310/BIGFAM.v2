"""
개인 단위 데이터 전처리 모듈

공변량 보정, 표준화, 아웃라이어 제거 처리를 담당합니다.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Optional, List


# =============================================================================
# 1. 개인 단위 전처리 (Individual Cleaning)
# =============================================================================

def clean_individual_continuous(
    df: pd.DataFrame,
    trait_col: str,
    cov_cols: Optional[List[str]] = None,
    outlier_threshold: float = 3.0,
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    개인 수준에서 연속형 형질의 전처리를 완결합니다.
    (공변량 보정 -> 표준화 -> 아웃라이어 제거)
    
    Args:
        df: 표현형 + 공변량이 병합된 개인 데이터프레임
        trait_col: 형질 컬럼명
        cov_cols: 공변량 컬럼 목록
        outlier_threshold: 아웃라이어 임계값 (Z-score)
        verbose: 출력 여부
        
    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임 (해당 iid들만 남음)
    """
    if verbose:
        print(f"\n[Individual Clean] Starting preprocessing for '{trait_col}'")
    
    # 1. Regress out (Residualize)
    if cov_cols and len(cov_cols) > 0:
        formula = f"{trait_col} ~ 1 + " + " + ".join(cov_cols)
        if verbose:
            print(f"  - Step 1: residualize with covariates (intercept + {cov_cols})")
    else:
        formula = f"{trait_col} ~ 1"
        if verbose:
            print("  - Step 1: centering (intercept only)")
            
    model = smf.ols(formula, data=df).fit()
    resid = model.resid
    
    # 2. Standardize (Z-score)
    z_score = (resid - resid.mean()) / resid.std()
    if verbose:
        print(f"  - Step 2: standardize residuals (Z-score)")
    
    # 3. Remove Outliers
    outliers = z_score.abs() > outlier_threshold
    n_outliers = outliers.sum()
    
    df_clean = df.copy()
    df_clean[trait_col] = z_score
    df_clean = df_clean[~outliers].copy()
    
    # 4. Re-standardize
    # Outlier 제거로 인해 틀어진 Mean/Var를 다시 0/1로 맞춤
    current_vals = df_clean[trait_col]
    df_clean[trait_col] = (current_vals - current_vals.mean()) / current_vals.std()
    
    if verbose:
        if n_outliers > 0:
            print(f"  - Step 3: remove outliers ({n_outliers:,} rows, |Z| > {outlier_threshold})")
        print(f"  - Step 4: re-standardize (mean=0, variance=1)")
        print(f"  - Final valid individuals: {len(df_clean):,}")
        print(f"[Individual Clean] Done ✓")
        
    return df_clean
