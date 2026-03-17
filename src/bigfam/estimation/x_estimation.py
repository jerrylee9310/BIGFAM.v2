"""
X Estimation 모듈

obj2.py의 기능을 통합하여 환경 분산(V_X)을 추정합니다.
- estimateX: 단일 환경 분산 추정
- estimateXmXfR: 성별별 환경 분산 + 상관계수 추정
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


# =============================================================================
# Helper Functions
# =============================================================================

def _match_type(df_frreg: pd.DataFrame) -> pd.DataFrame:
    """FR-reg 결과의 컬럼 타입 정리"""
    col_dicts = {
        "DOR": int,
        "relationship": str,
        "sex_type": str,
        "slope": float,
        "se": float,
        "Erx": float,
        "n_pairs": int
    }
    
    for coln, col_type in col_dicts.items():
        if coln in df_frreg.columns:
            try:
                df_frreg = df_frreg.astype({coln: col_type})
            except (ValueError, TypeError):
                pass
    return df_frreg


def _resample_frreg_coefficients(
    df_frreg: pd.DataFrame,
    n_resample: int = 100
    ) -> pd.DataFrame:
    """
    FR-reg 결과로부터 slope를 정규분포에서 재추출하여 Bootstrap 샘플 생성.
    
    Args:
        df_frreg: FR-reg 요약 결과 (relationship별)
        n_resample: 각 relationship당 생성할 샘플 수
        
    Returns:
        DataFrame: relationship, DOR, sex_type, Erx, slope, block 컬럼 포함
    """
    dfs = []
    
    for _, row in df_frreg.iterrows():
        relationship = row.get('relationship', f"rel_{row.name}")
        
        # Gaussian 재추출
        resampled_slopes = np.random.normal(
            row['slope'],
            row['se'],
            size=n_resample
        )
        
        df_tmp = pd.DataFrame({
            "DOR": row.get('DOR', 1),
            "relationship": relationship,
            "sex_type": row.get('sex_type', 'MF'),
            "Erx": row.get('Erx', 0.5),
            "slope": resampled_slopes,
            "se": row.get('se', 1.0),  # Propagate SE
            "block": np.arange(n_resample)
        })
        
        dfs.append(df_tmp)
    
    if not dfs:
        return pd.DataFrame(columns=["DOR", "relationship", "sex_type", "Erx", "slope", "block"])
    
    return pd.concat(dfs, axis=0, ignore_index=True)


def _regress_out_mean(
    df_block: pd.DataFrame,
    bin_cols: List[str] = ["DOR"]
    ) -> pd.DataFrame:
    """
    각 그룹(bin)에서 평균을 회귀하여 잔차(residual)와 eta를 계산.
    
    Args:
        df_block: 하나의 block에 해당하는 resampled 데이터
        bin_cols: 그룹화 기준 컬럼 (예: ["DOR"] 또는 ["DOR", "sex_type"])
        
    Returns:
        DataFrame: eta, residual, tl 컬럼 추가됨
    """
    result_dfs = []
    
    for name, group in df_block.groupby(bin_cols):
        group = group.copy()
        try:
            ll = smf.ols(formula="slope ~ 1", data=group).fit()
            group["eta"] = 2**group["DOR"].iloc[0] * ll.params["Intercept"]
            group["residual"] = ll.resid
            group["tl"] = group["Erx"] - group["Erx"].mean()
            # SE is constant within group (from resampling), so just take first
            # But we need to make sure it aligns with residual index if needed.
            # Here group is a slice, so it's fine.
        except Exception:
            group["eta"] = 0
            group["residual"] = 0
            group["tl"] = 0
        result_dfs.append(group)
    
    if not result_dfs:
        return df_block
    
    return pd.concat(result_dfs, ignore_index=True)


def _get_meta_h2(df_frreg: pd.DataFrame) -> Tuple[float, float]:
    """
    IVW (Inverse Variance Weighting)를 사용한 메타 h2 추정.
    
    Returns:
        (meta_mean, meta_se)
    """
    def meta_h2(means, ses):
        ses = np.array(ses)
        means = np.array(means)
        
        # 0이거나 NaN인 SE 처리
        valid = (ses > 0) & ~np.isnan(ses) & ~np.isnan(means)
        if not valid.any():
            return np.mean(means), np.nan
            
        ses = ses[valid]
        means = means[valid]
        
        meta_se = np.sqrt(1 / np.sum(1 / ses**2))
        meta_mean = np.sum(means / ses**2) * (meta_se**2)
        
        return meta_mean, meta_se
    
    means_all = []
    ses_all = []
    
    for d in sorted(df_frreg["DOR"].unique()):
        df_d = df_frreg[df_frreg["DOR"] == d]
        means_d = df_d["slope"].to_numpy()
        ses_d = df_d["se"].to_numpy()
        
        # 2^d 스케일링
        mean_d, se_d = meta_h2(2**d * means_d, 2**d * ses_d)
        means_all.append(mean_d)
        ses_all.append(se_d)
    
    return meta_h2(means_all, ses_all)


# =============================================================================
# Loss Functions & Optimization
# =============================================================================

def _loss_func_x(x: float, df: pd.DataFrame, alpha: float) -> float:
    """
    단일 X 추정을 위한 Loss 함수.
    
    Loss = Σ(residual - tl × X)² + α × X²
    
    Args:
        x: 추정할 X 값
        df: residual, tl 컬럼을 포함한 데이터
        alpha: L2 정규화 가중치
        
    Returns:
        float: Loss 값
    """
    # Fidelity term
    loss_fid = np.sum((df["residual"] - df["tl"] * x) ** 2)
    
    # L2 regularization term
    loss_l2 = alpha * (x ** 2)
    
    return loss_fid + loss_l2


def _loss_func_x_sex_r(
    xs: np.ndarray,
    df: pd.DataFrame,
    alpha: float,
    r: float
    ) -> float:
    """
    성별별 X 추정을 위한 Loss 함수.
    
    Args:
        xs: [X_male, X_female]
        df: sex_type, residual, tl 컬럼 포함
        alpha: L2 정규화 가중치
        r: 남녀 간 상관계수
        
    Returns:
        float: Loss 값
    """
    x_male, x_female = xs
    
    is_malepair = (df["sex_type"] == "MM")
    is_femalepair = (df["sex_type"] == "FF")
    
    df_mm = df[is_malepair]
    df_mf = df[~(is_malepair | is_femalepair)]  # MF (includes FM)
    df_ff = df[is_femalepair]
    
    # Fidelity term
    loss_mm = np.sum((df_mm["residual"] - df_mm["tl"] * x_male) ** 2) if len(df_mm) > 0 else 0
    loss_mf = np.sum((df_mf["residual"] - df_mf["tl"] * r * np.sqrt(x_male * x_female)) ** 2) if len(df_mf) > 0 else 0
    loss_ff = np.sum((df_ff["residual"] - df_ff["tl"] * x_female) ** 2) if len(df_ff) > 0 else 0
    
    # L2 regularization
    loss_l2 = alpha * (x_male**2 + x_female**2)
    
    return loss_mm + loss_mf + loss_ff + loss_l2


def _loss_func_x_weighted(x: float, df: pd.DataFrame, alpha: float) -> float:
    """
    Weighted Loss for single X estimation.
    Loss = Σ w * (residual - tl * X)^2 + α * X^2
    where w = 1 / se^2 (normalized to mean=1)
    """
    # Calculate weights
    se = df["se"].replace(0, np.nan).fillna(1.0)
    weights = 1 / (se ** 2)
    # Normalize weights to have mean 1
    weights = weights / weights.mean()
    
    # Fidelity term
    loss_fid = np.sum(weights * (df["residual"] - df["tl"] * x) ** 2)
    
    # L2 regularization term
    loss_l2 = alpha * (x ** 2)
    
    return loss_fid + loss_l2


def _loss_func_x_sex_r_weighted(
    xs: np.ndarray,
    df: pd.DataFrame,
    alpha: float,
    r: float
    ) -> float:
    """
    Weighted Loss for sex-specific X estimation.
    """
    x_male, x_female = xs
    
    # Calculate weights
    se = df["se"].replace(0, np.nan).fillna(1.0)
    weights = 1 / (se ** 2)
    # Normalize weights to have mean 1
    weights = weights / weights.mean()
    
    df = df.copy()
    df['weight'] = weights
    
    is_malepair = (df["sex_type"] == "MM")
    is_femalepair = (df["sex_type"] == "FF")
    
    df_mm = df[is_malepair]
    df_mf = df[~(is_malepair | is_femalepair)]  # MF (includes FM)
    df_ff = df[is_femalepair]
    
    # Fidelity term
    loss_mm = np.sum(df_mm['weight'] * (df_mm["residual"] - df_mm["tl"] * x_male) ** 2) if len(df_mm) > 0 else 0
    loss_mf = np.sum(df_mf['weight'] * (df_mf["residual"] - df_mf["tl"] * r * np.sqrt(x_male * x_female)) ** 2) if len(df_mf) > 0 else 0
    loss_ff = np.sum(df_ff['weight'] * (df_ff["residual"] - df_ff["tl"] * x_female) ** 2) if len(df_ff) > 0 else 0
    
    # L2 regularization
    loss_l2 = alpha * (x_male**2 + x_female**2)
    
    return loss_mm + loss_mf + loss_ff + loss_l2


def _optimize_x(
    df_block: pd.DataFrame,
    alpha: float,
    use_weights: bool = False,
    lower_lim: float = 1e-6,
    upper_lim: float = 1
    ) -> object:
    """단일 X 최적화"""
    x0 = [0.01]
    bounds = [(lower_lim, upper_lim)]
    
    loss_func = _loss_func_x_weighted if use_weights else _loss_func_x
    
    result = minimize(
        fun=loss_func,
        x0=x0,
        args=(df_block, alpha),
        bounds=bounds,
        tol=1e-4
    )
    
    return result


def _optimize_x_sex_r(
    df_block: pd.DataFrame,
    alpha: float,
    r0: float,
    use_weights: bool = False,
    lower_lim: float = 1e-6,
    upper_lim: float = 1
    ) -> object:
    """성별별 X + 상관계수 최적화"""
    x0 = [0.01, 0.01]
    bounds = [(lower_lim, upper_lim), (lower_lim, upper_lim)]
    
    loss_func = _loss_func_x_sex_r_weighted if use_weights else _loss_func_x_sex_r
    
    result = minimize(
        fun=loss_func,
        x0=x0,
        args=(df_block, alpha, r0),
        bounds=bounds,
        tol=1e-6
    )
    
    return result


# =============================================================================
# Main Estimation Functions
# =============================================================================

def estimate_x_variance(
    df_frreg: pd.DataFrame,
    n_resample: int = 100,
    regout_bin: List[str] = ["DOR"],
    alpha_weight: float = 2.0,
    alpha_type: str = "lambda",
    use_weights: bool = False,
    verbose: bool = True
    ) -> Dict:
    """
    환경 분산 (V_X) 추정.
    
    Args:
        df_frreg: FR-reg 결과 (relationship별, DOR, Erx, slope, se 필수)
        n_resample: Bootstrap 샘플 수
        regout_bin: 잔차 계산 시 그룹화 기준
        alpha_weight: L2 정규화 가중치
        alpha_type: "lambda" (lambda에 의존) 또는 "fixed" (고정값)
        use_weights: Weighted Ridge Regression 사용 여부 (1/SE^2 가중치)
        verbose: 출력 여부
        
    Returns:
        Dict: V_X 추정 결과 (V_X, V_X_lower, V_X_upper, raw_df)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[X Estimation] Starting estimation")
        print(f"{'='*60}")
        print(f"  - Resample: {n_resample} iterations")
        print(f"  - Regout bin: {regout_bin}")
        print(f"  - Alpha weight: {alpha_weight}")
        print(f"  - Weighted Loss: {use_weights}")
    
    # 타입 정리
    df_frreg = _match_type(df_frreg.copy())
    
    # 메타 lambda 계산
    meta_lambda, _ = _get_meta_h2(df_frreg)
    
    if verbose:
        print(f"  - Meta λ (IVW): {meta_lambda:.4f}")
    
    # Bootstrap resampling
    df_lmbds = _resample_frreg_coefficients(df_frreg, n_resample=n_resample)
    
    raw_results = []
    
    iterator = range(n_resample)
    if verbose and n_resample > 10:
        iterator = tqdm(range(n_resample), desc="  Bootstrap")
    
    for ib in iterator:
        df_block = df_lmbds[df_lmbds["block"] == ib].copy()
        
        # 잔차 계산
        df_block = _regress_out_mean(df_block, bin_cols=regout_bin)
        
        # Alpha 계산
        if alpha_type == "lambda" and meta_lambda > 0:
            alpha = (1 / meta_lambda) ** alpha_weight
        else:
            alpha = alpha_weight
        
        if alpha < 0:
            continue
        
        # X 최적화
        result = _optimize_x(df_block, alpha, use_weights=use_weights)
        
        raw_results.append({
            "lambda": meta_lambda,
            "alpha": alpha,
            "X": result.x[0]
        })
    
    df_raw = pd.DataFrame(raw_results)
    
    if len(df_raw) == 0:
        if verbose:
            print("  Warning: all bootstrap iterations failed")
        return {"V_X": np.nan, "V_X_lower": np.nan, "V_X_upper": np.nan}
    
    V_X = np.median(df_raw["X"])
    V_X_lower = np.percentile(df_raw["X"], 2.5)
    V_X_upper = np.percentile(df_raw["X"], 97.5)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[X Estimation] Done ✓")
        print(f"  - V_X: {V_X:.4f} (95% CI: [{V_X_lower:.4f}, {V_X_upper:.4f}])")
        print(f"{'='*60}")
    
    return {
        "V_X": V_X,
        "V_X_lower": V_X_lower,
        "V_X_upper": V_X_upper,
        "raw_df": df_raw
    }


def estimate_sex_specific_x(
    df_frreg: pd.DataFrame,
    n_resample: int = 100,
    regout_bin: List[str] = ["DOR", "sex_type"],
    alpha_weight: float = 2.0,
    alpha_type: str = "lambda",
    n_r_grid: int = 11,
    use_weights: bool = False,
    verbose: bool = True
    ) -> Dict:
    """
    성별별 환경 분산 (V_X_male, V_X_female) + 상관계수 (r) 추정.
    
    Args:
        df_frreg: FR-reg 결과 (relationship별, DOR, Erx, sex_type, slope, se 필수)
        n_resample: Bootstrap 샘플 수
        regout_bin: 잔차 계산 시 그룹화 기준
        alpha_weight: L2 정규화 가중치
        alpha_type: "lambda" 또는 "fixed"
        n_r_grid: r 그리드 수 (-1 ~ 1)
        use_weights: Weighted Ridge Regression 사용 여부 (1/SE^2 가중치)
        verbose: 출력 여부
        
    Returns:
        Dict: 성별별 V_X + r 추정 결과
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Sex-Specific X Estimation] Starting estimation")
        print(f"{'='*60}")
        print(f"  - Resample: {n_resample} iterations")
        print(f"  - Regout bin: {regout_bin}")
        print(f"  - Alpha weight: {alpha_weight}")
        print(f"  - r Grid: {n_r_grid} points")
        print(f"  - Weighted Loss: {use_weights}")
    
    # 타입 정리
    df_frreg = _match_type(df_frreg.copy())
    
    # sex_type 확인
    if 'sex_type' not in df_frreg.columns:
        raise ValueError("The 'sex_type' column is required.")
    
    # 메타 lambda 계산
    meta_lambda, _ = _get_meta_h2(df_frreg)
    
    if verbose:
        print(f"  - Meta λ (IVW): {meta_lambda:.4f}")
    
    # Bootstrap resampling
    df_lmbds = _resample_frreg_coefficients(df_frreg, n_resample=n_resample)
    
    raw_results = []
    
    iterator = range(n_resample)
    if verbose and n_resample > 10:
        iterator = tqdm(range(n_resample), desc="  Bootstrap")
    
    for ib in iterator:
        df_block = df_lmbds[df_lmbds["block"] == ib].copy()
        
        # 잔차 계산
        df_block = _regress_out_mean(df_block, bin_cols=regout_bin)
        
        # Alpha 계산
        if alpha_type == "lambda" and meta_lambda > 0:
            alpha = (1 / meta_lambda) ** alpha_weight
        else:
            alpha = alpha_weight
        
        if alpha < 0:
            continue
        
        # r 그리드 서치
        best_result = None
        best_loss = np.inf
        
        for r0 in np.linspace(-1, 1, n_r_grid):
            result = _optimize_x_sex_r(df_block, alpha, r0, use_weights=use_weights)
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = {
                    "lambda": meta_lambda,
                    "alpha": alpha,
                    "X_male": result.x[0],
                    "X_female": result.x[1],
                    "r": r0,
                    "loss": result.fun
                }
        
        if best_result:
            raw_results.append(best_result)
    
    df_raw = pd.DataFrame(raw_results)
    
    if len(df_raw) == 0:
        if verbose:
            print("  Warning: all bootstrap iterations failed")
        return {
            "V_X_male": np.nan, "V_X_female": np.nan, "r": np.nan
        }
    
    X_male = np.median(df_raw["X_male"])
    X_male_lower = np.percentile(df_raw["X_male"], 2.5)
    X_male_upper = np.percentile(df_raw["X_male"], 97.5)
    
    X_female = np.median(df_raw["X_female"])
    X_female_lower = np.percentile(df_raw["X_female"], 2.5)
    X_female_upper = np.percentile(df_raw["X_female"], 97.5)
    
    r_est = np.median(df_raw["r"])
    r_lower = np.percentile(df_raw["r"], 2.5)
    r_upper = np.percentile(df_raw["r"], 97.5)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Sex-Specific X Estimation] Done ✓")
        print(f"  - V_X (Male): {X_male:.4f} (95% CI: [{X_male_lower:.4f}, {X_male_upper:.4f}])")
        print(f"  - V_X (Female): {X_female:.4f} (95% CI: [{X_female_lower:.4f}, {X_female_upper:.4f}])")
        print(f"  - r: {r_est:.4f} (95% CI: [{r_lower:.4f}, {r_upper:.4f}])")
        print(f"{'='*60}")
    
    return {
        "V_X_male": X_male,
        "V_X_male_lower": X_male_lower,
        "V_X_male_upper": X_male_upper,
        "V_X_female": X_female,
        "V_X_female_lower": X_female_lower,
        "V_X_female_upper": X_female_upper,
        "r": r_est,
        "r_lower": r_lower,
        "r_upper": r_upper,
        "raw_df": df_raw
    }
