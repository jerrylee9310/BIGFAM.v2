"""
Variance Component Estimation 모듈

V_G (유전분산), V_S (환경분산), w (환경감쇄율) 추정 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Literal, List
from ..utils.pairs import drop_symmetric_duplicates
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# Helper Functions for CV-based Estimation
# =============================================================================

def _resample_coefficients(
    df_frreg: pd.DataFrame, 
    n_resample: int = 100,
    use_bias_correction: bool = False
    ) -> pd.DataFrame:
    """
    FR-reg 결과로부터 slope를 resampling하여 가상의 데이터셋을 생성합니다.
    
    Args:
        df_frreg: FR-reg 결과 (DOR, log_slope, log_se 필수)
        n_resample: 각 DOR당 생성할 샘플 수
        
    Returns:
        DataFrame with columns: DOR, idx, slope
    """
    dfs = []
    work = _prepare_variance_log_input(df_frreg, use_bias_correction=use_bias_correction)
    work = work[["DOR", "log_rho_obs", "log_rho_obs_se"]].copy()
    
    for dor in work["DOR"].dropna().unique():
        row = work[work["DOR"] == dor].iloc[0]
        
        log_slope = row['log_rho_obs']
        log_se = row.get('log_rho_obs_se', 0.1)
        if pd.isna(log_se) or log_se <= 0:
            log_se = 0.1
        
        if pd.isna(log_slope):
            continue
        
        # Resample on log scale, then convert back
        resampled_log = np.random.normal(log_slope, log_se, size=n_resample)
        resampled_slope = 2 ** resampled_log
        
        df_tmp = pd.DataFrame({
            "DOR": dor,
            "idx": range(n_resample),
            "slope": resampled_slope
        })
        dfs.append(df_tmp)
    
    if not dfs:
        return pd.DataFrame(columns=["DOR", "idx", "slope"])
    
    return pd.concat(dfs, axis=0, ignore_index=True)


def _label_resampled_coefficients(df_lmbds: pd.DataFrame, n_block: int = 10) -> pd.DataFrame:
    """
    Resampled coefficient에 CV block 라벨을 할당합니다.
    
    Args:
        df_lmbds: Resampled DataFrame (DOR, idx, slope)
        n_block: CV fold 수
        
    Returns:
        DataFrame with 'block' column added
    """
    df_lmbds = df_lmbds.copy()
    df_lmbds["block"] = -1  # Initialize with -1
    
    for d in df_lmbds["DOR"].unique():
        is_dor = df_lmbds["DOR"] == d
        n_samples = df_lmbds[is_dor].shape[0]
        
        if n_samples > 0:
            # Create block labels (repeat if needed, shuffle)
            block_labels = [i % n_block for i in range(n_samples)]
            np.random.shuffle(block_labels)
            df_lmbds.loc[is_dor, "block"] = block_labels
    
    return df_lmbds


def _prepare_variance_log_input(
    df_frreg: pd.DataFrame,
    use_bias_correction: bool = False
) -> pd.DataFrame:
    """
    variance estimation용 로그 값/SE 정규화.

    - 기본: log_rho = log_rho/log_slope 중 가용한 값 사용
    - log_bc 사용 시:
      1) log_rho_bc 컬럼이 있으면 우선 사용
      2) 없으면 rho/se 또는 log_rho/log_rho_se 기반으로 Jensen 보정 근사 적용
    """
    df = df_frreg.copy()
    df["DOR"] = pd.to_numeric(df["DOR"], errors="coerce")

    if "log_rho" in df.columns:
        log_rho = pd.to_numeric(df["log_rho"], errors="coerce")
    elif "log_slope" in df.columns:
        log_rho = pd.to_numeric(df["log_slope"], errors="coerce")
    elif "rho" in df.columns:
        rho = pd.to_numeric(df["rho"], errors="coerce")
        log_rho = np.where(rho > 0, np.log2(rho), np.nan)
    else:
        log_rho = np.nan

    if "log_rho_se" in df.columns:
        log_rho_se = pd.to_numeric(df["log_rho_se"], errors="coerce")
    elif "log_se" in df.columns:
        log_rho_se = pd.to_numeric(df["log_se"], errors="coerce")
    elif "se" in df.columns and "slope" in df.columns:
        rho = pd.to_numeric(df["slope"], errors="coerce")
        se = pd.to_numeric(df["se"], errors="coerce")
        log_rho_se = np.where((rho > 0) & (se > 0), se / (rho * np.log(2)), np.nan)
    elif "se" in df.columns and "rho" in df.columns:
        rho = pd.to_numeric(df["rho"], errors="coerce")
        se = pd.to_numeric(df["se"], errors="coerce")
        log_rho_se = np.where((rho > 0) & (se > 0), se / (rho * np.log(2)), np.nan)
    else:
        log_rho_se = np.nan

    if use_bias_correction:
        if "log_rho_bc" in df.columns:
            log_rho_obs = pd.to_numeric(df["log_rho_bc"], errors="coerce")
        elif {"rho", "se"}.issubset(df.columns):
            rho = pd.to_numeric(df["rho"], errors="coerce")
            se = pd.to_numeric(df["se"], errors="coerce")
            log_rho_obs = np.where(
                (rho > 0),
                np.log2(rho) + np.where((se > 0) & (~np.isnan(se)),
                                         (se ** 2) / (2 * (rho ** 2) * np.log(2)),
                                         0.0),
                np.nan
            )
        elif "log_rho" in df.columns and "log_rho_se" in df.columns:
            rho = 2 ** pd.to_numeric(df["log_rho"], errors="coerce")
            se = np.where(
                (rho > 0) & (log_rho_se > 0),
                log_rho_se * rho * np.log(2),
                np.nan
            )
            log_rho_obs = np.where(
                (rho > 0),
                np.log2(rho) + np.where((~np.isnan(se)) & (se > 0),
                                       (se ** 2) / (2 * (rho ** 2) * np.log(2)),
                                       0.0),
                np.nan
            )
        else:
            log_rho_obs = log_rho
    else:
        log_rho_obs = log_rho

    df["log_rho_obs"] = pd.to_numeric(log_rho_obs, errors="coerce")
    df["log_rho_obs_se"] = pd.to_numeric(log_rho_se, errors="coerce")
    return df


def _loss_func(params, df_lmbds_w):
    """
    CV 최적화를 위한 loss function (log scale).
    
    Args:
        params: [V_G, V_S]
        df_lmbds_w: (df_lmbds, w) tuple
        
    Returns:
        Sum of squared log errors
    """
    V_G, V_S = params
    df_lmbds, w = df_lmbds_w
    
    df_lmbds = df_lmbds.copy()
    df_lmbds["slope"] = df_lmbds["slope"].astype(float)
    df_lmbds["log_slope"] = np.log2(df_lmbds["slope"])
    df_lmbds["slope_pred"] = (0.5)**df_lmbds["DOR"] * V_G + w**(df_lmbds["DOR"] - 1) * V_S
    
    # Avoid log of non-positive
    df_lmbds["slope_pred"] = df_lmbds["slope_pred"].clip(lower=1e-10)
    df_lmbds["log_slope_pred"] = np.log2(df_lmbds["slope_pred"])
    
    return ((df_lmbds["log_slope"] - df_lmbds["log_slope_pred"])**2).sum()


def _loss_func_wls(params, df_lmbds_w):
    """
    CV 최적화를 위한 loss function (linear scale, weighted).
    
    Args:
        params: [V_G, V_S]
        df_lmbds_w: (df_lmbds, w) tuple. df_lmbds must have 'weight' column.
        
    Returns:
        Weighted sum of squared errors
    """
    V_G, V_S = params
    df_lmbds, w = df_lmbds_w
    
    # Prediction on linear scale
    slope_pred = (0.5)**df_lmbds["DOR"] * V_G + w**(df_lmbds["DOR"] - 1) * V_S
    
    # Weighted SSE
    # Weight should be 1/Var. 
    # If weight column exists, use it. Otherwise default to 1.
    weights = df_lmbds.get("weight", 1.0)
    
    return (weights * (df_lmbds["slope"] - slope_pred)**2).sum()


def _estimate_cv_fold_worker(
    dor_values: np.ndarray,
    train_mean: np.ndarray,
    train_weight: np.ndarray,
    test_mean: np.ndarray,
    test_weight: np.ndarray,
    w_range: Tuple[float, ...],
    loss_method: str
    ) -> Optional[Dict[str, float]]:
    """
    한 개 fold(CV split)에 대해 w 후보를 순회하며 최적 파라미터(V_G, V_S, w)를 찾습니다.

    속도 최적화:
    - df_train/df_test 전체 샘플을 직접 최소화에 넣지 않고,
      DOR별 mean/weight(=count 또는 sum(weight))로 요약한 뒤 최소화를 수행합니다.
    - log(loss)에서 샘플별 SSE는 DOR별 mean(log_slope)만으로 argmin이 동일합니다
      (within-DOR 분산 항은 파라미터와 무관한 상수).
    """
    from scipy.optimize import minimize

    if dor_values is None or len(dor_values) == 0:
        return None

    dor_values = np.asarray(dor_values, dtype=float)
    train_mean = np.asarray(train_mean, dtype=float)
    train_weight = np.asarray(train_weight, dtype=float)
    test_mean = np.asarray(test_mean, dtype=float)
    test_weight = np.asarray(test_weight, dtype=float)

    if not (
        dor_values.shape == train_mean.shape == train_weight.shape == test_mean.shape == test_weight.shape
    ):
        raise ValueError("fold summary arrays must have the same shape")

    if float(train_weight.sum()) <= 0.0 or float(test_weight.sum()) <= 0.0:
        return None

    a = (0.5) ** dor_values
    dor_minus_1 = dor_values - 1.0
    ln2 = float(np.log(2.0))

    best_loss = np.inf
    best_params = None
    x0 = np.array([0.5, 0.1], dtype=float)

    for w in w_range:
        try:
            b = np.power(float(w), dor_minus_1)

            if loss_method == "linear_wls":
                def fun(x: np.ndarray) -> float:
                    v_g_est, v_s_est = float(x[0]), float(x[1])
                    pred = a * v_g_est + b * v_s_est
                    resid = train_mean - pred
                    return float(np.sum(train_weight * resid * resid))

                def jac(x: np.ndarray) -> np.ndarray:
                    v_g_est, v_s_est = float(x[0]), float(x[1])
                    pred = a * v_g_est + b * v_s_est
                    resid = train_mean - pred
                    grad_vg = -2.0 * float(np.sum(train_weight * resid * a))
                    grad_vs = -2.0 * float(np.sum(train_weight * resid * b))
                    return np.array([grad_vg, grad_vs], dtype=float)
            else:
                def fun(x: np.ndarray) -> float:
                    v_g_est, v_s_est = float(x[0]), float(x[1])
                    pred = a * v_g_est + b * v_s_est
                    pred = np.clip(pred, a_min=1e-10, a_max=None)
                    log_pred = np.log2(pred)
                    resid = train_mean - log_pred
                    return float(np.sum(train_weight * resid * resid))

                def jac(x: np.ndarray) -> np.ndarray:
                    v_g_est, v_s_est = float(x[0]), float(x[1])
                    pred = a * v_g_est + b * v_s_est
                    pred = np.clip(pred, a_min=1e-10, a_max=None)
                    log_pred = np.log2(pred)
                    resid = train_mean - log_pred
                    factor = (-2.0 * train_weight * resid) / (pred * ln2)
                    grad_vg = float(np.sum(factor * a))
                    grad_vs = float(np.sum(factor * b))
                    return np.array([grad_vg, grad_vs], dtype=float)

            result = minimize(
                fun=fun,
                x0=x0,
                jac=jac,
                bounds=((1e-6, 1), (1e-6, 1)),
                method='L-BFGS-B'
            )

            v_g_est, v_s_est = float(result.x[0]), float(result.x[1])
            if np.isfinite(v_g_est) and np.isfinite(v_s_est):
                x0 = np.array([v_g_est, v_s_est], dtype=float)  # warm start

            pred_test = a * v_g_est + b * v_s_est

            if loss_method == "linear_wls":
                resid_test = test_mean - np.clip(pred_test, a_min=0.0, a_max=None)
                test_loss = float(np.sum(test_weight * resid_test * resid_test))
            else:
                pred_test = np.clip(pred_test, a_min=1e-10, a_max=None)
                log_pred_test = np.log2(pred_test)
                resid_test = test_mean - log_pred_test
                test_loss = float(np.sum(test_weight * resid_test * resid_test))

            if test_loss < best_loss:
                best_loss = float(test_loss)
                best_params = (float(v_g_est), float(v_s_est), float(w))
        except Exception:
            continue

    if best_params is None:
        return None

    return {
        'V_G': best_params[0],
        'V_S': best_params[1],
        'w': best_params[2]
    }


# =============================================================================
# Variance Component Estimation
# =============================================================================

def _loss_function_log(V_G: float, V_S: float, w: float, df_frreg: pd.DataFrame) -> float:
    """
    Weighted loss function using log_slope (inverse variance weighting).
    
    이론적 모델:
        log2(E[λ_d]) = log2(2^{-d} * V_G + w^{d-1} * V_S)
    
    Args:
        V_G: 유전 분산
        V_S: 공유 환경 분산
        w: 환경 감쇄 인자 (1/w_s, 0~1 사이 값)
        df_frreg: FR-reg 결과 (log_slope, log_se 필수)
        
    Returns:
        Weighted sum of squared errors
    """
    loss = 0
    df = _prepare_variance_log_input(df_frreg, use_bias_correction=False)

    for _, row in df.iterrows():
        d = row['DOR']
        log_rho_obs = row['log_rho_obs']
        log_se = row.get('log_rho_obs_se', 0.1)
        
        if np.isnan(log_rho_obs):
            continue
        
        # 이론적 예측: λ_d = (1/2)^d * V_G + w^{d-1} * V_S
        slope_pred = (0.5)**d * V_G + w**(d - 1) * V_S
        if slope_pred <= 0:
            return 1e10  # 유효하지 않은 예측
        
        log_slope_pred = np.log2(slope_pred)
        weight = 1 / (log_se ** 2) if log_se > 0 and not np.isnan(log_se) else 1
        loss += weight * (log_rho_obs - log_slope_pred)**2
    
    return loss


def _optimize_variance_for_w(w: float, df_frreg: pd.DataFrame) -> Tuple[float, float, float]:
    """
    주어진 w 값에서 V_G, V_S를 최적화합니다.
    
    Args:
        w: 고정된 w 값
        df_frreg: FR-reg 결과
        
    Returns:
        Tuple of (V_G, V_S, loss)
    """
    from scipy.optimize import minimize
    
    x0 = [0.5, 0.1]  # Initial V_G, V_S
    bounds = ((1e-6, 1.0), (1e-6, 1.0))
    
    def objective(params):
        return _loss_function_log(params[0], params[1], w, df_frreg)
    
    try:
        result = minimize(
            fun=objective,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            tol=1e-6
        )
        return result.x[0], result.x[1], result.fun
    except Exception:
        return np.nan, np.nan, np.inf


def _loss_function_3param(params: np.ndarray, df_frreg: pd.DataFrame) -> float:
    """
    3-파라미터 (V_G, V_S, w) 최적화를 위한 loss function.
    
    Args:
        params: [V_G, V_S, w] 배열
        df_frreg: FR-reg 결과
        
    Returns:
        Weighted sum of squared errors
    """
    V_G, V_S, w = params
    return _loss_function_log(V_G, V_S, w, df_frreg)


def _estimate_variance_delta(
    df_frreg: pd.DataFrame,
    slope_significance: str = "similar",
    verbose: bool = True
    ) -> Dict:
    """
    Delta Method를 이용한 분산 성분 추정 (Hessian에서 SE 추출).
    
    3-파라미터 (V_G, V_S, w)를 동시에 최적화하고,
    Hessian 역행렬로부터 표준오차를 계산합니다.
    
    Args:
        df_frreg: FR-reg 결과
        slope_significance: slope test 결과 (초기값 설정에 사용)
        verbose: 상세 출력 여부
        
    Returns:
        Dictionary with V_G, V_S, w estimates and their SEs/CIs
    """
    from scipy.optimize import minimize
    import warnings
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Variance Components] Starting estimation (Delta Method)")
        print(f"{'='*60}")
    
    df = df_frreg.dropna(subset=['log_slope']).copy()
    
    if len(df) < 2:
        if verbose:
            print(f"  Warning: insufficient data (DOR groups: {len(df)} < minimum 2)")
        return None
    
    # Set initial w based on slope test
    sig = (slope_significance or "similar").lower()
    if sig == "high":
        sig = "fast"
    elif sig == "low":
        sig = "slow"

    if sig == "fast":
        w0 = 0.25  # w_s > 2
    elif sig == "slow":
        w0 = 0.75  # w_s < 2
    else:
        w0 = 0.50  # w_s ~ 2
    
    x0 = [0.3, 0.1, w0]  # Initial [V_G, V_S, w]
    bounds = ((1e-6, 1.0), (1e-6, 1.0), (0.01, 0.99))  # Bounds
    
    if verbose:
        print(f"\n  [Step 1] Three-parameter optimization")
        print(f"    - Initial values: V_G={x0[0]:.2f}, V_S={x0[1]:.2f}, w={x0[2]:.2f}")
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                fun=_loss_function_3param,
                x0=x0,
                args=(df,),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
        
        V_G_opt, V_S_opt, w_opt = result.x
        loss = result.fun
        
        if verbose:
            print(f"    - Optimized values: V_G={V_G_opt:.4f}, V_S={V_S_opt:.4f}, w={w_opt:.4f}")
            print(f"    - Loss: {loss:.6f}")
            print(f"    - Converged: {result.success}")
        
    except Exception as e:
        if verbose:
            print(f"  Warning: optimization failed: {e}")
        return None
    
    # Step 2: Extract SE from Hessian inverse
    if verbose:
        print(f"\n  [Step 2] Extracting SE with the delta method")
    
    try:
        # L-BFGS-B returns hess_inv as LinearOperator, convert to dense
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            hess_inv = np.array(result.hess_inv.todense())
            
            se_V_G = np.sqrt(max(hess_inv[0, 0], 0))
            se_V_S = np.sqrt(max(hess_inv[1, 1], 0))
            se_w = np.sqrt(max(hess_inv[2, 2], 0))
            
            if verbose:
                print(f"    - SE(V_G): {se_V_G:.4f}")
                print(f"    - SE(V_S): {se_V_S:.4f}")
                print(f"    - SE(w): {se_w:.4f}")
        else:
            # Fallback: no Hessian available
            se_V_G = se_V_S = se_w = np.nan
            if verbose:
                print(f"    Warning: Hessian is unavailable (SE = NaN)")
                
    except Exception as e:
        se_V_G = se_V_S = se_w = np.nan
        if verbose:
            print(f"    Warning: failed to compute SE: {e}")
    
    # Calculate 95% CI
    V_G_lower = max(0, V_G_opt - 1.96 * se_V_G) if not np.isnan(se_V_G) else np.nan
    V_G_upper = min(1, V_G_opt + 1.96 * se_V_G) if not np.isnan(se_V_G) else np.nan
    V_S_lower = max(0, V_S_opt - 1.96 * se_V_S) if not np.isnan(se_V_S) else np.nan
    V_S_upper = min(1, V_S_opt + 1.96 * se_V_S) if not np.isnan(se_V_S) else np.nan
    w_lower = max(0.01, w_opt - 1.96 * se_w) if not np.isnan(se_w) else np.nan
    w_upper = min(0.99, w_opt + 1.96 * se_w) if not np.isnan(se_w) else np.nan
    
    res = {
        'V_G': V_G_opt,
        'V_G_se': se_V_G,
        'V_G_lower': V_G_lower,
        'V_G_upper': V_G_upper,
        'V_S': V_S_opt,
        'V_S_se': se_V_S,
        'V_S_lower': V_S_lower,
        'V_S_upper': V_S_upper,
        'w': w_opt,
        'w_se': se_w,
        'w_lower': w_lower,
        'w_upper': w_upper,
        'loss': loss,
        'method': 'delta'
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Variance Components] Done ✓ (Delta Method)")
        print(f"  - V_G: {V_G_opt:.4f} ± {se_V_G:.4f} (95% CI: [{V_G_lower:.4f}, {V_G_upper:.4f}])")
        print(f"  - V_S: {V_S_opt:.4f} ± {se_V_S:.4f} (95% CI: [{V_S_lower:.4f}, {V_S_upper:.4f}])")
        print(f"  - w_S: {1/w_opt:.2f} ± {se_w/w_opt**2:.2f} (w={w_opt:.4f})")
        print(f"{'='*60}")
    
    return res


def estimate_variance_components(
    df_frreg: pd.DataFrame,
    slope_significance: str,
    loss_method: Literal['log', 'log_bc', 'linear_wls'] = 'log',
    n_resample: int = 1000,
    n_repeat_cv: int = 10,
    n_block: int = 10,
    use_parallel: bool = True,
    verbose: bool = True
    ) -> Dict:
    """
    FR-reg 결과로부터 분산 성분(V_G, V_S, w)을 추정합니다.
    
    Args:
        df_frreg: FR-reg 결과 (log_slope, log_se 컬럼 필수)
        slope_significance: slope test 결과 ("fast", "slow", "similar")
        method: 추정 방법
            - 'known_var': Jensen 보정 + Fixed-scale WLS (권장, 소수 데이터에서 안정적)
            - 'direct': Bootstrap 또는 Delta Method (n_bootstrap에 따라)
            - 'resample': Cross-Validation 기반 (기존 BIGFAM 방식)
        method: 추정 방법
            - 'known_var': Jensen 보정 + Fixed-scale WLS (권장, 소수 데이터에서 안정적)
            - 'direct': Bootstrap 또는 Delta Method (n_bootstrap에 따라)
            - 'resample': Cross-Validation 기반 (기존 BIGFAM 방식)
            - 'lognormal': Log-normal 분포 기반 Bootstrap
    loss_method: 'resample' 방식에서 사용할 loss function ('log', 'log_bc', 'linear_wls')
        n_bootstrap: Bootstrap 반복 횟수 (method='direct')
        n_resample: 각 DOR당 resample 수 (df_bootstrap 없을 때만 사용)
        n_repeat_cv: CV 반복 횟수 (method='resample')
        n_block: CV fold 수 (method='resample')
        df_bootstrap: FR-reg에서 저장된 bootstrap slope DataFrame (있으면 직접 사용)
        verbose: 상세 출력 여부
        
    Returns:
        Dictionary with V_G, V_S, w estimates and their 95% CIs
    """
    return _estimate_variance_resample(
        df_frreg, 
        slope_significance, 
        n_resample, 
        n_repeat_cv, 
        n_block, 
        loss_method,
        use_parallel,
        verbose
    )


# def _estimate_variance_direct(
#     df_frreg: pd.DataFrame,
#     slope_significance: str = "similar",
#     n_bootstrap: int = 100,
#     verbose: bool = True
#     ) -> Dict:
#     """
#     Bootstrap 기반 분산 성분 추정 (Direct 방식).
    
#     방법:
#     - w: Grid search (discrete, 0~1 사이의 감쇄 인자)
#     - V_G, V_S: scipy.optimize.minimize (continuous)
#     - CI: Bootstrap
#     """
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"[Variance Components] 분산 성분 추정 시작 (Direct 방식)")
#         print(f"{'='*60}")
    
#     df = df_frreg.dropna(subset=['log_slope']).copy()
    
#     if len(df) < 2:
#         if verbose:
#             print(f"  ⚠️ 데이터 부족 (DOR 그룹 {len(df)}개 < 최소 2개)")
#         return None
    
#     # Set w range based on slope test (w is decay factor 1/ws)
#     if slope_significance == "high":
#         # w_s > 2 => 1/w_s < 0.5
#         w_range = np.arange(0.01, 0.45 + 0.01, 0.01)
#     elif slope_significance == "low":
#         # w_s < 2 => 1/w_s > 0.5
#         w_range = np.arange(0.55, 0.95 + 0.01, 0.01)
#     else:  # "similar"
#         # w_s ~ 2 => 1/w_s ~ 0.5
#         w_range = np.arange(0.40, 0.60 + 0.01, 0.01)
    
#     if verbose:
#         print(f"\n  [Step 1] Grid Search 설정")
#         print(f"    - Slope Test 결과: {slope_significance}")
#         print(f"    - w 검색 범위: [{1/w_range.max():.2f}, {1/w_range.min():.2f}]")
    
#     # Point estimate
#     if verbose:
#         print(f"\n  [Step 2] Point Estimate (Grid Search + Optimization)")
    
#     best_loss = np.inf
#     best_params = None
    
#     for w in w_range:
#         V_G_opt, V_S_opt, loss = _optimize_variance_for_w(w, df)
#         if loss < best_loss:
#             best_loss = loss
#             best_params = (V_G_opt, V_S_opt, w)
    
#     if best_params is None:
#         if verbose:
#             print(f"    ⚠️ 최적화 실패")
#         return None
    
#     V_G_point, V_S_point, w_point = best_params
    
#     if verbose:
#         print(f"    - V_G: {V_G_point:.4f}")
#         print(f"    - V_S: {V_S_point:.4f}")
#         print(f"    - w_S: {1/w_point:.2f}")
#         print(f"    - Loss: {best_loss:.6f}")
    
#     # Bootstrap for CI
#     if verbose:
#         print(f"\n  [Step 3] Bootstrap ({n_bootstrap}회)")
    
#     np.random.seed(42)
#     boot_results = []
    
#     for i in range(n_bootstrap):
#         df_boot = df.copy()
#         if 'log_se' in df.columns:
#             df_boot['log_slope'] = np.random.normal(
#                 df['log_slope'].values,
#                 df['log_se'].fillna(0.1).values
#             )
        
#         best_loss_boot = np.inf
#         best_params_boot = None
        
#         for w in w_range:
#             V_G_opt, V_S_opt, loss = _optimize_variance_for_w(w, df_boot)
#             if loss < best_loss_boot:
#                 best_loss_boot = loss
#                 best_params_boot = (V_G_opt, V_S_opt, w)
        
#         if best_params_boot:
#             boot_results.append({
#                 'V_G': best_params_boot[0],
#                 'V_S': best_params_boot[1],
#                 'w': best_params_boot[2]
#             })
    
#     if verbose:
#         print(f"    - 성공: {len(boot_results)}/{n_bootstrap}회")
    
#     if boot_results:
#         df_boot_results = pd.DataFrame(boot_results)
#         result = {
#             'V_G': V_G_point,
#             'V_G_lower': df_boot_results['V_G'].quantile(0.025),
#             'V_G_upper': df_boot_results['V_G'].quantile(0.975),
#             'V_S': V_S_point,
#             'V_S_lower': df_boot_results['V_S'].quantile(0.025),
#             'V_S_upper': df_boot_results['V_S'].quantile(0.975),
#             'w': w_point,
#             'w_lower': df_boot_results['w'].quantile(0.025),
#             'w_upper': df_boot_results['w'].quantile(0.975),
#             'loss': best_loss
#         }
#     else:
#         result = {
#             'V_G': V_G_point, 'V_G_lower': np.nan, 'V_G_upper': np.nan,
#             'V_S': V_S_point, 'V_S_lower': np.nan, 'V_S_upper': np.nan,
#             'w': w_point, 'w_lower': np.nan, 'w_upper': np.nan,
#             'loss': best_loss
#         }
    
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"[Variance Components] 추정 완료 ✓")
#         print(f"  - V_G (유전분산): {result['V_G']:.4f} (95% CI: [{result['V_G_lower']:.4f}, {result['V_G_upper']:.4f}])")
#         print(f"  - V_S (환경분산): {result['V_S']:.4f} (95% CI: [{result['V_S_lower']:.4f}, {result['V_S_upper']:.4f}])")
#         print(f"  - w_S (환경감쇠율): {1/result['w']:.2f} (95% CI: [{1/result['w_upper']:.2f}, {1/result['w_lower']:.2f}])")
#         print(f"{'='*60}")
    
#     return result


def _estimate_variance_resample(
    df_frreg: pd.DataFrame,
    slope_significance: str = "similar",
    n_resample: int = 1000,
    n_repeat_cv: int = 10,
    n_block: int = 10,
    loss_method: Literal['log', 'log_bc', 'linear_wls'] = 'log_bc',
    use_parallel: bool = True,
    verbose: bool = True
    ) -> Dict:
    """
    Cross-Validation 기반 분산 성분 추정 (기존 BIGFAM 방식).
    
    Args:
        df_frreg: FR-reg 결과
        slope_significance: slope test 결과
        n_resample: 각 DOR당 resample 수 (df_bootstrap 없을 때만 사용)
        n_repeat_cv: CV 반복 횟수
        n_block: CV fold 수
        df_bootstrap: FR-reg에서 저장된 cluster bootstrap 결과 (있으면 직접 사용)
        use_parallel: ProcessPoolExecutor 병렬 실행 여부
        loss_method: Loss function type ('log', 'log_bc', 'linear_wls')
        verbose: 상세 출력 여부
        
    Returns:
        Dictionary with V_G, V_S, w estimates and their 95% CIs
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Variance Components] Starting estimation (CV mode)")
        print(f"{'='*60}")
    
    df = df_frreg.dropna(subset=['log_slope']).copy()
    
    if len(df) < 2:
        if verbose:
            print(f"  Warning: insufficient data (DOR groups: {len(df)} < minimum 2)")
        return None
    
    # Set w range based on slope test
    sig = (slope_significance or "similar").lower()
    if sig == "high":
        sig = "fast"
    elif sig == "low":
        sig = "slow"

    if sig == "fast":
        w_range = np.arange(0.01, 0.45 + 0.01, 0.01)
    elif sig == "slow":
        w_range = np.arange(0.55, 0.95 + 0.01, 0.01)
    else:
        w_range = np.arange(0.30, 0.70 + 0.01, 0.01)
    w_range_tuple = tuple(float(x) for x in w_range)
    
    # Use provided bootstrap samples or resample from summary statistics
    # use_cluster_bootstrap = (df_bootstrap is not None and len(df_bootstrap) > 0)
    
    if verbose:
        print(f"\n  [Step 1] Setup")
        print(f"    - Slope test result: {sig}")
        print(f"    - w_S search range: [{1/w_range.max():.2f}, {1/w_range.min():.2f}]")
        print(f"    - Cross-validation: {n_block}-fold × {n_repeat_cv} repeats = {n_block * n_repeat_cv} runs")
    
    # Get bootstrap slopes
    np.random.seed(42)
    use_bias_correction = (loss_method == "log_bc")
    df_lmbds = _resample_coefficients(
        df,
        n_resample=n_resample,
        use_bias_correction=use_bias_correction
    )
    

    
    if len(df_lmbds) == 0:
        if verbose:
            print(f"  Warning: no bootstrap samples available")
        return None
    
    # Calculate weights if using linear_wls
    if loss_method == 'linear_wls':
        # Calculate weights from df_frreg
        df_weights = df[['DOR', 'log_slope', 'log_se']].copy()
        # Linear slope
        df_weights['slope_linear'] = 2 ** df_weights['log_slope']
        # Linear SE approximation (Delta method)
        df_weights['se_linear'] = df_weights['slope_linear'] * np.log(2) * df_weights['log_se'].fillna(0.1)
        # Weight = 1 / Variance
        df_weights['weight'] = 1 / (df_weights['se_linear'] ** 2)
        
        # Merge weights into df_lmbds
        weight_map = df_weights.set_index('DOR')['weight'].to_dict()
        df_lmbds['weight'] = df_lmbds['DOR'].map(weight_map).fillna(1.0)
        
        if verbose:
            print(f"    - Loss function: weighted least squares (linear scale)")
            print(f"    - Weighting: 1/Var(slope) (delta method approximation)")
    else:
        if verbose:
            if loss_method == 'log_bc':
                print(f"    - Loss function: log-scale SSE (Jensen bias corrected)")
            else:
                print(f"    - Loss function: log-scale SSE (original)")

    df_lmbds = df_lmbds.astype({"DOR": int, "idx": int, "slope": float})
    
    # Filter out invalid slopes (slope <= 0 or very small)
    min_slope = 1e-5
    n_before = len(df_lmbds)
    df_lmbds = df_lmbds[df_lmbds["slope"] > min_slope].copy()
    n_after = len(df_lmbds)
    
    if n_after == 0:
        if verbose:
            print(f"  Warning: no valid bootstrap samples remain (slope > {min_slope})")
        return None
    
    if verbose and n_before != n_after:
        print(f"    - Filtering: {n_before} -> {n_after} samples (slope > {min_slope})")
    
    if verbose:
        print(f"\n  [Step 2] Running cross-validation")
    
    results_list = []
    n_workers = max(1, min((os.cpu_count() or 1), n_repeat_cv * n_block))

    # Pre-compute per-sample arrays once (block assignment만 repeat마다 바뀜)
    dor_samples = df_lmbds["DOR"].to_numpy(dtype=int, copy=False)
    slope_samples = df_lmbds["slope"].to_numpy(dtype=float, copy=False)
    unique_dors = df_lmbds["DOR"].unique().astype(int)  # keep appearance order
    dor_to_idx = {dor: np.flatnonzero(dor_samples == dor) for dor in unique_dors}

    if loss_method == "linear_wls":
        weight_samples = df_lmbds["weight"].to_numpy(dtype=float, copy=False)
        log_slope_samples = None
    else:
        weight_samples = None
        log_slope_samples = np.log2(slope_samples)

    dor_values = unique_dors.astype(float)
    n_dor = len(unique_dors)

    def run_one_repeat(i_rcv: int, executor: Optional[ProcessPoolExecutor] = None) -> None:
        # Assign blocks per DOR (shuffle within each DOR)
        block_labels = np.empty_like(dor_samples, dtype=int)
        for dor in unique_dors:
            idxs = dor_to_idx.get(int(dor))
            if idxs is None or len(idxs) == 0:
                continue
            n = len(idxs)
            labels = (np.arange(n, dtype=int) % int(n_block))
            np.random.shuffle(labels)
            block_labels[idxs] = labels

        # Build per-(DOR, block) summaries
        if loss_method == "linear_wls":
            sum_w = np.zeros((n_dor, n_block), dtype=float)
            sum_w_slope = np.zeros((n_dor, n_block), dtype=float)
            for j, dor in enumerate(unique_dors):
                idxs = dor_to_idx[int(dor)]
                blocks = block_labels[idxs]
                w = weight_samples[idxs]
                s = slope_samples[idxs]
                sum_w[j, :] = np.bincount(blocks, weights=w, minlength=n_block)
                sum_w_slope[j, :] = np.bincount(blocks, weights=w * s, minlength=n_block)
            total_w = sum_w.sum(axis=1)
            total_w_slope = sum_w_slope.sum(axis=1)
        else:
            sum_log = np.zeros((n_dor, n_block), dtype=float)
            cnt = np.zeros((n_dor, n_block), dtype=float)
            for j, dor in enumerate(unique_dors):
                idxs = dor_to_idx[int(dor)]
                blocks = block_labels[idxs]
                v = log_slope_samples[idxs]
                sum_log[j, :] = np.bincount(blocks, weights=v, minlength=n_block)
                cnt[j, :] = np.bincount(blocks, minlength=n_block).astype(float)
            total_sum_log = sum_log.sum(axis=1)
            total_cnt = cnt.sum(axis=1)

        futures = []
        for i_b in range(n_block):
            if loss_method == "linear_wls":
                test_w = sum_w[:, i_b]
                test_w_slope = sum_w_slope[:, i_b]
                train_w = total_w - test_w
                train_w_slope = total_w_slope - test_w_slope
                if float(train_w.sum()) <= 0.0 or float(test_w.sum()) <= 0.0:
                    continue
                train_mean = np.divide(train_w_slope, train_w, out=np.zeros_like(train_w_slope), where=train_w > 0)
                test_mean = np.divide(test_w_slope, test_w, out=np.zeros_like(test_w_slope), where=test_w > 0)
                train_weight = train_w
                test_weight = test_w
            else:
                test_n = cnt[:, i_b]
                test_sum = sum_log[:, i_b]
                train_n = total_cnt - test_n
                train_sum = total_sum_log - test_sum
                if float(train_n.sum()) <= 0.0 or float(test_n.sum()) <= 0.0:
                    continue
                train_mean = np.divide(train_sum, train_n, out=np.zeros_like(train_sum), where=train_n > 0)
                test_mean = np.divide(test_sum, test_n, out=np.zeros_like(test_sum), where=test_n > 0)
                train_weight = train_n
                test_weight = test_n

            if executor is not None:
                futures.append(
                    executor.submit(
                        _estimate_cv_fold_worker,
                        dor_values,
                        train_mean,
                        train_weight,
                        test_mean,
                        test_weight,
                        w_range_tuple,
                        loss_method,
                    )
                )
            else:
                try:
                    best_params = _estimate_cv_fold_worker(
                        dor_values,
                        train_mean,
                        train_weight,
                        test_mean,
                        test_weight,
                        w_range_tuple,
                        loss_method,
                    )
                except Exception:
                    continue
                if best_params:
                    results_list.append(best_params)

        if executor is not None:
            for future in as_completed(futures):
                try:
                    best_params = future.result()
                except Exception:
                    continue
                if best_params:
                    results_list.append(best_params)

    if use_parallel:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for i_rcv in range(n_repeat_cv):
                    run_one_repeat(i_rcv, executor=executor)
        except (PermissionError, OSError) as exc:
            if verbose:
                print(f"  Warning: parallel execution failed ({type(exc).__name__}: {exc}). Retrying serially.")
            for i_rcv in range(n_repeat_cv):
                run_one_repeat(i_rcv, executor=None)
    else:
        for i_rcv in range(n_repeat_cv):
            run_one_repeat(i_rcv, executor=None)
    
    if not results_list:
        if verbose:
            print(f"  Warning: no cross-validation results were produced")
        return None
    
    df_results = pd.DataFrame(results_list)
    
    # Calculate point estimates and CIs
    result = {
        'V_G': df_results['V_G'].median(),
        'V_G_lower': df_results['V_G'].quantile(0.025),
        'V_G_upper': df_results['V_G'].quantile(0.975),
        'V_S': df_results['V_S'].median(),
        'V_S_lower': df_results['V_S'].quantile(0.025),
        'V_S_upper': df_results['V_S'].quantile(0.975),
        'w': df_results['w'].median(),
        'w_lower': df_results['w'].quantile(0.025),
        'w_upper': df_results['w'].quantile(0.975),
        'n_estimates': len(df_results)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Variance Components] Done ✓ (CV mode)")
        print(f"  - V_G: {result['V_G']:.4f} (95% CI: [{result['V_G_lower']:.4f}, {result['V_G_upper']:.4f}])")
        print(f"  - V_S: {result['V_S']:.4f} (95% CI: [{result['V_S_lower']:.4f}, {result['V_S_upper']:.4f}])")
        print(f"  - w_S: {1/result['w']:.2f} (95% CI: [{1/result['w_upper']:.2f}, {1/result['w_lower']:.2f}])")
        print(f"  - Number of estimates: {result['n_estimates']}")
        print(f"{'='*60}")
    
    return result, df_results


# =============================================================================
# Pairwise Re-fitting with Fixed w (Stage 4)
# =============================================================================

def _to_asymmetric_pairs(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    입력 pair 데이터에서 (A,B)/(B,A) 중복을 제거한 asymmetric pool을 생성합니다.
    """
    required = ['volid', 'relid', 'DOR', 'vol_trait', 'rel_trait']
    missing = [c for c in required if c not in df_pairs.columns]
    if missing:
        raise ValueError(f"Missing required columns for pairwise estimation: {missing}")

    df = df_pairs.dropna(subset=['volid', 'relid', 'DOR', 'vol_trait', 'rel_trait']).copy()
    df['DOR'] = pd.to_numeric(df['DOR'], errors='coerce')
    df['vol_trait'] = pd.to_numeric(df['vol_trait'], errors='coerce')
    df['rel_trait'] = pd.to_numeric(df['rel_trait'], errors='coerce')
    df = df.dropna(subset=['DOR', 'vol_trait', 'rel_trait']).copy()

    return drop_symmetric_duplicates(df, vol_col='volid', rel_col='relid')


def _fit_pairwise_continuous_fixed_w(
    df_asym: pd.DataFrame,
    w: float,
    two_way_cluster: bool = True
    ) -> Dict:
    """
    Continuous trait용 pairwise 재추정.

    E[z_i z_j] = 2^{-d} * V_G + w^{d-1} * V_S
    """
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

    d = df_asym['DOR'].to_numpy(dtype=float)
    g = (0.5 ** d)
    s = (w ** (d - 1))

    y = (
        df_asym['vol_trait'].to_numpy(dtype=float) *
        df_asym['rel_trait'].to_numpy(dtype=float)
    )

    valid = np.isfinite(y) & np.isfinite(g) & np.isfinite(s)
    if valid.sum() < 10:
        return {
            'success': False,
            'V_G': np.nan,
            'V_S': np.nan,
            'V_G_se': np.nan,
            'V_S_se': np.nan,
            'n_obs': int(valid.sum())
        }

    dfv = df_asym.loc[valid].copy()
    X = np.column_stack([g[valid], s[valid]])
    yv = y[valid]

    try:
        model = sm.OLS(yv, X).fit()
        V_G = float(model.params[0])
        V_S = float(model.params[1])

        if two_way_cluster:
            cov = cov_cluster_2groups(
                model,
                dfv['volid'].to_numpy(),
                dfv['relid'].to_numpy()
            )[0]
        else:
            cov = cov_cluster(model, dfv['volid'].to_numpy())

        se_g = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) and cov[0, 0] >= 0 else np.nan
        se_s = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) and cov[1, 1] >= 0 else np.nan
        return {
            'success': True,
            'V_G': V_G,
            'V_S': V_S,
            'V_G_se': se_g,
            'V_S_se': se_s,
            'n_obs': int(valid.sum())
        }
    except Exception:
        return {
            'success': False,
            'V_G': np.nan,
            'V_S': np.nan,
            'V_G_se': np.nan,
            'V_S_se': np.nan,
            'n_obs': int(valid.sum())
        }


def _fit_univariate_probit_init(
    y: np.ndarray,
    X: np.ndarray
    ) -> np.ndarray:
    """
    Binary pairwise 재추정을 위한 univariate probit 초기값(beta) 계산.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    def neg_loglik(beta: np.ndarray) -> float:
        eta = np.clip(X @ beta, -30, 30)
        p = np.clip(norm.cdf(eta), 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    beta0 = np.zeros(X.shape[1], dtype=float)
    result = minimize(
        neg_loglik,
        beta0,
        method='L-BFGS-B',
        options={'maxiter': 300, 'disp': False}
    )
    return result.x if result.success else beta0


def _fit_pairwise_binary_fixed_w(
    df_asym: pd.DataFrame,
    w: float,
    covariate_cols: Optional[List[str]] = None,
    two_way_cluster: bool = True
    ) -> Dict:
    """
    Binary trait용 pairwise 재추정 (Bivariate Probit, fixed w + sandwich SE).

    Corr(L_i, L_j | d) = 2^{-d} * V_G + w^{d-1} * V_S
    """
    from scipy.optimize import minimize
    from scipy.stats import norm
    from ..frreg.helpers import normalize_covariate_prefixes, bvn_cdf_vectorized

    def _cluster_meat(scores: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
        m = np.zeros((scores.shape[1], scores.shape[1]))
        for cid in np.unique(cluster_ids):
            s_c = scores[cluster_ids == cid].sum(axis=0)
            m += np.outer(s_c, s_c)
        return m

    def _loglik_contrib(params: np.ndarray, data: Dict, return_invalid: bool = False):
        beta = params[:data['n_beta']]
        V_G = params[data['n_beta']]
        V_S = params[data['n_beta'] + 1]

        eta1 = data['X1'] @ beta
        eta2 = data['X2'] @ beta

        p00 = np.zeros(data['n_obs'], dtype=float)
        invalid = 0

        for d_val, mask in data['dor_masks'].items():
            rho_d = (0.5 ** d_val) * V_G + (w ** (d_val - 1)) * V_S
            if rho_d <= 1e-6 or rho_d >= 0.995:
                invalid += 1
            rho_d = float(np.clip(rho_d, 1e-6, 0.995))
            p00[mask] = bvn_cdf_vectorized(-eta1[mask], -eta2[mask], rho_d)

        eps = 1e-12
        phi1 = norm.cdf(-eta1)
        phi2 = norm.cdf(-eta2)

        p01 = np.clip(phi1 - p00, eps, 1 - eps)
        p10 = np.clip(phi2 - p00, eps, 1 - eps)
        p11 = np.clip(1 - phi1 - phi2 + p00, eps, 1 - eps)
        p00 = np.clip(p00, eps, 1 - eps)

        probs = (
            p00 * data['outcome00'] +
            p01 * data['outcome01'] +
            p10 * data['outcome10'] +
            p11 * data['outcome11']
        )
        probs = np.clip(probs, eps, 1 - eps)
        ll_i = np.log(probs)
        if return_invalid:
            return ll_i, invalid
        return ll_i

    def _objective(params: np.ndarray, data: Dict) -> float:
        ll_i, invalid = _loglik_contrib(params, data, return_invalid=True)
        nll = -float(np.sum(ll_i))
        if not np.isfinite(nll):
            return 1e12
        if invalid > 0:
            return nll + (invalid * 1e6)
        return nll

    def _sandwich_se(params: np.ndarray, data: Dict, two_way: bool = True) -> np.ndarray:
        n_params = len(params)
        eps_fd = 1e-5

        # Hessian of total log-likelihood
        H = np.zeros((n_params, n_params))
        for i in range(n_params):
            for j in range(i, n_params):
                h_i = eps_fd * max(1.0, abs(params[i]))
                h_j = eps_fd * max(1.0, abs(params[j]))

                params_pp = params.copy(); params_pp[i] += h_i; params_pp[j] += h_j
                params_pm = params.copy(); params_pm[i] += h_i; params_pm[j] -= h_j
                params_mp = params.copy(); params_mp[i] -= h_i; params_mp[j] += h_j
                params_mm = params.copy(); params_mm[i] -= h_i; params_mm[j] -= h_j

                l_pp = float(np.sum(_loglik_contrib(params_pp, data)))
                l_pm = float(np.sum(_loglik_contrib(params_pm, data)))
                l_mp = float(np.sum(_loglik_contrib(params_mp, data)))
                l_mm = float(np.sum(_loglik_contrib(params_mm, data)))

                H_ij = (l_pp - l_pm - l_mp + l_mm) / (4 * h_i * h_j)
                H[i, j] = H_ij
                H[j, i] = H_ij

        try:
            B_inv = np.linalg.inv(-H)
        except np.linalg.LinAlgError:
            return np.full(n_params, np.nan)

        # Per-observation score vectors
        scores = np.zeros((data['n_obs'], n_params))
        for k in range(n_params):
            h_k = eps_fd * max(1.0, abs(params[k]))
            params_plus = params.copy(); params_plus[k] += h_k
            params_minus = params.copy(); params_minus[k] -= h_k

            ll_plus = _loglik_contrib(params_plus, data)
            ll_minus = _loglik_contrib(params_minus, data)
            scores[:, k] = (ll_plus - ll_minus) / (2 * h_k)

        M1 = _cluster_meat(scores, data['vol_ids'])
        if two_way:
            M2 = _cluster_meat(scores, data['rel_ids'])
            inter_ids = np.array([f"{a}__{b}" for a, b in zip(data['vol_ids'], data['rel_ids'])], dtype=object)
            M12 = _cluster_meat(scores, inter_ids)
            M = M1 + M2 - M12
        else:
            M = M1

        try:
            V = B_inv @ M @ B_inv
            se = np.sqrt(np.maximum(np.diag(V), 0.0))
            return se
        except Exception:
            return np.full(n_params, np.nan)

    df = df_asym.copy()
    Y1 = df['vol_trait'].to_numpy(dtype=float)
    Y2 = df['rel_trait'].to_numpy(dtype=float)
    dor = df['DOR'].to_numpy(dtype=int)

    valid = (
        np.isfinite(Y1) & np.isfinite(Y2) &
        np.isfinite(dor)
    )
    if valid.sum() < 10:
        return {
            'success': False,
            'V_G': np.nan,
            'V_S': np.nan,
            'V_G_se': np.nan,
            'V_S_se': np.nan,
            'n_obs': int(valid.sum()),
            'n_params': np.nan
        }

    df = df.loc[valid].copy()
    Y1 = df['vol_trait'].to_numpy(dtype=float)
    Y2 = df['rel_trait'].to_numpy(dtype=float)
    dor = df['DOR'].to_numpy(dtype=int)

    cov_prefixes = normalize_covariate_prefixes(df, covariate_cols)

    X1_list = [np.ones(len(df), dtype=float)]
    X2_list = [np.ones(len(df), dtype=float)]
    for col in cov_prefixes:
        X1_list.append(df[f'vol_{col}'].to_numpy(dtype=float))
        X2_list.append(df[f'rel_{col}'].to_numpy(dtype=float))

    X1 = np.column_stack(X1_list)
    X2 = np.column_stack(X2_list)

    design_valid = (
        np.all(np.isfinite(X1), axis=1) &
        np.all(np.isfinite(X2), axis=1) &
        np.isfinite(Y1) &
        np.isfinite(Y2) &
        np.isfinite(dor)
    )
    if design_valid.sum() < 10:
        return {
            'success': False,
            'V_G': np.nan,
            'V_S': np.nan,
            'V_G_se': np.nan,
            'V_S_se': np.nan,
            'n_obs': int(design_valid.sum()),
            'n_params': np.nan
        }

    df = df.loc[design_valid].copy()
    Y1 = Y1[design_valid]
    Y2 = Y2[design_valid]
    dor = dor[design_valid]
    X1 = X1[design_valid]
    X2 = X2[design_valid]

    # initialize beta from univariate probit
    vol_data = df[['volid', 'vol_trait']].copy()
    vol_data.columns = ['pid', 'y']
    rel_data = df[['relid', 'rel_trait']].copy()
    rel_data.columns = ['pid', 'y']
    for col in cov_prefixes:
        vol_data[col] = df[f'vol_{col}'].values
        rel_data[col] = df[f'rel_{col}'].values
    df_indiv = pd.concat([vol_data, rel_data], ignore_index=True).drop_duplicates(subset=['pid'])

    Xi = [np.ones(len(df_indiv), dtype=float)]
    for col in cov_prefixes:
        Xi.append(df_indiv[col].to_numpy(dtype=float))
    X_indiv = np.column_stack(Xi)
    y_indiv = df_indiv['y'].to_numpy(dtype=float)
    beta0 = _fit_univariate_probit_init(y_indiv, X_indiv)

    # masks by DOR for faster BVN CDF
    unique_dor = np.unique(dor)
    dor_masks = {d_val: (dor == d_val) for d_val in unique_dor}

    outcome00 = (Y1 == 0) & (Y2 == 0)
    outcome01 = (Y1 == 0) & (Y2 == 1)
    outcome10 = (Y1 == 1) & (Y2 == 0)
    outcome11 = (Y1 == 1) & (Y2 == 1)

    n_beta = X1.shape[1]
    data = {
        'X1': X1,
        'X2': X2,
        'n_beta': n_beta,
        'n_obs': len(df),
        'dor_masks': dor_masks,
        'outcome00': outcome00,
        'outcome01': outcome01,
        'outcome10': outcome10,
        'outcome11': outcome11,
        'vol_ids': df['volid'].to_numpy(),
        'rel_ids': df['relid'].to_numpy()
    }

    bounds = [(None, None)] * n_beta + [(1e-8, 1.0), (1e-8, 1.0)]
    candidate_starts = [
        np.array([0.05, 0.05], dtype=float),
        np.array([0.10, 0.20], dtype=float),
        np.array([0.20, 0.10], dtype=float),
        np.array([0.30, 0.10], dtype=float),
        np.array([0.60, 0.20], dtype=float),
    ]

    best_result = None
    best_fun = np.inf

    for start in candidate_starts:
        params0 = np.concatenate([beta0, start])
        try:
            result = minimize(
                _objective,
                params0,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 600, 'disp': False}
            )
            if result.success and np.isfinite(result.fun) and result.fun < best_fun:
                best_fun = float(result.fun)
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {
            'success': False,
            'V_G': np.nan,
            'V_S': np.nan,
            'V_G_se': np.nan,
            'V_S_se': np.nan,
            'n_obs': int(len(df)),
            'n_params': int(n_beta + 2)
        }

    params_hat = best_result.x
    se = _sandwich_se(params_hat, data, two_way=two_way_cluster)

    return {
        'success': True,
        'V_G': float(params_hat[n_beta]),
        'V_S': float(params_hat[n_beta + 1]),
        'V_G_se': float(se[n_beta]) if len(se) > n_beta and np.isfinite(se[n_beta]) else np.nan,
        'V_S_se': float(se[n_beta + 1]) if len(se) > (n_beta + 1) and np.isfinite(se[n_beta + 1]) else np.nan,
        'n_obs': int(len(df)),
        'n_params': int(n_beta + 2),
        'opt_fun': float(best_fun)
    }


def _fit_variance_from_rho_gls(
    df_rho: pd.DataFrame,
    w: float
    ) -> Tuple[Dict, pd.DataFrame]:
    """
    DOR별 rho_hat(mean, SE) 요약 통계로부터 GLS로 V_G, V_S를 추정합니다.

    모델:
        rho_d = 2^{-d} * V_G + w^{d-1} * V_S + error_d
    """
    required = ['DOR', 'slope', 'se']
    missing = [c for c in required if c not in df_rho.columns]
    if missing:
        raise ValueError(f"Missing required columns for rho_gls mode: {missing}")

    df = df_rho.copy()
    df['DOR'] = pd.to_numeric(df['DOR'], errors='coerce')
    df['slope'] = pd.to_numeric(df['slope'], errors='coerce')
    df['se'] = pd.to_numeric(df['se'], errors='coerce')
    df = df.dropna(subset=['DOR', 'slope', 'se']).copy()
    df = df[df['se'] > 0].copy()

    if len(df) < 2:
        raise ValueError("rho_gls mode requires at least 2 valid DOR summaries.")

    # 동일 DOR가 여러 행이면 inverse-variance weighted mean으로 합침
    rows = []
    for dor, group in df.groupby('DOR', sort=True):
        w_iv = 1.0 / (group['se'].to_numpy(dtype=float) ** 2)
        rho_mean = float(np.sum(w_iv * group['slope'].to_numpy(dtype=float)) / np.sum(w_iv))
        rho_se = float(np.sqrt(1.0 / np.sum(w_iv)))
        rows.append({'DOR': float(dor), 'rho_hat': rho_mean, 'rho_se': rho_se, 'n_rows': int(len(group))})

    df_use = pd.DataFrame(rows).sort_values('DOR').reset_index(drop=True)

    d = df_use['DOR'].to_numpy(dtype=float)
    y = df_use['rho_hat'].to_numpy(dtype=float)
    se = df_use['rho_se'].to_numpy(dtype=float)
    w_iv = 1.0 / (se ** 2)

    X = np.column_stack([0.5 ** d, w ** (d - 1)])
    XtWX = X.T @ (w_iv[:, None] * X)
    XtWy = X.T @ (w_iv * y)

    try:
        beta = np.linalg.solve(XtWX, XtWy)
        cov_beta = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"rho_gls linear system is singular (weak identifiability): {e}") from e

    V_G = float(beta[0])
    V_S = float(beta[1])
    V_G_se = float(np.sqrt(cov_beta[0, 0])) if cov_beta[0, 0] >= 0 else np.nan
    V_S_se = float(np.sqrt(cov_beta[1, 1])) if cov_beta[1, 1] >= 0 else np.nan

    z = 1.96
    V_G_lower = V_G - z * V_G_se if np.isfinite(V_G_se) else np.nan
    V_G_upper = V_G + z * V_G_se if np.isfinite(V_G_se) else np.nan
    V_S_lower = V_S - z * V_S_se if np.isfinite(V_S_se) else np.nan
    V_S_upper = V_S + z * V_S_se if np.isfinite(V_S_se) else np.nan

    rho_pred = X @ beta
    resid = y - rho_pred
    dof = max(len(y) - 2, 1)
    sse = float(np.sum((resid / se) ** 2))
    sigma2 = sse / dof

    corr = np.nan
    denom = np.sqrt(max(cov_beta[0, 0], 0) * max(cov_beta[1, 1], 0))
    if denom > 0:
        corr = float(cov_beta[0, 1] / denom)

    cond_xtwx = float(np.linalg.cond(XtWX))

    result = {
        'V_G': V_G,
        'V_G_se': V_G_se,
        'V_G_lower': V_G_lower,
        'V_G_upper': V_G_upper,
        'V_S': V_S,
        'V_S_se': V_S_se,
        'V_S_lower': V_S_lower,
        'V_S_upper': V_S_upper,
        'corr_VG_VS': corr,
        'cond_XtWX': cond_xtwx,
        'weighted_sse': sse,
        'sigma2_hat': sigma2,
        'n_dor_used': int(len(df_use)),
        'method': 'rho_gls'
    }

    df_diag = df_use.copy()
    df_diag['rho_pred'] = rho_pred
    df_diag['residual'] = resid
    return result, df_diag


def estimate_pairwise_variance_components(
    df_pairs: pd.DataFrame,
    trait_type: Literal['continuous', 'binary'],
    w: float,
    stage4_mode: Literal['pairwise_sandwich', 'rho_gls'] = 'pairwise_sandwich',
    df_rho: Optional[pd.DataFrame] = None,
    covariate_cols: Optional[List[str]] = None,
    two_way_cluster: bool = True,
    verbose: bool = True
    ) -> Tuple[Dict, pd.DataFrame]:
    """
    Stage 4: 고정된 median w(=1/w_S)를 사용해 V_G, V_S를 재추정.

    모드:
    - pairwise_sandwich: all-pair 직접 적합 + sandwich SE
    - rho_gls: DOR별 rho_hat(mean,SE) 요약치로 GLS 추정
    """
    if not (0 < w < 1):
        raise ValueError(f"w must be in the range (0, 1). Received: {w}")

    if stage4_mode not in ['pairwise_sandwich', 'rho_gls']:
        raise ValueError(f"Unknown stage4_mode: {stage4_mode}")

    if stage4_mode == 'rho_gls':
        if df_rho is None:
            raise ValueError("rho_gls mode requires df_rho with columns (DOR, slope, se).")

        result, df_diag = _fit_variance_from_rho_gls(df_rho, w=w)
        result['w'] = float(w)
        result['w_S'] = float(1 / w)
        result['stage4_mode'] = 'rho_gls'

        if verbose:
            print(f"\n{'='*60}")
            print("[Pairwise Variance Refit] Starting Stage 4")
            print(f"{'='*60}")
            print(f"  - mode: rho_gls")
            print(f"  - Fixed w: {w:.4f} (w_S={1/w:.3f})")
            print(f"  - DOR used: {result['n_dor_used']}")
            print(f"\n{'='*60}")
            print("[Pairwise Variance Refit] Done ✓")
            print(f"  - V_G: {result['V_G']:.4f} ± {result['V_G_se']:.4f} (95% CI: [{result['V_G_lower']:.4f}, {result['V_G_upper']:.4f}])")
            print(f"  - V_S: {result['V_S']:.4f} ± {result['V_S_se']:.4f} (95% CI: [{result['V_S_lower']:.4f}, {result['V_S_upper']:.4f}])")
            print(f"  - corr(V_G, V_S): {result['corr_VG_VS']:.4f}")
            print(f"  - cond(X'WX): {result['cond_XtWX']:.2f}")
            print(f"{'='*60}")

        return result, df_diag

    df_asym = _to_asymmetric_pairs(df_pairs)
    if len(df_asym) < 20:
        raise ValueError(f"Too few valid asymmetric pairs: {len(df_asym)}")

    if verbose:
        print(f"\n{'='*60}")
        print("[Pairwise Variance Refit] Starting Stage 4")
        print(f"{'='*60}")
        print(f"  - Trait type: {trait_type}")
        print(f"  - Fixed w: {w:.4f} (w_S={1/w:.3f})")
        print(f"  - Asymmetric pairs: {len(df_asym):,}")
        print(f"  - Clustering: {'two-way(vol,rel)' if two_way_cluster else 'one-way(vol)'}")

    # Continuous trait: optional covariate adjustment + global standardization
    if trait_type == 'continuous':
        if covariate_cols and len(covariate_cols) > 0:
            from ..frreg.continuous import _preprocess_covariates
            df_asym = _preprocess_covariates(df_asym, covariate_cols, verbose=False)
            df_asym = _to_asymmetric_pairs(df_asym)

        for col in ['vol_trait', 'rel_trait']:
            col_vals = df_asym[col].to_numpy(dtype=float)
            col_std = np.std(col_vals)
            if not np.isfinite(col_std) or col_std <= 0:
                raise ValueError(f"Failed to standardize {col}: standard deviation is zero or invalid.")
            df_asym[col] = (col_vals - np.mean(col_vals)) / col_std

        fit_result = _fit_pairwise_continuous_fixed_w(
            df_asym,
            w=w,
            two_way_cluster=two_way_cluster
        )

    elif trait_type == 'binary':
        fit_result = _fit_pairwise_binary_fixed_w(
            df_asym,
            w=w,
            covariate_cols=covariate_cols,
            two_way_cluster=two_way_cluster
        )
    else:
        raise ValueError(f"Unknown trait_type: {trait_type}")

    if not fit_result.get('success', False):
        raise RuntimeError("Point-estimate optimization failed.")

    V_G_point = float(fit_result['V_G'])
    V_S_point = float(fit_result['V_S'])
    V_G_se = float(fit_result.get('V_G_se', np.nan))
    V_S_se = float(fit_result.get('V_S_se', np.nan))

    z = 1.96
    V_G_lower = V_G_point - z * V_G_se if np.isfinite(V_G_se) else np.nan
    V_G_upper = V_G_point + z * V_G_se if np.isfinite(V_G_se) else np.nan
    V_S_lower = V_S_point - z * V_S_se if np.isfinite(V_S_se) else np.nan
    V_S_upper = V_S_point + z * V_S_se if np.isfinite(V_S_se) else np.nan

    df_diag = pd.DataFrame([fit_result])

    result = {
        'V_G': V_G_point,
        'V_G_se': V_G_se,
        'V_G_lower': V_G_lower,
        'V_G_upper': V_G_upper,
        'V_S': V_S_point,
        'V_S_se': V_S_se,
        'V_S_lower': V_S_lower,
        'V_S_upper': V_S_upper,
        'w': float(w),
        'w_S': float(1 / w),
        'n_pairs_asym': int(len(df_asym)),
        'cluster': 'two-way' if two_way_cluster else 'one-way',
        'method': f'pairwise_fixed_w_{trait_type}_sandwich',
        'stage4_mode': 'pairwise_sandwich'
    }

    if verbose:
        print(f"\n{'='*60}")
        print("[Pairwise Variance Refit] Done ✓")
        print(f"  - V_G: {result['V_G']:.4f} ± {result['V_G_se']:.4f} (95% CI: [{result['V_G_lower']:.4f}, {result['V_G_upper']:.4f}])")
        print(f"  - V_S: {result['V_S']:.4f} ± {result['V_S_se']:.4f} (95% CI: [{result['V_S_lower']:.4f}, {result['V_S_upper']:.4f}])")
        print(f"  - Fixed w_S: {result['w_S']:.3f}")
        print(f"  - Clustering: {result['cluster']}")
        print(f"{'='*60}")

    return result, df_diag
