"""
이진형 FR-regression 모듈 (Bivariate Probit)

이진형 표현형에 대한 FR-regression을 수행합니다.
Liability Threshold Model 기반의 Bivariate Probit 모델을 사용합니다.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Optional, List, Tuple
from tqdm import tqdm
from .. import processing
from ..utils.group_meta import relationship_group_metadata
from ..utils.pairs import drop_symmetric_duplicates
from ..utils.stats import slope_to_log_stats

# Import helper functions from helpers module
from .helpers import (
    bvn_cdf_vectorized as _bvn_cdf_vectorized,
    tetrachoric_correlation as _tetrachoric_correlation,
    compute_contingency_table as _compute_contingency_table,
    bivariate_probit_negloglik as _bivariate_probit_negloglik,
    normalize_covariate_prefixes as _normalize_covariate_prefixes,
)

_ROBUST_SE_NEAR_ZERO = 1e-12
_ROBUST_RHO_BOUNDARY_EPS = 1e-6
_ROBUST_RHO_UPPER = 0.99


def fit_binary_frreg(
    df_pairs: pd.DataFrame,
    group_by: str = 'DOR',
    prevalence: Optional[float] = None,
    covariate_cols: Optional[List[str]] = None,
    n_bootstrap: int = 100,
    min_pairs: int = 50,
    verbose: bool = True,
    return_bootstrap: bool = False
    ) -> pd.DataFrame:
    """    
    이진형 표현형에 대한 FR-regression 수행
    
    Args:
        verbose: True이면 상세 로그, False이면 간략한 진행 상황만 출력
        return_bootstrap: True이면 (df_results, df_bootstrap) 튜플 반환
    """
    n_groups = df_pairs[group_by].nunique()
    
    df = df_pairs.copy()
    vol_col = 'vol_trait'
    rel_col = 'rel_trait'
    
    if vol_col not in df.columns:
        raise ValueError(f"Missing required column: '{vol_col}'")
    
    # Prevalence 계산
    all_traits = pd.concat([df[vol_col], df[rel_col]])
    computed_prevalence = (all_traits == 1).mean() * 100
    
    if prevalence is None:
        prevalence = computed_prevalence
    
    # 공변량 설정 (age / vol_age / rel_age 모두 지원)
    cov_prefixes = _normalize_covariate_prefixes(df, covariate_cols)
    
    results = []
    bootstrap_results = []  # Store all bootstrap rhos
    
    for group_val in sorted(df[group_by].unique()):
        df_group = df[df[group_by] == group_val].copy()
        # 1. Asymmetric Pool 생성 (중복 pair 제거)
        df_asym = drop_symmetric_duplicates(df_group, vol_col='volid', rel_col='relid')
        n_asym = len(df_asym)
        n_vols = df_asym['volid'].nunique()
        n_asym_sym = n_asym * 2
        
        if n_vols < min_pairs:
            continue
        
        # 2. 전체 데이터에 대한 초기 추정 (Bootstrap 전)
        n00, n01, n10, n11 = _compute_contingency_table(df_group, vol_col, rel_col)
        rho_init, _ = _tetrachoric_correlation(n00, n01, n10, n11)
        if np.isnan(rho_init): rho_init = 0.3
        
        # 3. Bootstrap
        boot_rhos = []
        n_bootstrap_requested = int(n_bootstrap)
        n_bootstrap_no_case = 0
        
        # volid 별 인덱스 리스트 생성
        vol_indices = df_asym.groupby('volid').groups
        vol_ids = list(vol_indices.keys())
        
        iterator = range(n_bootstrap)
        if verbose and n_bootstrap > 10:
            iterator = tqdm(range(n_bootstrap), desc="  | Bootstrap", leave=False)
            
        for _ in iterator:
            # (1) Volunteer ID 복원 추출 (Cluster Resampling)
            resampled_vols = np.random.choice(vol_ids, size=len(vol_ids), replace=True)
            
            # (2) 각 Volunteer Instance에 대해 친척 1명 랜덤 선택
            boot_indices = [np.random.choice(vol_indices[v]) for v in resampled_vols]
            
            df_boot_asym = df_asym.loc[boot_indices]

            # Skip bootstrap replicates without any observed case pair.
            has_case = ((df_boot_asym[vol_col] == 1) | (df_boot_asym[rel_col] == 1)).any()
            if not has_case:
                n_bootstrap_no_case += 1
                continue
            
            # (3) 대칭화 확보 (Flip & Concat)
            df_boot = processing.symmetrize_pairs(df_boot_asym, verbose=False)
            
            # 데이터 준비
            Y1 = df_boot[vol_col].values.astype(float)
            Y2 = df_boot[rel_col].values.astype(float)
            
            X1_list = [np.ones(len(df_boot))]
            X2_list = [np.ones(len(df_boot))]
            
            for col in cov_prefixes:
                X1_list.append(df_boot[f'vol_{col}'].values)
                X2_list.append(df_boot[f'rel_{col}'].values)
            
            X1 = np.column_stack(X1_list)
            X2 = np.column_stack(X2_list)
            
            n_beta = X1.shape[1]
            params_init = np.concatenate([np.zeros(n_beta), [rho_init]])
            bounds = [(None, None)] * n_beta + [(1e-6, 0.99)]
            
            try:
                result = minimize(
                    fun=_bivariate_probit_negloglik,
                    x0=params_init,
                    args=(Y1, Y2, X1, X2),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'disp': False}
                )
                if result.success:
                    boot_rhos.append(result.x[-1])
                    if return_bootstrap:
                        bootstrap_results.append({group_by: group_val, 'idx': len(boot_rhos)-1, 'slope': result.x[-1]})
            except Exception:
                continue

        n_bootstrap_success = len(boot_rhos)
        n_bootstrap_fail = max(n_bootstrap_requested - n_bootstrap_success, 0)
        bootstrap_success_rate = float(n_bootstrap_success / n_bootstrap_requested) if n_bootstrap_requested > 0 else 0.0
        
        if not boot_rhos:
            continue

        rho_mean = float(np.mean(boot_rhos))
        rho_se = float(np.std(boot_rhos))

        # 결과 생성 (기본 컬럼만 유지)
        result_row = {
            group_by: group_val,
            'slope': rho_mean,
            'se': rho_se,
            'n_asym': n_asym,
            'n_asym_sym': n_asym_sym,
            'n_vols': n_vols,
            'n_bootstrap_requested': n_bootstrap_requested,
            'n_bootstrap_success': n_bootstrap_success,
            'n_bootstrap_fail': n_bootstrap_fail,
            'bootstrap_success_rate': bootstrap_success_rate,
            'n_bootstrap_no_case': n_bootstrap_no_case,
        }
        
        # group_by='relationship'일 때 추가 정보 저장
        if group_by == 'relationship':
            result_row.update(relationship_group_metadata(df_group))
        
        results.append(result_row)
        
        if verbose:
            print(f"  | {group_by}={group_val}: slope={rho_mean:.4f} ± {rho_se:.4f}")
            
    df_results = pd.DataFrame(results)
    
    if return_bootstrap:
        df_bootstrap = pd.DataFrame(bootstrap_results)
        return df_results, df_bootstrap
    
    return df_results


# =============================================================================
# Method 1: Posterior Mean Liability + Vol-Summary Bootstrap
# =============================================================================

def _compute_posterior_mean_liability(
    y: np.ndarray,
    eta: np.ndarray,
    use_residual: bool = True
    ) -> np.ndarray:
    """
    Compute posterior mean liability given binary outcome and linear predictor.
    
    For probit model: L = η + ε, P(Y=1|x) = Φ(η), where η = x'β
    
    Two options:
    1. Full posterior mean (use_residual=False):
       - If Y=1: E[L|Y=1,x] = η + φ(η)/Φ(η)
       - If Y=0: E[L|Y=0,x] = η - φ(η)/(1-Φ(η))
       
    2. Residual posterior mean (use_residual=True, RECOMMENDED):
       - If Y=1: E[ε|Y=1,x] = φ(η)/Φ(η)
       - If Y=0: E[ε|Y=0,x] = -φ(η)/(1-Φ(η))
       
       This removes the fixed effect η, leaving only the residual which
       represents the "excess liability" relative to covariate expectation.
       This is crucial for correctly estimating family correlation.
    
    Args:
        y: Binary outcomes (0/1)
        eta: Linear predictor (x'β)
        use_residual: If True, return E[ε|y,x] (residual only).
                     If False, return E[L|y,x] (full liability).
        
    Returns:
        Posterior mean liability (or residual) scores
    """
    phi = norm.pdf(eta)  # φ(η)
    Phi = norm.cdf(eta)  # Φ(η)
    
    # Avoid division by zero
    Phi = np.clip(Phi, 1e-10, 1 - 1e-10)
    
    # Residual posterior mean (without η)
    # E[ε|Y=1,x] = φ(η)/Φ(η)  (inverse Mills ratio)
    # E[ε|Y=0,x] = -φ(η)/(1-Φ(η))
    residual = np.where(
        y == 1,
        phi / Phi,           # Case: Y=1
        -phi / (1 - Phi)     # Control: Y=0
    )
    
    if use_residual:
        return residual
    else:
        # Full posterior: L = η + ε
        return eta + residual


def _fit_univariate_probit(
    y: np.ndarray,
    X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit univariate probit model using MLE.
    
    Args:
        y: Binary outcomes (0/1)
        X: Design matrix (n x p), should include intercept column
        
    Returns:
        beta: Estimated coefficients
        eta: Linear predictor (X @ beta)
    """
    from scipy.optimize import minimize
    
    def neg_loglik(beta):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)  # Numerical stability
        p = norm.cdf(eta)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return -ll
    
    # Initial values
    beta_init = np.zeros(X.shape[1])
    
    result = minimize(
        neg_loglik,
        beta_init,
        method='L-BFGS-B',
        options={'maxiter': 200, 'disp': False}
    )
    
    beta = result.x
    eta = X @ beta
    
    return beta, eta


def fit_binary_frreg_liability(
    df_pairs: pd.DataFrame,
    group_by: str = 'DOR',
    covariate_cols: Optional[List[str]] = None,
    n_bootstrap: int = 100,
    aggregation: str = 'mean',
    min_pairs: int = 50,
    verbose: bool = True,
    return_bootstrap: bool = False
    ) -> pd.DataFrame:
    """
    이진형 표현형에 대한 FR-regression (Posterior Mean Liability 방식)
    
    Binary trait을 univariate probit을 통해 posterior mean liability로 변환한 후,
    연속형처럼 A_v, B_v 요약 + vol-level bootstrap을 수행합니다.
    
    수학적 배경:
    - Probit: P(Y=1|x) = Φ(η), η = x'β
    - Posterior mean liability:
      - Y=1: E[L|Y=1,x] = η + φ(η)/Φ(η)  
      - Y=0: E[L|Y=0,x] = η - φ(η)/(1-Φ(η))
    - 이후 연속형과 동일한 slope 계산
    
    Args:
        df_pairs: 쌍 데이터프레임 (vol_trait, rel_trait 포함)
        group_by: 그룹화 컬럼 (기본: 'DOR')
        covariate_cols: 공변량 컬럼 리스트 (예: ['age', 'sex', 'PC1'])
        n_bootstrap: 부트스트랩 반복 횟수
        aggregation: vol별 요약 방법 ('mean' or 'sum')
        min_pairs: 최소 volunteer 수
        verbose: 상세 로그 출력 여부
        return_bootstrap: True면 (df_results, df_bootstrap) 튜플 반환
        
    Returns:
        pd.DataFrame: 그룹별 slope, se, log_slope, log_se 등
    """
    n_groups = df_pairs[group_by].nunique()
    
    print(f"\n[FR-reg Binary (Liability)] Starting analysis for {n_groups} groups...")
    
    if verbose:
        print(f"{'='*60}")
        print(f"  - Total pairs: {len(df_pairs):,}")
        print(f"  - Grouping: {group_by}")
        print(f"  - Bootstrap: {n_bootstrap} iterations (volunteer resampling only)")
        print(f"  - Aggregation: {aggregation}")
    
    df = df_pairs.copy()
    vol_col = 'vol_trait'
    rel_col = 'rel_trait'
    
    if vol_col not in df.columns:
        raise ValueError(f"Missing required column: '{vol_col}'")
    
    # 공변량 설정 (age / vol_age / rel_age 모두 지원)
    cov_prefixes = _normalize_covariate_prefixes(df, covariate_cols)
    if verbose and cov_prefixes:
        print(f"  - Covariates: {cov_prefixes}")
    
    results = []
    bootstrap_results = []
    
    for group_val in sorted(df[group_by].unique()):
        df_group = df[df[group_by] == group_val].copy()
        df_asym = drop_symmetric_duplicates(df_group, vol_col='volid', rel_col='relid')
        n_asym = len(df_asym)
        n_asym_sym = n_asym
        n_vols = df_asym['volid'].nunique()
        
        if n_vols < min_pairs:
            if verbose:
                    print(f"\n  [{group_by}={group_val}] Skipping due to small sample size (volunteers: {n_vols} < {min_pairs})")
            continue
        
        if verbose:
            print(f"\n  [{group_by}={group_val}] Analyzing {len(df_group):,} pairs (volunteers: {n_vols:,})...")
        
        if verbose:
            print(f"    [Step 0] Asymmetric pool: {len(df_group):,} -> {n_asym:,}")
        
        # Step 1: 개인 단위 테이블 생성
        # vol과 rel을 합쳐서 unique person 테이블 생성
        vol_data = df_asym[['volid', vol_col]].copy()
        vol_data.columns = ['pid', 'y']
        for col in cov_prefixes:
            vol_data[col] = df_asym[f'vol_{col}'].values
            
        rel_data = df_asym[['relid', rel_col]].copy()
        rel_data.columns = ['pid', 'y']
        for col in cov_prefixes:
            rel_data[col] = df_asym[f'rel_{col}'].values
        
        df_indiv = pd.concat([vol_data, rel_data], ignore_index=True)
        df_indiv = df_indiv.drop_duplicates(subset=['pid'])
        
        if verbose:
            print(f"    [Step 1] Individual-level table: {len(df_indiv):,} people")
            print(f"      - Cases: {(df_indiv['y'] == 1).sum():,}, Controls: {(df_indiv['y'] == 0).sum():,}")
        
        # Step 2: Univariate Probit으로 공변량 보정
        y_indiv = df_indiv['y'].values.astype(float)
        
        # Design matrix (intercept + covariates)
        X_list = [np.ones(len(df_indiv))]
        for col in cov_prefixes:
            X_list.append(df_indiv[col].values)
        X_indiv = np.column_stack(X_list)
        
        beta, eta_indiv = _fit_univariate_probit(y_indiv, X_indiv)
        
        if verbose:
            print(f"    [Step 2] Fitting univariate probit")
            if len(cov_prefixes) > 0:
                print(f"      - Beta coefficients: {beta}")
        
        # Step 3: Posterior Mean Liability 계산 (Residual: η 제거)
        L_indiv = _compute_posterior_mean_liability(y_indiv, eta_indiv, use_residual=True)
        
        # 센터링 (평균 효과 제거)
        L_mean = L_indiv.mean()
        L_std = L_indiv.std()
        L_indiv_centered = (L_indiv - L_mean) / L_std if L_std > 0 else L_indiv - L_mean
        
        # pid -> liability 매핑
        L_map = dict(zip(df_indiv['pid'], L_indiv_centered))
        
        if verbose:
            print(f"    [Step 3] Computing residual posterior mean liability")
            print(f"      - Raw L: mean={L_mean:.3f}, std={L_std:.3f}")
            print(f"      - Centered L: mean={L_indiv_centered.mean():.3f}, std={L_indiv_centered.std():.3f}")
        
        # Step 4: pair 테이블에 liability 붙이기
        df_asym['vol_L'] = df_asym['volid'].map(L_map)
        df_asym['rel_L'] = df_asym['relid'].map(L_map)
        
        # 진단: sign concordance (둘 다 같은 부호인 비율)
        sign_concordance = (np.sign(df_asym['vol_L']) == np.sign(df_asym['rel_L'])).mean()
        
        if verbose:
            print(f"    [Step 4] Sign concordance rate: {sign_concordance:.3f}")
        
        # Step 5: A_v, B_v 계산 (대칭화 접기)
        df_asym['A'] = 2.0 * df_asym['vol_L'] * df_asym['rel_L']
        df_asym['B'] = df_asym['vol_L']**2 + df_asym['rel_L']**2
        
        # vol별 요약
        agg_func = 'mean' if aggregation == 'mean' else 'sum'
        df_vol_summary = df_asym.groupby('volid').agg(
            A_v=('A', agg_func),
            B_v=('B', agg_func),
            n_rels=('relid', 'count')
        ).reset_index()
        
        A_values = df_vol_summary['A_v'].values
        B_values = df_vol_summary['B_v'].values
        n_vol = len(df_vol_summary)
        
        if verbose:
            print(f"    [Step 5] Volunteer-level summary: {n_vol:,} people")
            print(f"      - Relatives per volunteer: mean={df_vol_summary['n_rels'].mean():.1f}, "
                  f"median={df_vol_summary['n_rels'].median():.0f}, "
                  f"max={df_vol_summary['n_rels'].max()}")

        # Step 6: Bootstrap (vol만 복원추출)
        slopes = []
        
        iterator = range(n_bootstrap)
        if verbose and n_bootstrap > 10:
            iterator = tqdm(range(n_bootstrap), desc=f"    Bootstrap", leave=False)
        
        for _ in iterator:
            resampled_idxs = np.random.choice(n_vol, size=n_vol, replace=True)
            
            A_sum = A_values[resampled_idxs].sum()
            B_sum = B_values[resampled_idxs].sum()
            
            if B_sum == 0:
                continue
            
            slope = A_sum / B_sum
            slope = max(slope, 1e-6)
            slopes.append(slope)
            
            if return_bootstrap:
                bootstrap_results.append({
                    group_by: group_val,
                    'idx': len(slopes)-1,
                    'slope': slope
                })
        
        if not slopes:
            continue
        
        slope_mean = float(np.mean(slopes))
        slope_se = float(np.std(slopes))
        stats = slope_to_log_stats(slope_mean, slope_se)
        
        result_row = {
            group_by: group_val,
            **stats,
            'n_asym': n_asym,
            'n_asym_sym': n_asym_sym,
            'n_vols': n_vols,
            'n_indiv': len(df_indiv),
            'method': 'liability'
        }
        
        if group_by == 'relationship':
            result_row.update(relationship_group_metadata(df_group))
        
        results.append(result_row)
        
        if verbose:
            print(f"    → slope = {slope_mean:.4f} ± {slope_se:.4f}")
            print(f"    → log₂(slope) = {stats['log_slope']:.4f}")
    
    df_results = pd.DataFrame(results)
    
    print(f"[FR-reg Binary (Liability)] Done ✓ ({len(df_results)} groups analyzed)")
    
    if return_bootstrap:
        df_bootstrap = pd.DataFrame(bootstrap_results)
        return df_results, df_bootstrap
    
    return df_results


# =============================================================================
# Method 2: Bivariate Probit + Cluster-Robust SE (Sandwich)
# =============================================================================

def _bivariate_probit_score(
    params: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    eps: float = 1e-8
    ) -> np.ndarray:
    """
    Compute score vector (gradient of log-likelihood) for each observation.
    
    Returns:
        scores: (n_obs, n_params) matrix of individual score vectors
    """
    n_beta = X1.shape[1]
    beta = params[:n_beta]
    rho = np.clip(params[-1], -0.999, 0.999)
    
    eta1 = X1 @ beta
    eta2 = X2 @ beta
    n = len(Y1)
    
    # Compute probabilities
    phi1 = norm.cdf(-eta1)
    phi2 = norm.cdf(-eta2)
    p00 = _bvn_cdf_vectorized(-eta1, -eta2, rho)
    
    p01 = np.clip(phi1 - p00, eps, 1 - eps)
    p10 = np.clip(phi2 - p00, eps, 1 - eps)
    p11 = np.clip(1 - phi1 - phi2 + p00, eps, 1 - eps)
    p00 = np.clip(p00, eps, 1 - eps)
    
    # Select probability based on outcome
    probs = (
        p00 * ((Y1 == 0) & (Y2 == 0)) +
        p01 * ((Y1 == 0) & (Y2 == 1)) +
        p10 * ((Y1 == 1) & (Y2 == 0)) +
        p11 * ((Y1 == 1) & (Y2 == 1))
    )
    probs = np.clip(probs, eps, 1 - eps)
    
    # Numerical gradient for each parameter
    scores = np.zeros((n, len(params)))
    
    for k in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        h = eps * max(1, abs(params[k]))
        params_plus[k] += h
        params_minus[k] -= h
        
        ll_plus = -_bivariate_probit_negloglik(params_plus, Y1, Y2, X1, X2) / n
        ll_minus = -_bivariate_probit_negloglik(params_minus, Y1, Y2, X1, X2) / n
        
        # Approximate individual contributions (simplified)
        scores[:, k] = (ll_plus - ll_minus) / (2 * h)
    
    return scores


def _compute_sandwich_se(
    params: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_ids_2: Optional[np.ndarray] = None,
    eps: float = 1e-6
    ) -> np.ndarray:
    """
    Compute cluster-robust (sandwich) standard errors.
    
    Supports one-way or two-way clustering.
    
    Args:
        params: Estimated parameters
        Y1, Y2: Outcomes
        X1, X2: Design matrices
        cluster_ids: Primary cluster IDs (e.g., volid)
        cluster_ids_2: Optional secondary cluster IDs for two-way clustering (e.g., relid)
        
    Returns:
        Robust standard errors for each parameter
    """
    n = len(Y1)
    n_params = len(params)
    
    # Compute Hessian (Bread) using numerical differentiation
    H = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(i, n_params):
            params_pp = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            params_mm = params.copy()
            
            h_i = eps * max(1, abs(params[i]))
            h_j = eps * max(1, abs(params[j]))
            
            params_pp[i] += h_i; params_pp[j] += h_j
            params_pm[i] += h_i; params_pm[j] -= h_j
            params_mp[i] -= h_i; params_mp[j] += h_j
            params_mm[i] -= h_i; params_mm[j] -= h_j
            
            ll_pp = -_bivariate_probit_negloglik(params_pp, Y1, Y2, X1, X2)
            ll_pm = -_bivariate_probit_negloglik(params_pm, Y1, Y2, X1, X2)
            ll_mp = -_bivariate_probit_negloglik(params_mp, Y1, Y2, X1, X2)
            ll_mm = -_bivariate_probit_negloglik(params_mm, Y1, Y2, X1, X2)
            
            H[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * h_i * h_j)
            H[j, i] = H[i, j]
    
    # Invert Hessian (negative because we minimized negative log-likelihood)
    # NOTE: -H can be singular in sparse-binary / high-collinearity settings.
    # Use Moore-Penrose pseudo-inverse as fallback so we can still return an
    # approximate robust SE instead of all-NaN.
    try:
        B_inv = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(-H)

    if not np.all(np.isfinite(B_inv)):
        return np.full(n_params, np.nan)
    
    # Compute individual score vectors (numerical gradient)
    scores = np.zeros((n, n_params))
    
    for k in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()
        h = eps * max(1, abs(params[k]))
        params_plus[k] += h
        params_minus[k] -= h
        
        # Per-observation log-likelihood contributions
        for i_obs in range(n):
            Y1_i = Y1[i_obs:i_obs+1]
            Y2_i = Y2[i_obs:i_obs+1]
            X1_i = X1[i_obs:i_obs+1]
            X2_i = X2[i_obs:i_obs+1]
            
            ll_plus = -_bivariate_probit_negloglik(params_plus, Y1_i, Y2_i, X1_i, X2_i)
            ll_minus = -_bivariate_probit_negloglik(params_minus, Y1_i, Y2_i, X1_i, X2_i)
            
            scores[i_obs, k] = (ll_plus - ll_minus) / (2 * h)
    
    def compute_meat(cluster_ids_local):
        """Compute meat matrix for given clustering."""
        unique_clusters = np.unique(cluster_ids_local)
        M = np.zeros((n_params, n_params))
        
        for c in unique_clusters:
            mask = cluster_ids_local == c
            S_c = scores[mask].sum(axis=0)
            M += np.outer(S_c, S_c)
        
        return M
    
    # One-way or two-way clustering
    M1 = compute_meat(cluster_ids)
    
    if cluster_ids_2 is not None:
        M2 = compute_meat(cluster_ids_2)
        # Intersection clustering
        intersection_ids = np.array([f"{a}_{b}" for a, b in zip(cluster_ids, cluster_ids_2)])
        M_int = compute_meat(intersection_ids)
        M = M1 + M2 - M_int
    else:
        M = M1
    
    # Sandwich: B^{-1} M B^{-1}
    try:
        V = B_inv @ M @ B_inv
        if not np.all(np.isfinite(V)):
            return np.full(n_params, np.nan)
    except Exception:
        return np.full(n_params, np.nan)

    variances = np.diag(V)
    if not np.all(np.isfinite(variances)):
        return np.full(n_params, np.nan)
    variances = np.where(variances < 0, 0.0, variances)
    se = np.sqrt(variances)
    
    return se


def fit_binary_frreg_robust(
    df_pairs: pd.DataFrame,
    group_by: str = 'DOR',
    covariate_cols: Optional[List[str]] = None,
    two_way_cluster: bool = True,
    min_pairs: int = 50,
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    이진형 표현형에 대한 FR-regression (Bivariate Probit + Cluster-Robust SE)
    
    Bivariate Probit 모델을 1회 적합하고, 
    cluster-robust standard error (sandwich estimator)를 계산합니다.
    Bootstrap 없이 robust SE를 직접 계산합니다.
    
    수학적 배경:
    - Bivariate Probit: (L1, L2) ~ BVN(η1, η2, ρ)
    - Sandwich SE: Var(θ̂) = B⁻¹ M B⁻¹
      - B: Hessian (정보행렬)
      - M: Cluster-summed score outer products
    
    Args:
        df_pairs: 쌍 데이터프레임 (vol_trait, rel_trait 포함)
        group_by: 그룹화 컬럼 (기본: 'DOR')
        covariate_cols: 공변량 컬럼 리스트
        two_way_cluster: True면 (volid, relid) 양방향 클러스터링
        min_pairs: 최소 pair 수
        verbose: 상세 로그 출력 여부
        
    Returns:
        pd.DataFrame: 그룹별 slope, se, n_asym, n_asym_sym, n_vols, robust_unstable, note
    """
    n_groups = df_pairs[group_by].nunique()
    
    print(f"\n[FR-reg Binary (Robust)] Starting analysis for {n_groups} groups...")
    
    if verbose:
        print(f"{'='*60}")
        print(f"  - Total pairs: {len(df_pairs):,}")
        print(f"  - Grouping: {group_by}")
        print(f"  - Clustering: {'Two-way (vol, rel)' if two_way_cluster else 'One-way (vol)'}")
    
    df = df_pairs.copy()
    vol_col = 'vol_trait'
    rel_col = 'rel_trait'
    
    if vol_col not in df.columns:
        raise ValueError(f"Missing required column: '{vol_col}'")
    
    # 공변량 설정 (age / vol_age / rel_age 모두 지원)
    cov_prefixes = _normalize_covariate_prefixes(df, covariate_cols)
    if verbose and cov_prefixes:
        print(f"  - Covariates: {cov_prefixes}")
    
    results = []
    
    for group_val in sorted(df[group_by].unique()):
        df_group = df[df[group_by] == group_val].copy()
        # Asymmetric Pool (중복 제거, 대칭화 안 함)
        df_asym = drop_symmetric_duplicates(df_group, vol_col='volid', rel_col='relid')
        n_asym = len(df_asym)
        n_asym_sym = n_asym
        n_vols = df_asym['volid'].nunique()
        
        if n_vols < min_pairs:
            if verbose:
                    print(f"\n  [{group_by}={group_val}] Skipping due to small sample size (volunteers: {n_vols} < {min_pairs})")
            continue
        
        if verbose:
            print(f"\n  [{group_by}={group_val}] Analyzing {n_asym:,} pairs (volunteers: {n_vols:,})...")
            print(f"    [Step 1] Asymmetric pool: {n_asym:,} -> {n_asym:,}")

        # 방향성 무시를 위해 vol↔rel 쌍을 반전해 concat
        df_asym = processing.symmetrize_pairs(df_asym, verbose=False)
        n_asym_sym = len(df_asym)
        if verbose:
            print(f"    [Step 1] Symmetrized pool: {n_asym:,} -> {n_asym_sym:,}")
        
        # 데이터 준비
        Y1 = df_asym[vol_col].values.astype(float)
        Y2 = df_asym[rel_col].values.astype(float)
        
        X1_list = [np.ones(n_asym_sym)]
        X2_list = [np.ones(n_asym_sym)]
        
        for col in cov_prefixes:
            X1_list.append(df_asym[f'vol_{col}'].values)
            X2_list.append(df_asym[f'rel_{col}'].values)
        
        X1 = np.column_stack(X1_list)
        X2 = np.column_stack(X2_list)
        
        # 초기 ρ 추정
        n00, n01, n10, n11 = _compute_contingency_table(df_asym, vol_col, rel_col)
        rho_init, _ = _tetrachoric_correlation(n00, n01, n10, n11)
        if np.isnan(rho_init): rho_init = 0.3
        
        if verbose:
            print(f"    [Step 2] Fitting bivariate probit MLE...")
            print(f"      - Initial rho: {rho_init:.4f}")
        
        # MLE
        n_beta = X1.shape[1]
        params_init = np.concatenate([np.zeros(n_beta), [rho_init]])
        bounds = [(None, None)] * n_beta + [(1e-6, 0.99)]
        
        try:
            result = minimize(
                fun=_bivariate_probit_negloglik,
                x0=params_init,
                args=(Y1, Y2, X1, X2),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'disp': False}
            )
            
            if not result.success:
                if verbose:
                    print(f"    Warning: MLE failed to converge")
                continue

            params_hat = result.x
            rho_hat = params_hat[-1]
            
        except Exception as e:
            if verbose:
                print(f"    Warning: MLE error: {e}")
            continue
        
        if verbose:
            print(f"    [Step 3] Computing cluster-robust SE...")
        
        # Cluster IDs
        vol_ids = df_asym['volid'].values
        rel_ids = df_asym['relid'].values if two_way_cluster else None
        
        # Robust SE
        se_robust = _compute_sandwich_se(
            params_hat, Y1, Y2, X1, X2,
            cluster_ids=vol_ids,
            cluster_ids_2=rel_ids
        )

        rho_se = float(se_robust[-1]) if np.isfinite(se_robust[-1]) else float("nan")

        unstable_flags = []
        if not np.isfinite(rho_hat):
            unstable_flags.append("rho_non_finite")
        elif rho_hat >= (_ROBUST_RHO_UPPER - _ROBUST_RHO_BOUNDARY_EPS):
            unstable_flags.append("rho_boundary")

        if not np.isfinite(rho_se):
            unstable_flags.append("se_non_finite")
        elif rho_se <= _ROBUST_SE_NEAR_ZERO:
            unstable_flags.append("se_near_zero")

        if np.isfinite(rho_se) and rho_se <= _ROBUST_SE_NEAR_ZERO:
            rho_se = float("nan")
            if "se_near_zero" not in unstable_flags:
                unstable_flags.append("se_near_zero")

        result_row = {
            group_by: group_val,
            'slope': rho_hat,
            'se': rho_se,
            'n_asym': n_asym,
            'n_asym_sym': n_asym_sym,
            'n_vols': n_vols,
            'robust_unstable': bool(unstable_flags),
            'note': "; ".join(unstable_flags) if unstable_flags else "",
        }
        
        if group_by == 'relationship':
            result_row.update(relationship_group_metadata(df_group))

        results.append(result_row)
        
        if verbose:
            print(f"    → ρ = {rho_hat:.4f} ± {rho_se:.4f}")
    
    df_results = pd.DataFrame(results)
    
    print(f"[FR-reg Binary (Robust)] Done ✓ ({len(df_results)} groups analyzed)")
    
    return df_results
