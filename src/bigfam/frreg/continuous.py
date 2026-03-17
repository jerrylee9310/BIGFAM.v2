"""
연속형 FR-regression 모듈

연속형 표현형에 대한 FR-regression을 수행합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Literal
from tqdm import tqdm
from .. import processing
from ..utils.group_meta import relationship_group_metadata
from ..utils.pairs import drop_symmetric_duplicates
from ..utils.stats import slope_to_log_stats, summarize_bootstrap_slopes
from .helpers import normalize_covariate_prefixes as _normalize_covariate_prefixes


# =============================================================================
# Helper Functions (Internal - uses "| " prefix for logging)
# =============================================================================

def _preprocess_covariates(
    df_pairs: pd.DataFrame,
    covariate_cols: List[str],
    verbose: bool = False
    ) -> pd.DataFrame:
    """
    공변량 보정 (개인 수준 잔차화)
    """
    import statsmodels.formula.api as smf
    
    # 공변량 prefix 추출 (age, vol_age, rel_age 모두 지원)
    cov_prefixes = _normalize_covariate_prefixes(df_pairs, covariate_cols)
    if not cov_prefixes:
        return df_pairs
    
    # 1. 개인 단위 테이블 생성 (vol + rel 합치기)
    vol_data = df_pairs[['volid', 'vol_trait']].copy()
    vol_data.columns = ['pid', 'trait']
    for prefix in cov_prefixes:
        vol_data[prefix] = df_pairs[f'vol_{prefix}'].values
        
    rel_data = df_pairs[['relid', 'rel_trait']].copy()
    rel_data.columns = ['pid', 'trait']
    for prefix in cov_prefixes:
        rel_data[prefix] = df_pairs[f'rel_{prefix}'].values
    
    df_indiv = pd.concat([vol_data, rel_data], ignore_index=True)
    df_indiv = df_indiv.drop_duplicates(subset=['pid'])
    n_indiv_orig = len(df_indiv)
    
    # 2. 잔차화 (regress out covariates)
    formula = "trait ~ 1 + " + " + ".join(cov_prefixes)
    model = smf.ols(formula, data=df_indiv).fit()
    resid = model.resid
    
    # 3. 표준화
    z_score = (resid - resid.mean()) / resid.std()
    
    # 4. 아웃라이어 제거
    outlier_threshold = 3.0
    outliers = z_score.abs() > outlier_threshold
    n_outliers = outliers.sum()
    
    df_indiv['trait'] = z_score
    df_indiv = df_indiv[~outliers].copy()
    
    # 5. 재표준화
    df_indiv['trait'] = (df_indiv['trait'] - df_indiv['trait'].mean()) / df_indiv['trait'].std()
    
    # 6. 전처리된 trait을 df_pairs에 적용
    trait_map = dict(zip(df_indiv['pid'], df_indiv['trait']))
    df_pairs = df_pairs.copy()
    df_pairs['vol_trait'] = df_pairs['volid'].map(trait_map)
    df_pairs['rel_trait'] = df_pairs['relid'].map(trait_map)
    
    # NaN 제거 (아웃라이어로 제거된 개인이 포함된 pair)
    n_before = len(df_pairs)
    df_pairs = df_pairs.dropna(subset=['vol_trait', 'rel_trait'])
    n_after = len(df_pairs)
    
    if verbose:
        # print(f"  | Covariate adjustment: {n_indiv_orig:,} individuals → {len(df_indiv):,} after outlier removal")
        if n_outliers > 0:
            print(f"  | Removed {n_outliers} outliers (|Z| > {outlier_threshold}) after covariate adjustment")
    
    return df_pairs


def _create_asymmetric_pool(
    df_group: pd.DataFrame,
    standardize: bool = True,
    verbose: bool = False
    ) -> Tuple[pd.DataFrame, int]:
    """
    Asymmetric Pool 생성 (중복 pair 제거)
    """
    df_asym = drop_symmetric_duplicates(df_group, vol_col='volid', rel_col='relid')
    n_asym = len(df_asym)
    
    # 표준화 (공변량 처리 안 했으면 여기서)
    if standardize:
        df_asym['vol_trait'] = (df_asym['vol_trait'] - df_asym['vol_trait'].mean()) / df_asym['vol_trait'].std()
        df_asym['rel_trait'] = (df_asym['rel_trait'] - df_asym['rel_trait'].mean()) / df_asym['rel_trait'].std()
    
    return df_asym, n_asym


def _bootstrap_cluster(
    df_asym: pd.DataFrame,
    n_bootstrap: int,
    verbose: bool = False,
    return_bootstrap: bool = False,
    group_by: str = 'DOR',
    group_val: Any = None
    ) -> Tuple[List[float], List[dict]]:
    """
    Cluster Bootstrap: Vol 복원추출 → 친척 1명 랜덤 선택 → 대칭화 → 회귀
    """
    vol_indices = df_asym.groupby('volid').groups
    vol_ids = list(vol_indices.keys())
    
    slopes = []
    bootstrap_results = []
    
    iterator = range(n_bootstrap)
    if verbose and n_bootstrap > 10:
        iterator = tqdm(range(n_bootstrap), desc="  | Bootstrap", leave=False)
        
    for _ in iterator:
        resampled_vols = np.random.choice(vol_ids, size=len(vol_ids), replace=True)
        boot_indices = [np.random.choice(vol_indices[v]) for v in resampled_vols]
        df_boot_asym = df_asym.loc[boot_indices]
        df_boot = processing.symmetrize_pairs(df_boot_asym, verbose=False)
        
        # standardize
        df_boot['vol_trait'] = (df_boot['vol_trait'] - df_boot['vol_trait'].mean()) / df_boot['vol_trait'].std()
        df_boot['rel_trait'] = (df_boot['rel_trait'] - df_boot['rel_trait'].mean()) / df_boot['rel_trait'].std()
        
        # print(df_boot.head())
        # print(df_boot[["vol_trait", "rel_trait"]].agg(["mean", "var"]))

        y = df_boot['vol_trait'].values
        X = df_boot['rel_trait'].values
        
        denom = X @ X
        if denom == 0:
            continue
        slope = (X @ y) / denom
        slope = max(slope, 1e-6)
        slopes.append(slope)
        
        if return_bootstrap:
            bootstrap_results.append({group_by: group_val, 'idx': len(slopes)-1, 'slope': slope})
    
    return slopes, bootstrap_results


def _bootstrap_volsummary(
    df_asym: pd.DataFrame,
    n_bootstrap: int,
    aggregation: str = 'mean',
    verbose: bool = False,
    return_bootstrap: bool = False,
    group_by: str = 'DOR',
    group_val: Any = None
    ) -> Tuple[List[float], List[dict]]:
    """
    Vol-level Summary Bootstrap: Vol별 요약 후 Vol만 복원추출
    """
    vol_col = 'vol_trait'
    rel_col = 'rel_trait'
    
    # pair별 A, B 계산 (대칭화 "접기")
    df_asym = df_asym.copy()
    df_asym['A'] = 2.0 * df_asym[vol_col] * df_asym[rel_col]
    df_asym['B'] = df_asym[vol_col]**2 + df_asym[rel_col]**2
    
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
    
    slopes = []
    bootstrap_results = []
    
    iterator = range(n_bootstrap)
    if verbose and n_bootstrap > 10:
        iterator = tqdm(range(n_bootstrap), desc="  | Bootstrap", leave=False)
    
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
            bootstrap_results.append({group_by: group_val, 'idx': len(slopes)-1, 'slope': slope})
    
    return slopes, bootstrap_results


def _compute_statistics(slopes: List[float]) -> Dict[str, float]:
    """
    Bootstrap 결과로부터 통계량 계산
    """
    return summarize_bootstrap_slopes(slopes)


def _compute_statistics_from_point_estimate(
    slope: float,
    se: float
    ) -> Dict[str, float]:
    """
    Point estimate + SE로부터 통계량 계산 (robust 방식용)
    """
    return slope_to_log_stats(slope, se)


def _fit_robust_slope(
    df_asym: pd.DataFrame,
    two_way_cluster: bool = True
    ) -> Tuple[float, float]:
    """
    OLS slope + cluster-robust SE (sandwich).
    Binary robust와 동일한 B^{-1}MB^{-1} 형태를 직접 계산합니다.
    """
    y = df_asym['vol_trait'].to_numpy(dtype=float)
    x = df_asym['rel_trait'].to_numpy(dtype=float)
    vol_ids = df_asym['volid'].to_numpy()
    rel_ids = df_asym['relid'].to_numpy()

    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 3:
        return np.nan, np.nan

    y = y[valid]
    x = x[valid]
    vol_ids = vol_ids[valid]
    rel_ids = rel_ids[valid]

    X = x.reshape(-1, 1)  # no intercept
    XtX = X.T @ X

    if not np.isfinite(XtX[0, 0]) or XtX[0, 0] <= 0:
        return np.nan, np.nan

    try:
        B_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    beta_hat = B_inv @ (X.T @ y)
    slope = float(beta_hat[0])

    resid = y - X[:, 0] * slope
    scores = X * resid.reshape(-1, 1)  # per-observation score: x_i * u_i

    def _cluster_meat(cluster_ids: np.ndarray) -> np.ndarray:
        m = np.zeros((1, 1), dtype=float)
        for cid in np.unique(cluster_ids):
            s_c = scores[cluster_ids == cid].sum(axis=0)
            m += np.outer(s_c, s_c)
        return m

    M1 = _cluster_meat(vol_ids)
    if two_way_cluster:
        M2 = _cluster_meat(rel_ids)
        intersection_ids = np.array([f"{a}__{b}" for a, b in zip(vol_ids, rel_ids)], dtype=object)
        M12 = _cluster_meat(intersection_ids)
        M = M1 + M2 - M12
    else:
        M = M1

    try:
        V = B_inv @ M @ B_inv
        variance = float(V[0, 0])
    except Exception:
        variance = np.nan

    robust_se = float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else np.nan
    return slope, robust_se


# =============================================================================
# Main Function
# =============================================================================

def fit_continuous_frreg(
    df_pairs: pd.DataFrame,
    group_by: str = 'DOR',
    method: Literal['bootstrap', 'volsummary', 'robust'] = 'robust',
    n_bootstrap: int = 100,
    aggregation: str = 'mean',

    covariate_cols: Optional[List[str]] = None,
    min_pairs: int = 100,
    
    verbose: bool = False,
    return_bootstrap: bool = False,
    two_way_cluster: bool = True
    ) -> pd.DataFrame:
    """
    연속형 표현형에 대한 FR-regression 수행
    
    Args:
        df_pairs: 쌍 데이터프레임 (vol_trait, rel_trait 포함)
        group_by: 그룹화 컬럼 (기본: 'DOR')
        method: Bootstrap 방법 ('bootstrap', 'volsummary', or 'robust')
        n_bootstrap: 부트스트랩 반복 횟수
        aggregation: Vol별 요약 방법 (method='volsummary'일 때만 사용)
        covariate_cols: 공변량 컬럼 목록
        two_way_cluster: method='robust'에서 two-way clustering 적용 여부
        min_pairs: 최소 volunteer 수
        verbose: 상세 로그 출력 여부
        return_bootstrap: True면 (df_results, df_bootstrap) 튜플 반환
        
        Returns:
            pd.DataFrame: 그룹별 slope, se, n_asym, n_asym_sym, n_vols
    """
    n_groups = df_pairs[group_by].nunique()
    vol_col = 'vol_trait'
    
    if vol_col not in df_pairs.columns:
        raise ValueError(f"Missing required column: '{vol_col}'")
    
    # 공변량 보정 (전체 데이터에 대해 1회 수행)
    if covariate_cols and len(covariate_cols) > 0:
        df_pairs = _preprocess_covariates(df_pairs, covariate_cols, verbose=verbose)
        standardize_in_pool = False
    else:
        standardize_in_pool = True
    
    # 그룹별 분석
    results = []
    bootstrap_results = []
    
    groups = sorted(df_pairs[group_by].unique())
    for idx, group_val in enumerate(groups):
        df_group = df_pairs[df_pairs[group_by] == group_val].copy()

        # Asymmetric pool을 만들기 전에 중복된 방향을 제거하고 unique vol 개수를 계산
        # (bootstrap/robust에서 사용될 n_asym 및 n_asym_sym은 raw 그룹 기준으로 고정)
        df_asym, n_asym = _create_asymmetric_pool(
            df_group,
            standardize=standardize_in_pool,
            verbose=verbose
        )
        n_vols = df_asym['volid'].nunique()
        n_asym_sym = n_asym
        
        if n_vols < min_pairs:
            continue
        
        # std   
        df_asym['vol_trait'] = (df_asym['vol_trait'] - df_asym['vol_trait'].mean()) / df_asym['vol_trait'].std()
        df_asym['rel_trait'] = (df_asym['rel_trait'] - df_asym['rel_trait'].mean()) / df_asym['rel_trait'].std()
        # print(df_asym[["vol_trait", "rel_trait"]].agg(["mean", "var"]))

        # 추정 (method에 따라)
        if method == 'bootstrap':
            n_asym_sym = n_asym * 2
            slopes, boot_results = _bootstrap_cluster(
                df_asym, n_bootstrap,
                verbose=verbose,
                return_bootstrap=return_bootstrap,
                group_by=group_by,
                group_val=group_val
                )
            if not slopes:
                continue
            stats = _compute_statistics(slopes)
            slope = float(stats["slope"])
            se = float(stats["se"])
            n_bootstrap_requested = int(n_bootstrap)
            n_bootstrap_success = len(slopes)
            n_bootstrap_fail = max(n_bootstrap_requested - n_bootstrap_success, 0)
            bootstrap_success_rate = float(n_bootstrap_success / n_bootstrap_requested) if n_bootstrap_requested > 0 else 0.0
        elif method == 'volsummary':
            n_asym_sym = n_asym
            slopes, boot_results = _bootstrap_volsummary(
                df_asym, n_bootstrap, aggregation,
                verbose=verbose,
                return_bootstrap=return_bootstrap,
                group_by=group_by,
                group_val=group_val
            )
            if not slopes:
                continue
            stats = _compute_statistics(slopes)
            slope = float(stats["slope"])
            se = float(stats["se"])
            n_bootstrap_requested = int(n_bootstrap)
            n_bootstrap_success = len(slopes)
            n_bootstrap_fail = max(n_bootstrap_requested - n_bootstrap_success, 0)
            bootstrap_success_rate = float(n_bootstrap_success / n_bootstrap_requested) if n_bootstrap_requested > 0 else 0.0
        elif method == 'robust':
            boot_results = []
            df_robust = processing.symmetrize_pairs(df_asym, verbose=False)
            n_asym_sym = len(df_robust)
            slope, robust_se = _fit_robust_slope(
                df_robust,
                two_way_cluster=two_way_cluster
            )
            if np.isnan(slope):
                continue
            slope = float(slope)
            se = float(robust_se)
            n_bootstrap_requested = None
            n_bootstrap_success = None
            n_bootstrap_fail = None
            bootstrap_success_rate = None
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bootstrap', 'volsummary', or 'robust'.")
        
        bootstrap_results.extend(boot_results)
        
        result_row = {
            group_by: group_val,
            'slope': slope,
            'se': se,
            'n_asym': n_asym,
            'n_asym_sym': n_asym_sym,
            'n_vols': n_vols,
            'n_bootstrap_requested': n_bootstrap_requested,
            'n_bootstrap_success': n_bootstrap_success,
            'n_bootstrap_fail': n_bootstrap_fail,
            'bootstrap_success_rate': bootstrap_success_rate,
        }
        
        # group_by='relationship'일 때 추가 정보
        if group_by == 'relationship':
            result_row.update(relationship_group_metadata(df_group))
        
        results.append(result_row)
        
        if verbose:
            print(f"  | {group_by}={group_val}: slope={slope:.4f} ± {se:.4f}")
    
    df_results = pd.DataFrame(results)
    
    if return_bootstrap:
        df_bootstrap = pd.DataFrame(bootstrap_results)
        return df_results, df_bootstrap
    
    return df_results


# =============================================================================
# Backward Compatibility
# =============================================================================

def fit_continuous_frreg_volsummary(
    df_pairs: pd.DataFrame,
    group_by: str = 'DOR',
    n_bootstrap: int = 100,
    aggregation: str = 'mean',
    covariate_cols: Optional[List[str]] = None,
    min_pairs: int = 100,
    verbose: bool = False,
    return_bootstrap: bool = False
    ) -> pd.DataFrame:
    """[DEPRECATED] Use fit_continuous_frreg(method='volsummary') instead."""
    import warnings
    warnings.warn(
        "fit_continuous_frreg_volsummary is deprecated. "
        "Use fit_continuous_frreg(method='volsummary') instead.",
        DeprecationWarning, stacklevel=2
    )
    return fit_continuous_frreg(
        df_pairs=df_pairs, group_by=group_by, method='volsummary',
        n_bootstrap=n_bootstrap, aggregation=aggregation,
        covariate_cols=covariate_cols, min_pairs=min_pairs,
        verbose=verbose, return_bootstrap=return_bootstrap
    )
