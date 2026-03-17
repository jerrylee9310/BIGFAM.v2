"""
Pair 데이터 생성 모듈

개인 데이터를 쌍(pair) 데이터로 병합하고, 대칭화 및 필터링을 수행합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


# =============================================================================
# 데이터 병합 (Merging to Pairs)
# =============================================================================

def merge_to_pairs(
    df_pheno: pd.DataFrame,
    df_cov: pd.DataFrame,
    df_rel: pd.DataFrame,
    pheno_id_col: str = 'iid',
    cov_id_col: str = 'iid',
    trait_col: str = 'trait',
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    표현형, 공변량, 관계 데이터를 병합하여 쌍(pair) 데이터를 생성합니다.
    
    Args:
        df_pheno: 표현형 DataFrame (id, trait)
        df_cov: 공변량 DataFrame (id, age, sex, ...)
        df_rel: 관계 DataFrame (volid, relid, DOR, ...)
        pheno_id_col: 표현형 데이터의 ID 컬럼명
        cov_id_col: 공변량 데이터의 ID 컬럼명
        trait_col: 표현형 컬럼명
        verbose: 진행 상황 출력 여부
        
    Returns:
        pd.DataFrame: 병합된 쌍 데이터
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Merge] Starting merge")
        print(f"{'='*60}")
        print(f"  - Phenotypes: {len(df_pheno):,}")
        print(f"  - Covariates: {len(df_cov):,}")
        print(f"  - Relationship pairs: {len(df_rel):,}")
    
    # 1. 표현형과 공변량 병합 (개인 단위)
    df_individual = df_pheno.merge(
        df_cov,
        left_on=pheno_id_col,
        right_on=cov_id_col,
        how='inner'
    )
    
    # ID 컬럼 정리
    if pheno_id_col != cov_id_col:
        df_individual = df_individual.drop(columns=[cov_id_col])
    
    if verbose:
        print(f"\n  [Step 1] Merge phenotype + covariates")
        print(f"    - Result: {len(df_individual):,} individuals (shared IDs)")
    
    # 2. 관계 데이터의 volid와 병합
    vol_cols = {col: f'vol_{col}' for col in df_individual.columns if col != pheno_id_col}
    df_vol = df_individual.rename(columns={pheno_id_col: 'volid', **vol_cols})
    
    df_pairs = df_rel.merge(df_vol, on='volid', how='inner')
    n_after_vol = len(df_pairs)
    
    if verbose:
        print(f"\n  [Step 2] Merge relationship + volid information")
        print(f"    - Result: {n_after_vol:,} pairs")
    
    # 3. 관계 데이터의 relid와 병합
    rel_cols = {col: f'rel_{col}' for col in df_individual.columns if col != pheno_id_col}
    df_rel_info = df_individual.rename(columns={pheno_id_col: 'relid', **rel_cols})
    
    df_pairs = df_pairs.merge(df_rel_info, on='relid', how='inner')
    n_final = len(df_pairs)
    
    if verbose:
        print(f"\n  [Step 3] Merge relationship + relid information")
        print(f"    - Result: {n_final:,} pairs")

    # Normalize the phenotype columns so downstream analysis can rely on
    # vol_trait/rel_trait regardless of the user-provided trait column name.
    vol_trait_col = f'vol_{trait_col}'
    rel_trait_col = f'rel_{trait_col}'
    rename_map = {}
    if vol_trait_col in df_pairs.columns and vol_trait_col != 'vol_trait':
        rename_map[vol_trait_col] = 'vol_trait'
    if rel_trait_col in df_pairs.columns and rel_trait_col != 'rel_trait':
        rename_map[rel_trait_col] = 'rel_trait'
    if rename_map:
        df_pairs = df_pairs.rename(columns=rename_map)

    # 4. sex_type 컬럼 생성 (있는 경우)
    # 다양한 컬럼명 패턴 지원: vol_sex, vol_sex_x, vol_sex_y 등
    vol_sex_col = None
    rel_sex_col = None
    
    for col in df_pairs.columns:
        if col.startswith('vol_sex'):
            vol_sex_col = col
            break
    for col in df_pairs.columns:
        if col.startswith('rel_sex'):
            rel_sex_col = col
            break
    
    if vol_sex_col and rel_sex_col:
        def get_sex_type(row):
            s1, s2 = int(row[vol_sex_col]), int(row[rel_sex_col])
            # sex: 0=Female, 1=Male
            if s1 == 1 and s2 == 1:
                return 'MM'
            elif s1 == 0 and s2 == 0:
                return 'FF'
            else:
                return 'MF'  # MF와 FM 모두 MF로 통일
        
        df_pairs['sex_type'] = df_pairs.apply(get_sex_type, axis=1)
        
        if verbose:
            print(f"\n  [Step 4] Create sex_type column")
            print(f"    - Source columns: {vol_sex_col}, {rel_sex_col}")
            sex_counts = df_pairs['sex_type'].value_counts()
            for st, cnt in sex_counts.items():
                print(f"    • {st}: {cnt:,} pairs")
        
    # 요약 출력
    if verbose:
        # 데이터 손실 요약
        n_lost = len(df_rel) - n_final
        loss_pct = n_lost / len(df_rel) * 100
        print(f"\n  [Summary]")
        print(f"    - Original pairs: {len(df_rel):,}")
        print(f"    - Valid pairs: {n_final:,}")
        print(f"    - Lost pairs: {n_lost:,} ({loss_pct:.1f}%) due to missing phenotype/covariate records")
        
        # DOR별 분포
        dor_counts = df_pairs['DOR'].value_counts().sort_index()
        print(f"\n    - Pairs by DOR:")
        for dor, count in dor_counts.items():
            print(f"      • DOR {dor}: {count:,} pairs")

        print(f"\n[Merge] Done ✓")
    
    return df_pairs


# =============================================================================
# 3. 대칭화 (Symmetrization)
# =============================================================================

def symmetrize_pairs(
    df_pairs: pd.DataFrame,
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    쌍 데이터를 대칭화합니다 (vol ↔ rel).
    
    (A, B) 쌍이 있으면 (B, A) 쌍도 추가하여 데이터를 두 배로 만듭니다.
    이를 통해 회귀 분석 시 양방향 정보를 모두 활용합니다.
    
    Args:
        df_pairs: 쌍 데이터
        verbose: 진행 상황 출력 여부
        
    Returns:
        pd.DataFrame: 대칭화된 쌍 데이터
    """
    if verbose:
        print(f"\n[Symmetrize] Symmetrizing pairs (flip & concat)")
        print(f"  - Input pairs: {len(df_pairs):,}")
    
    # vol ↔ rel 컬럼 매핑 생성
    flip_map = {}
    for col in df_pairs.columns:
        if col.startswith('vol_'):
            rel_col = 'rel_' + col[4:]
            if rel_col in df_pairs.columns:
                flip_map[col] = rel_col
        elif col.startswith('rel_'):
            vol_col = 'vol_' + col[4:]
            if vol_col in df_pairs.columns:
                flip_map[col] = vol_col
    
    # volid ↔ relid 추가
    flip_map['volid'] = 'relid'
    flip_map['relid'] = 'volid'
    
    # Flip된 데이터 생성
    df_flipped = df_pairs.copy().rename(columns=flip_map)
    
    # Concat
    df_symmetric = pd.concat([df_pairs, df_flipped], axis=0, ignore_index=True)
    
    if verbose:
        print(f"  - Output pairs: {len(df_symmetric):,} (doubled)")
        print(f"[Symmetrize] Done ✓")
    
    return df_symmetric


# =============================================================================
# 4. 그룹 필터링 (Group Filtering)
# =============================================================================

def filter_groups_by_size(
    df_pairs: pd.DataFrame,
    trait_col: str = 'trait',
    group_by: str = 'DOR',
    min_pairs: int = 100,
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    샘플 수(쌍 수)가 너무 적은 그룹을 분석에서 제외합니다.
    
    Args:
        df_pairs: 병합된 쌍 데이터
        trait_col: 표현형 컬럼명 (vol_trait, rel_trait의 접두어 부분)
        group_by: 그룹화 기준 ('DOR' 또는 'REL')
        min_pairs: 최소 쌍 수 (이보다 적은 그룹은 제외)
        verbose: 진행 상황 출력 여부
        
    Returns:
        pd.DataFrame: 필터링된 쌍 데이터
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Filter] Removing small groups (minimum {min_pairs} pairs required)")
        print(f"{'='*60}")
        print(f"  - Input pairs: {len(df_pairs):,}")
        print(f"  - Group by: {group_by}")
        print(f"  - Minimum pairs: {min_pairs}")
    
    df = df_pairs.copy()
    
    # 그룹별 쌍 수 확인
    group_counts = df[group_by].value_counts().sort_index()
    
    if verbose:
        print(f"\n  [Group sizes]")
        for group, count in group_counts.items():
            status = "✓" if count >= min_pairs else "✗ (excluded)"
            print(f"    • {group_by} {group}: {count:,} pairs {status}")
    
    # 최소 쌍 수 미달 그룹 제외
    valid_groups = group_counts[group_counts >= min_pairs].index
    df_filtered = df[df[group_by].isin(valid_groups)].copy()
    
    n_excluded = len(df) - len(df_filtered)
    
    if verbose:
        print(f"\n  [Filter result]")
        print(f"    - Valid groups: {len(valid_groups)}")
        print(f"    - Excluded pairs: {n_excluded:,}")
        print(f"    - Final pairs: {len(df_filtered):,}")
        print(f"\n[Filter] Done ✓")
    
    return df_filtered
