"""
데이터 입출력 모듈

표현형, 공변량, 관계 데이터를 로드하는 함수들을 제공합니다.
각 함수는 로딩 과정과 결과를 명확하게 출력합니다.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_phenotype(
    filepath: str,
    trait_col: str = 'trait',
    id_col: str = 'iid',
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    표현형(phenotype) 데이터를 로드합니다.
    
    Args:
        filepath: TSV 파일 경로
        trait_col: 표현형 컬럼명 (기본: 'trait')
        id_col: ID 컬럼명 (기본: 'iid')
        verbose: 진행 상황 출력 여부
        
    Returns:
        pd.DataFrame: [id_col, trait_col] 컬럼을 가진 DataFrame
        
    Example:
        >>> df = load_phenotype('data/gs/continuous.tsv')
        [Phenotype] 로딩: data/gs/continuous.tsv
        [Phenotype] 전체: 10,000명, 결측값 제거 후: 9,500명
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Phenotype] Loading: {filepath}")
        print(f"{'='*60}")
    
    # 데이터 로드 (구분자 자동 감지 - 공백/탭 모두 지원)
    df = pd.read_csv(filepath, sep=r'\s+', engine='python')
    n_total = len(df)
    
    if verbose:
        print(f"  - Total rows: {n_total:,}")
        print(f"  - Columns: {list(df.columns)}")
    
    # 필요한 컬럼 확인
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' is missing. Available columns: {list(df.columns)}")
    if trait_col not in df.columns:
        raise ValueError(f"Trait column '{trait_col}' is missing. Available columns: {list(df.columns)}")
    
    # 필요한 컬럼만 선택
    df = df[[id_col, trait_col]].copy()
    
    # 결측값 제거
    n_missing = df[trait_col].isna().sum()
    df = df.dropna(subset=[trait_col])
    n_valid = len(df)
    
    if verbose:
        print(f"  - Missing values removed: {n_missing:,}")
        print(f"  - Valid rows: {n_valid:,}")
        
        # Trait 유형 판단
        unique_values = df[trait_col].unique()
        if set(unique_values).issubset({0, 1, 0.0, 1.0}):
            n_cases = (df[trait_col] == 1).sum()
            n_controls = (df[trait_col] == 0).sum()
            prevalence = n_cases / n_valid * 100
            print(f"  - Trait type: Binary (Cases: {n_cases:,}, Controls: {n_controls:,})")
            print(f"  - Prevalence: {prevalence:.2f}%")
        else:
            print(f"  - Trait type: Continuous")
            print(f"  - Mean: {df[trait_col].mean():.3f}, Std: {df[trait_col].std():.3f}")

        print(f"[Phenotype] Done ✓")
    
    return df


def load_covariate(
    filepath: str,
    id_col: str = 'iid',
    verbose: bool = True
) -> pd.DataFrame:
    """
    공변량(covariate) 데이터를 로드합니다.
    
    Sex 컬럼이 있으면 F→0, M→1로 변환합니다.
    
    Args:
        filepath: TSV 파일 경로
        id_col: ID 컬럼명 (기본: 'iid')
        verbose: 진행 상황 출력 여부
        
    Returns:
        pd.DataFrame: 공변량 DataFrame
        
    Example:
        >>> df = load_covariate('data/gs/covariate.tsv')
        [Covariate] 로딩: data/gs/covariate.tsv
        [Covariate] 컬럼: ['iid', 'age', 'sex']
        [Covariate] sex 변환: F→0, M→1
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Covariate] Loading: {filepath}")
        print(f"{'='*60}")
    
    # 데이터 로드 (구분자 자동 감지 - 공백/탭 모두 지원)
    df = pd.read_csv(filepath, sep=r'\s+', engine='python')
    n_total = len(df)
    
    if verbose:
        print(f"  - Total rows: {n_total:,}")
        print(f"  - Columns: {list(df.columns)}")
    
    # ID 컬럼 확인
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' is missing.")
    
    # Sex 변환 (F→0, M→1)
    if 'sex' in df.columns:
        unique_sex = df['sex'].unique()
        if set(unique_sex).issubset({'F', 'M', 'f', 'm'}):
            df['sex'] = df['sex'].map({'F': 0, 'M': 1, 'f': 0, 'm': 1})
            if verbose:
                n_female = (df['sex'] == 0).sum()
                n_male = (df['sex'] == 1).sum()
                print(f"  - Sex recoding: F→0, M→1 (Female: {n_female:,}, Male: {n_male:,})")
    
    # 각 공변량 통계
    cov_cols = [c for c in df.columns if c != id_col]
    if verbose:
        print(f"  - Number of covariates: {len(cov_cols)}")
        for col in cov_cols:
            if df[col].dtype in ['int64', 'float64']:
                print(f"    • {col}: mean={df[col].mean():.2f}, range=[{df[col].min():.1f}, {df[col].max():.1f}]")
        print(f"[Covariate] Done ✓")
    
    return df


def load_relationship(
    filepath: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    친척 관계(relationship) 데이터를 로드합니다.
    
    Args:
        filepath: TSV 파일 경로
        verbose: 진행 상황 출력 여부
        
    Returns:
        pd.DataFrame: 관계 DataFrame (volid, relid, DOR, ...)
        
    Example:
        >>> df = load_relationship('data/gs/relationship.tsv')
        [Relationship] 로딩: data/gs/relationship.tsv
        [Relationship] 전체 쌍: 50,000개
        [Relationship] DOR 분포: {1: 30000, 2: 15000, 3: 5000}
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Relationship] Loading: {filepath}")
        print(f"{'='*60}")
    
    # 데이터 로드
    df = pd.read_csv(filepath, sep='\t')
    n_pairs = len(df)
    
    if verbose:
        print(f"  - Total pairs: {n_pairs:,}")
        print(f"  - Columns: {list(df.columns)}")
    
    # 필수 컬럼 확인
    required_cols = ['volid', 'relid', 'DOR']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # DOR 분포
    if verbose:
        dor_counts = df['DOR'].value_counts().sort_index()
        print(f"  - DOR distribution:")
        for dor, count in dor_counts.items():
            print(f"    • DOR {dor}: {count:,} pairs")
        
        # 고유 개인 수
        unique_individuals = set(df['volid'].unique()) | set(df['relid'].unique())
        print(f"  - Unique individuals: {len(unique_individuals):,}")
        
        # Relationship 종류 (있으면)
        if 'relationship' in df.columns:
            rel_counts = df['relationship'].value_counts().head(5)
            print(f"  - Relationship types (top 5):")
            for rel, count in rel_counts.items():
                print(f"    • {rel}: {count:,} pairs")

        print(f"[Relationship] Done ✓")
    
    return df
