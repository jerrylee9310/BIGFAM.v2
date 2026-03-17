"""
bigfam: Familial Relationship Regression Analysis

가족 관계를 활용한 분산 성분(유전/환경) 추정 패키지.

Module Structure:
- io: 데이터 입출력
- processing: 데이터 전처리
- frreg: FR-regression (연속형/이진형)
- estimation: Slope Test, 분산 추정, 환경 분산 추정
- utils: 공통 유틸리티
"""

from .bigfam import BIGFAM
from .io import load_phenotype, load_covariate, load_relationship
from .processing import merge_to_pairs, filter_groups_by_size, clean_individual_continuous, symmetrize_pairs
from .frreg import (
    fit_continuous_frreg,
    fit_continuous_frreg_volsummary,
    fit_binary_frreg,
    fit_binary_frreg_liability,
    fit_binary_frreg_robust
)
from .estimation import (
    run_slope_test,
    estimate_variance_components,
    estimate_pairwise_variance_components,
    estimate_x_variance,
    estimate_sex_specific_x
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    'BIGFAM',
    
    # IO functions
    'load_phenotype',
    'load_covariate', 
    'load_relationship',
    
    # Processing functions
    'merge_to_pairs',
    'filter_groups_by_size',
    'clean_individual_continuous',
    'symmetrize_pairs',
    
    # FR-reg functions (Continuous)
    'fit_continuous_frreg',
    'fit_continuous_frreg_volsummary',
    
    # FR-reg functions (Binary)
    'fit_binary_frreg',
    'fit_binary_frreg_liability',
    'fit_binary_frreg_robust',
    
    # Estimation functions
    'run_slope_test',
    'estimate_variance_components',
    'estimate_pairwise_variance_components',
    'estimate_x_variance',
    'estimate_sex_specific_x',
]
