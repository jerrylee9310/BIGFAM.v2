"""
IO 모듈

표현형, 공변량, 관계 데이터 로딩 함수를 제공합니다.
"""

from .loaders import load_phenotype, load_covariate, load_relationship

__all__ = [
    'load_phenotype',
    'load_covariate',
    'load_relationship',
]
