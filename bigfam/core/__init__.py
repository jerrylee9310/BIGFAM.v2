"""Shared numerical utilities used across phases."""
from .nnls import nnls_2d_batch, nnls_2d
from .design import design_matrix, design_matrix_batch
from .linalg import safe_inv, nearest_psd

__all__ = [
    "nnls_2d_batch", "nnls_2d",
    "design_matrix", "design_matrix_batch",
    "safe_inv", "nearest_psd",
]
