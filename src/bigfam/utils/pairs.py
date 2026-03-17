"""
Pair-level utility functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_symmetric_duplicates(
    df_pairs: pd.DataFrame,
    vol_col: str = "volid",
    rel_col: str = "relid",
) -> pd.DataFrame:
    """
    Remove duplicated unordered pairs like (A, B) and (B, A).

    This returns an asymmetric pool while preserving all original columns.
    """
    required = [vol_col, rel_col]
    missing = [c for c in required if c not in df_pairs.columns]
    if missing:
        raise ValueError(f"Missing required columns for pair deduplication: {missing}")

    df = df_pairs.copy()

    left = df[vol_col].astype(str).to_numpy()
    right = df[rel_col].astype(str).to_numpy()
    pair_id = np.where(left <= right, left + "__" + right, right + "__" + left)

    df["_pair_id"] = pair_id
    df = df.drop_duplicates(subset=["_pair_id"]).drop(columns=["_pair_id"])
    return df
