"""
Helpers for group-level metadata attachment.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def relationship_group_metadata(df_group: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract metadata fields commonly attached when group_by='relationship'.
    """
    meta: Dict[str, Any] = {}

    if "DOR" in df_group.columns:
        meta["DOR"] = df_group["DOR"].iloc[0]
    if "Erx" in df_group.columns:
        meta["Erx"] = df_group["Erx"].iloc[0]
    if "sex_type" in df_group.columns:
        mode = df_group["sex_type"].mode()
        meta["sex_type"] = mode.iloc[0] if len(mode) > 0 else "MF"

    return meta

