"""
Shared statistical helpers for slope outputs.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def slope_to_log_stats(slope: float, se: float) -> Dict[str, float]:
    """
    Convert slope-scale estimate into log2-scale statistics.
    """
    slope = float(slope)
    se = float(se)

    if slope > 0:
        log_slope = np.log2(slope)
        log_se = se / (slope * np.log(2)) if se > 0 else np.nan
    else:
        log_slope = np.nan
        log_se = np.nan

    return {
        "slope": slope,
        "se": se,
        "log_slope": float(log_slope) if np.isfinite(log_slope) else np.nan,
        "log_se": float(log_se) if np.isfinite(log_se) else np.nan,
    }


def summarize_bootstrap_slopes(slopes: Sequence[float]) -> Dict[str, float]:
    """
    Summarize bootstrap slope samples and derive log-scale stats.
    """
    slope_mean = float(np.mean(slopes))
    slope_se = float(np.std(slopes))
    return slope_to_log_stats(slope_mean, slope_se)


def ensure_log_columns(
    df: pd.DataFrame,
    slope_col: str = "slope",
    se_col: str = "se",
    log_slope_col: str = "log_slope",
    log_se_col: str = "log_se",
) -> pd.DataFrame:
    """
    Ensure log_slope/log_se columns exist based on slope/se columns.
    """
    out = df.copy()

    if slope_col not in out.columns or se_col not in out.columns:
        raise ValueError(f"Missing required columns: {slope_col}, {se_col}")

    slope = pd.to_numeric(out[slope_col], errors="coerce")
    se = pd.to_numeric(out[se_col], errors="coerce")

    if log_slope_col not in out.columns:
        out[log_slope_col] = np.where(slope > 0, np.log2(slope), np.nan)

    if log_se_col not in out.columns:
        out[log_se_col] = np.where(
            (slope > 0) & (se > 0),
            se / (slope * np.log(2)),
            0.0,
        )

    return out
