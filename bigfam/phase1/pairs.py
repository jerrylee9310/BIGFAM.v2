"""Pair-table construction: filter to usable pairs, then flip & concat.

Reference: phase1_continuous.py / phase1_binary.py build_pair_table, flip_concat.

Duplicate handling (the input contract flip_concat assumes):
  - cov / pheno must be indexed by *unique* id. A duplicate id would make the
    later `cov.loc[ids]` expand rows and silently misalign covariates with
    phenotypes -> fail loud instead.
  - Each undirected pair must appear at most once. flip_concat always adds the
    (j, i) direction, so an input that already carries both directions (or an
    exact duplicate row) would double-count that pair: the OLS / probit point
    estimate gets reweighted toward it and the bread is inflated, distorting the
    sandwich SE. We canonicalise to sorted (lo, hi) keys and drop duplicates
    *before* flip_concat.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _require_unique_index(obj, name):
    """Fail loud on a duplicated id index (would misalign cov/pheno on .loc)."""
    dup = obj.index[obj.index.duplicated()].unique()
    if len(dup):
        raise ValueError(
            f"{name} must be indexed by unique id; "
            f"{len(dup)} duplicated id(s), e.g. {list(dup[:5])}")


def _dedup_pairs(pairs):
    """Canonicalise to undirected (lo, hi) keys and drop duplicate pairs.

    - id1 == id2 self-pairs -> error (degenerate, a data bug).
    - same pair with conflicting dor -> error (a data bug, not a safe drop).
    - exact / both-direction duplicates -> dropped, with a warning of the count.
    """
    p = pairs.copy()
    id1 = p["id1"].values
    id2 = p["id2"].values

    swap = id1 > id2                       # dtype-agnostic (numeric or string ids)
    lo = np.where(swap, id2, id1)
    hi = np.where(swap, id1, id2)

    if (lo == hi).any():
        n = int((lo == hi).sum())
        raise ValueError(f"{n} self-pair(s) with id1 == id2")

    p["_lo"], p["_hi"] = lo, hi
    n_dor = p.groupby(["_lo", "_hi"])["dor"].nunique()
    bad = n_dor[n_dor > 1]
    if len(bad):
        raise ValueError(
            f"conflicting dor for {len(bad)} pair(s), e.g. {list(bad.index[:5])}")

    before = len(p)
    p = p.drop_duplicates(subset=["_lo", "_hi"])
    dropped = before - len(p)
    if dropped:
        warnings.warn(
            f"dropped {dropped} duplicate pair row(s) "
            f"({before} -> {len(p)}) before flip & concat",
            stacklevel=2)
    return p.drop(columns=["_lo", "_hi"]).reset_index(drop=True)


def build_pair_table(pairs, cov, pheno, cov_cols, binary=False):
    """Keep pairs where both members have phenotype + covariate, attach values.

    Expects pairs with columns id1, id2, dor; pheno indexed by id with a
    'phenotype' column; cov indexed by id with cov_cols. cov / pheno must have
    a unique id index, and each undirected pair must appear at most once
    (see module docstring).
    """
    _require_unique_index(cov, "cov")
    _require_unique_index(pheno, "pheno")

    valid = pheno.index.intersection(cov.index)
    p = pairs[pairs["id1"].isin(valid) & pairs["id2"].isin(valid)].copy()
    p = _dedup_pairs(p)

    for col in cov_cols:
        p[f"x1_{col}"] = cov.loc[p["id1"].values, col].values
        p[f"x2_{col}"] = cov.loc[p["id2"].values, col].values
    y1 = pheno.loc[p["id1"].values, "phenotype"].values
    y2 = pheno.loc[p["id2"].values, "phenotype"].values
    if binary:
        y1 = y1.astype(float)
        y2 = y2.astype(float)
    p["y1"] = y1
    p["y2"] = y2
    return p.reset_index(drop=True)


def flip_concat(p, cov_cols):
    """Undirected: concat original + flipped (i,j) -> (j,i)."""
    flip = p.copy()
    flip["y1"] = p["y2"].values
    flip["y2"] = p["y1"].values
    flip["id1"] = p["id2"].values
    flip["id2"] = p["id1"].values
    for col in cov_cols:
        flip[f"x1_{col}"] = p[f"x2_{col}"].values
        flip[f"x2_{col}"] = p[f"x1_{col}"].values
    return pd.concat([p, flip], ignore_index=True)
