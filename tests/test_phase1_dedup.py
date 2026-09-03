"""Phase 1 pair-table duplicate handling (pairs.build_pair_table / _dedup_pairs).

Contract checked here:
  - cov / pheno must have a unique id index            -> fail loud
  - undirected pair appears at most once (i,j)+(j,i)    -> canonical dedup
  - exact duplicate pair row                            -> dropped (warned)
  - id1 == id2 self-pair                                -> error
  - same pair with conflicting dor                      -> error
  - dedup leaves the usable table identical to clean input
"""
import numpy as np
import pandas as pd
import pytest

COV_COLS = ["age", "sex", "age_x_sex", "age2_x_sex"]   # fixture covariates
from bigfam.phase1.pairs import build_pair_table

IDS = [1, 2, 3, 4, 5]


def _cov(ids=IDS):
    rng = np.arange(len(ids), dtype=float)
    data = {c: rng + i for i, c in enumerate(COV_COLS)}
    return pd.DataFrame(data, index=pd.Index(ids, name="id"))


def _pheno(ids=IDS):
    return pd.DataFrame(
        {"phenotype": np.linspace(-1.0, 1.0, len(ids))},
        index=pd.Index(ids, name="id"),
    )


def _pairs(rows):
    return pd.DataFrame(rows, columns=["id1", "id2", "dor"])


CLEAN = _pairs([(1, 2, 1), (1, 3, 2), (2, 3, 1), (4, 5, 1)])


def _canon_keys(tbl):
    a = np.minimum(tbl["id1"].values, tbl["id2"].values)
    b = np.maximum(tbl["id1"].values, tbl["id2"].values)
    return set(zip(a.tolist(), b.tolist()))


# ── happy path ────────────────────────────────────────────────────────────────
def test_clean_pairs_pass_through():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        tbl = build_pair_table(CLEAN, _cov(), _pheno(), COV_COLS)
    assert len(tbl) == 4
    assert _canon_keys(tbl) == {(1, 2), (1, 3), (2, 3), (4, 5)}


# ── (2) pair duplicates -> dropped with a warning ──────────────────────────────
def test_exact_duplicate_pair_dropped():
    dup = pd.concat([CLEAN, _pairs([(1, 2, 1), (2, 3, 1)])], ignore_index=True)
    with pytest.warns(UserWarning, match="duplicate pair"):
        tbl = build_pair_table(dup, _cov(), _pheno(), COV_COLS)
    assert len(tbl) == 4
    assert _canon_keys(tbl) == _canon_keys(build_pair_table(CLEAN, _cov(), _pheno(), COV_COLS))


def test_both_directions_collapsed():
    both = pd.concat([CLEAN, _pairs([(2, 1, 1), (3, 1, 2)])], ignore_index=True)
    with pytest.warns(UserWarning, match="duplicate pair"):
        tbl = build_pair_table(both, _cov(), _pheno(), COV_COLS)
    assert len(tbl) == 4
    assert _canon_keys(tbl) == {(1, 2), (1, 3), (2, 3), (4, 5)}


def test_dedup_table_identical_to_clean():
    """Usable table from duplicated input == from clean input (the payoff)."""
    clean_tbl = build_pair_table(CLEAN, _cov(), _pheno(), COV_COLS)
    dup = pd.concat([CLEAN, _pairs([(1, 2, 1), (2, 1, 1)])], ignore_index=True)
    with pytest.warns(UserWarning):
        dup_tbl = build_pair_table(dup, _cov(), _pheno(), COV_COLS)
    assert dup_tbl.shape == clean_tbl.shape
    assert _canon_keys(dup_tbl) == _canon_keys(clean_tbl)


# ── (2) hard data errors -> fail loud ──────────────────────────────────────────
def test_self_pair_raises():
    bad = pd.concat([CLEAN, _pairs([(3, 3, 1)])], ignore_index=True)
    with pytest.raises(ValueError, match="self-pair"):
        build_pair_table(bad, _cov(), _pheno(), COV_COLS)


def test_conflicting_dor_raises():
    bad = pd.concat([CLEAN, _pairs([(1, 2, 3)])], ignore_index=True)  # (1,2) also dor=1
    with pytest.raises(ValueError, match="conflicting dor"):
        build_pair_table(bad, _cov(), _pheno(), COV_COLS)


# ── (3) duplicate cov / pheno index -> fail loud ───────────────────────────────
def test_duplicate_cov_index_raises():
    cov = pd.concat([_cov(), _cov([1])])        # id 1 twice
    with pytest.raises(ValueError, match="cov must be indexed by unique id"):
        build_pair_table(CLEAN, cov, _pheno(), COV_COLS)


def test_duplicate_pheno_index_raises():
    pheno = pd.concat([_pheno(), _pheno([2])])  # id 2 twice
    with pytest.raises(ValueError, match="pheno must be indexed by unique id"):
        build_pair_table(CLEAN, _cov(), pheno, COV_COLS)
