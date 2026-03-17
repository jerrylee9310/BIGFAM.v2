import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import bigfam as bigfam_pkg  # noqa: E402
from bigfam import BIGFAM  # noqa: E402


def _load_shared_data(trait_type: str) -> BIGFAM:
    data_dir = ROOT.parent / "data" / "test"
    cfg = {
        "pheno_file": str(data_dir / "continuous.tsv"),
        "cov_file": str(data_dir / "covariate.tsv"),
        "rel_file": str(data_dir / "relationship.tsv"),
        "trait_col": "trait",
        "id_col": "iid",
    }
    model = BIGFAM(verbose=False)
    model.load_data(
        pheno_file=cfg["pheno_file"],
        cov_file=cfg["cov_file"],
        rel_file=cfg["rel_file"],
        trait_col=cfg["trait_col"],
        id_col=cfg["id_col"],
        trait_type=trait_type,
    )
    return model


def _assert_correlation_df(df):
    assert "DOR" in df.columns
    assert {"slope", "se"}.issubset(df.columns) or {"rho", "se"}.issubset(df.columns)
    assert not df.empty


def test_local_bigfam_imports_local_package():
    assert str(Path(bigfam_pkg.__file__).resolve()).startswith(str(SRC_PATH))


def test_estimate_correlations_continuous():
    model = _load_shared_data(trait_type="continuous")
    df = model.estimate_correlations(
        group_by="DOR",
        method="volsummary",
        n_bootstrap=10,
        min_pairs=1,
        aggregation="mean",
        verbose=False,
    )
    _assert_correlation_df(df)
