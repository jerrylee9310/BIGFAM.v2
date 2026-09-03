# BIGFAM

Relative-pair variance decomposition into genetic (`V_G`) and shared-environment
(`V_S`) components, with the environmental decay rate `w_S` as an intermediate
target. BIGFAM does not require genotype data — only phenotype and familial
relationship (degree-of-relatedness) data.

Three phases connected only by small typed data objects:

```text
Phase 1:  relative pairs         -> RhoEstimate   (rho_hat, Sigma_hat)
Phase 2:  RhoEstimate            -> WsEstimate    (w_s_cal)
Phase 3:  RhoEstimate+WsEstimate -> Decomposition (V_G, V_S, conditional SE, w-CI)
```

`D = 3` (three degree-of-relatedness levels) is hard-coded throughout.

Phase 3 emits no trust label. Whether the decomposition is identified is read off
the profile CI for `w_S`: at `w_S = 0.5` the design matrix is rank 1, so a CI
covering 0.5 means `V_G` and `V_S` are not separable. Where to cut that interval
belongs to the analysis, not the estimator.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

## Quickstart (synthetic data, no real phenotypes needed)

```bash
.venv/bin/python examples/quickstart.py
```

Generates synthetic relative pairs from a known `(V_G, V_S, w_S)`, runs the
full Phase 1 -> 2 -> 3 pipeline, and prints the recovered estimate next to the
truth. See `examples/quickstart.py` for the pair/covariate/phenotype table
shapes `bigfam.estimate_rho` expects.

## Use on your own data

```python
import bigfam
from bigfam.io import load_artifacts
from bigfam.config import COV_COLS

# pairs: DataFrame[id1, id2, dor]
# cov:   DataFrame indexed by id, columns = cov_cols
# pheno: DataFrame indexed by id, single 'phenotype' column
calib = load_artifacts("artifacts/")

rho    = bigfam.estimate_rho(pairs, cov, pheno, "continuous", COV_COLS)
ws     = bigfam.estimate_ws(rho, calib)
result = bigfam.decompose(rho, ws)
```

`bigfam.io.load_pairs` is a convenience loader for a fixed parquet layout
(`db/01_processed/<dataset>/`); it is not required — any code that produces
the three tables above works.

## Artifacts

The Phase 2 calibration (`artifacts/ws_calibration.json`) is learned offline
by simulation — it is the only inference artifact, and the one shipped here
was fit on synthetic data, not real phenotypes. Inference loads it; it never
re-trains. To reproduce it:

```bash
.venv/bin/python scripts/train_phase2.py    # writes artifacts/ws_calibration.json
```

## Tests

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/
```

## Docs

Model and derivations are in `docs/` — `docs/README.md` for the big picture,
`docs/method/phase0.md`..`phase3.md` for the formal treatment per phase.

## License

Non-commercial academic research use — see `LICENSE`.
