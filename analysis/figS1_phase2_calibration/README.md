# Figure S1 — Phase 2 w_hat_C calibration and separability-driven shrinkage

Reproduces the paper's **Figure S1**.

## What it shows

The Phase 2 ridge estimator predicts the shared-environment decay rate `w_C`
from the Phase 1 output `(rho_hat, Sigma_hat)`. This figure checks that
estimator in isolation, on synthetic data with a known truth:

- **a** — `w_hat_C` tracks the true `w_C`, but is pulled toward 0.5 at both
  ends of the range (shrinkage). Band = central 95% interval, dashed line =
  identity.
- **b** — a "safe-zone" map: median *retention* `r = (w_hat_C - 0.5) / (w_C - 0.5)`
  over the two axes that drive separability — signal (`V_C * (w_C - 0.5)^2`)
  on x, measurement noise (`max_d sqrt(Sigma_dd)`) on y. `r -> 1` (tracks
  truth) in the high-signal / low-noise corner; `r -> 0` (collapses to 0.5)
  in the opposite corner.

This is a synthetic diagnostic of the Phase 2 estimator itself — no phenotype
data of any kind is involved.

## Files

- `generate.py` — forward-simulates the shipped Phase 2 estimator over the
  paper's Supplement S2.4 prior (`w_C ~ U(0.01,0.99)`, `(V_G,V_S) ~
  Dirichlet(1,1,1)`, per-DOR noise `sigma_d ~ U(0.001,0.10)` with a free
  positive correlation, rejection-sampled to PSD). For each of 200,000 draws:
  builds `rho_true`, samples one noisy `rho_hat`, computes the 24 Phase 2
  features, and calls the *shipped* ridge calibration
  (`bigfam/artifacts/ws_calibration.json`, loaded via
  `bigfam.io.load.load_artifacts()`) to get `w_hat_C`. Writes `figS1.csv`.
- `plot.py` — reads `figS1.csv`, produces `figS1.png`.
- `_style.py` — shared plotting style (fonts, colors, NG-journal figure sizing).
  Local copy, not imported from elsewhere — this folder is self-contained.

## Run

```bash
cd BIGFAM.v2-publish
python3 -m venv .venv && .venv/bin/pip install -e ".[analysis]"
.venv/bin/python analysis/figS1_phase2_calibration/generate.py   # ~5-6 min, writes figS1.csv (200k rows)
.venv/bin/python analysis/figS1_phase2_calibration/plot.py       # seconds, writes figS1.png
```

## External dependencies

None beyond the `analysis` extra (`pip install -e ".[analysis]"` — adds
matplotlib, seaborn, pyarrow to the base `bigfam` install). No external
binaries, no R.

## Validation

Ran both scripts end to end in a fresh venv built from this repo only. Since
the draw is seeded (`SEED=123`, matching the paper's calibration eval draw)
and reads the same shipped artifact, `generate.py`'s output was compared
column-by-column against the frozen `figS1.csv` that produced the paper
figure:

| column | max abs diff |
|---|---|
| `w_S_true` | 3.3e-16 |
| `V_G` | 0.0 |
| `V_S` | 0.0 |
| `sigma_max` | 1.1e-16 |
| `w_hat` | 8.1e-6 |

The first four are floating-point/CSV-round-trip noise. `w_hat` picks up a
slightly larger (still tiny) diff after passing through Cholesky + 24-feature
computation + a ridge matmul — consistent with routine BLAS/numpy-version
floating-point differences across machines, not a logic difference. At this
magnitude (~1e-5 relative) it has no visible effect on the binned
means/medians the figure actually plots (each bin pools thousands of points).
`plot.py`'s output was visually compared against the paper's `figS1.png` —
same two-panel shape, shrinkage curve, and retention map.
