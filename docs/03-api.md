# API

## Phases

```python
bigfam.estimate_rho(pairs, cov, pheno, pheno_type, cov_cols=(), D=3) -> RhoEstimate
bigfam.estimate_wc(rho, calib)                                       -> WcEstimate
bigfam.decompose(rho, wc)                                            -> Decomposition
```

`cov_cols` is empty by default: covariates are opt-in. `D` is fixed at 3.

## Types

Frozen dataclasses, so anything that builds the fields works as input.

```python
RhoEstimate      rho_hat (3,)   Sigma_hat (3, 3)   D
                 .sigma_hat = sqrt(diag(Sigma_hat)), the per-dor SEs
WcEstimate       w_c_cal
Decomposition    V_A  V_C  se_VA_cond  se_VC_cond  z_VA  z_VC  w_c_cal  wci_lo  wci_hi
CalibrationCoef  feature_order  scaler_mean  scaler_scale  ridge_coef
                 ridge_intercept  ridge_alpha  clip
```

## `bigfam.io`

```python
load_pairs(data_dir, name, pheno_type)  -> (pairs, cov, pheno)   # layout below
load_artifacts(artifacts_dir=packaged)  -> CalibrationCoef
save_rho(rho, out_dir)                  # rho_hat.tsv, sigma_hat.tsv
save_decomposition(result, out_dir)     # decomposition.tsv
save_artifacts(calib, out_dir)          # wc_calibration.json
```

## Elsewhere

```python
from bigfam.phase2.features import FEAT_ALL, extract_features   # the 24 features
from bigfam.phase3.refit import refit_fixed_wc, refit_batch     # (V_A, V_C) at a fixed w_C
from bigfam.phase3.robust import profile_ci                     # the w_C CI alone
from bigfam.phase2.train import train_all                       # offline training
from bigfam.phase2.dgm import generate_training_frame           # simulated training data
```

Constants (`D`, `CLIP`, the grids, the epsilons) are in `bigfam.config`.

## Recipes

### Report the sensitivity band, not just the point

```python
from bigfam.phase3.refit import refit_fixed_wc

for w in (result.wci_lo, result.w_c_cal, result.wci_hi):
    beta, _ = refit_fixed_wc(rho.rho_hat, rho.Sigma_hat, w)
    print(f"w={w:.2f}  V_A={beta[0]:.3f}  V_C={beta[1]:.3f}")
```

On the toy dataset `V_A` runs 0.69 → 0.55 → 0.00 across the CI, while
`se_VA_cond` is 0.068.

### Set `w_C` yourself, skipping Phase 2

```python
from bigfam import WcEstimate, decompose

decompose(rho, WcEstimate(w_c_cal=0.2)).V_A
```

### Start from a `(rho_hat, Sigma_hat)` computed elsewhere

```python
from bigfam import RhoEstimate

rho = RhoEstimate(np.array([0.45, 0.17, 0.07]), np.diag([1e-4, 1e-4, 2e-4]))
```

`Sigma_hat` must be positive definite; `bigfam.core.nearest_psd` repairs one
that is not.

### Many phenotypes

Nothing is cached between calls and Phase 1 dominates, so loop over traits and
parallelise the loop if it is slow.

### Retrain the calibration

```bash
.venv/bin/python scripts/train_phase2.py
```

40,000 simulated draws, seed 42, about a minute. A retrained artifact shifts
every downstream `w_C`.
