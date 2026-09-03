# BIGFAM

How much of the resemblance between relatives is genetic (`V_G`), and how much
is shared environment (`V_S`)? BIGFAM answers that from phenotypes and family
structure alone — no genotype data.

Genetic resemblance halves with every degree of relatedness; shared environment
fades at an unknown rate `w_S`. They decay differently, and that difference is
what separates them:

```text
rho_d = 0.5^d * V_G + w_S^(d-1) * V_S,   d = 1, 2, 3
```

Estimation runs in three phases, each taking only the previous one's output:

```text
Phase 1   relative pairs            ->  rho_hat, Sigma_hat
Phase 2   rho_hat, Sigma_hat        ->  w_S
Phase 3   rho_hat, Sigma_hat, w_S   ->  V_G, V_S, conditional SE, w_S CI
```

- [Install](#install)
- [Run it](#run-it)
- [Use it in Python](#use-it-in-python)
- [Repository map](#repository-map)
- [Documentation](#documentation)
- [License](#license)

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

## Run it

```bash
.venv/bin/python examples/quickstart.py        # simulates its own data
```

Or on the example dataset in `examples/toy_data/`, simulated at
`V_G = 0.5, V_S = 0.2, w_S = 0.2`:

```bash
.venv/bin/python scripts/run_pipeline.py examples/toy_data height continuous --cov-cols age sex
```

```text
covariates: age, sex
rho_hat   = [0.44750058 0.17570969 0.08158463]
w_s_cal   = 0.2391  w-CI=[0.010, 0.440]
V_G=0.5460 (z 8.02)  V_S=0.1742 (z 4.66)
```

`disease binary` in place of `height continuous` runs the binary path.

## Use it in Python

```python
import bigfam
from bigfam.io import load_artifacts

calib  = load_artifacts()          # the packaged Phase 2 calibration
rho    = bigfam.estimate_rho(pairs, cov, pheno, "continuous", ["age", "sex"])
ws     = bigfam.estimate_ws(rho, calib)
result = bigfam.decompose(rho, ws)

print(result.V_G, result.se_VG_cond)
print(result.w_s_cal, (result.wci_lo, result.wci_hi))
```

`pairs`, `cov` and `pheno` are three DataFrames — their shape is in
[docs/01-usage.md](docs/01-usage.md), and what comes back is in
[docs/02-results.md](docs/02-results.md).

## Repository map

| path | |
|---|---|
| `bigfam/` | the package — `phase1/`, `phase2/`, `phase3/`, plus shared `core/` and `io/` |
| `bigfam/artifacts/ws_calibration.json` | trained Phase 2 coefficients, the only file inference reads; rebuild with `scripts/train_phase2.py` |
| `examples/` | `quickstart.py`, the toy dataset and its generator |
| `scripts/` | `run_pipeline.py` (estimate), `train_phase2.py` (retrain) |
| `tests/` | `pip install -e ".[dev]"`, then `pytest tests/` |

## Documentation

| | |
|---|---|
| [docs/01-usage.md](docs/01-usage.md) | Run it, and the shape of the input |
| [docs/02-results.md](docs/02-results.md) | What the output means |
| [docs/03-api.md](docs/03-api.md) | Reference and recipes |

Derivations are in the paper, not in this repository.

## License

Non-commercial academic research use — see `LICENSE`.
