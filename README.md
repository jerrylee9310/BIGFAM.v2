# BIGFAM

Given phenotypes measured on relatives, BIGFAM asks how much of their
resemblance is genetic (`V_G`) and how much comes from a shared environment
(`V_S`). It needs **no genotype data** — only phenotype values and how closely
each pair is related (degree of relatedness, `dor` = 1, 2, 3).

The idea: genetic resemblance halves with every step of relatedness
(`0.5^d`, fixed by biology), while shared-environment resemblance fades at an
unknown rate `w_S`. Because the two decay differently, the shape of the
similarity curve across `dor` separates them:

```text
rho_d = 0.5^d * V_G + w_S^(d-1) * V_S,   d = 1, 2, 3
```

Estimation runs in three phases, each taking only the previous one's output:

```text
Phase 1:  relative pairs         -> RhoEstimate   (rho_hat, Sigma_hat)
Phase 2:  RhoEstimate            -> WsEstimate    (w_s_cal)
Phase 3:  RhoEstimate+WsEstimate -> Decomposition (V_G, V_S, conditional SE, w-CI)
```

Three degree-of-relatedness levels (`D = 3`) are assumed throughout.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

## Try it (no data of your own needed)

```bash
.venv/bin/python examples/quickstart.py
```

Generates relative pairs from a known `(V_G, V_S, w_S)`, runs all three phases,
and prints the recovered estimates next to the truth. Read that file for the
exact table shapes the three phases expect.

There is also a small example dataset in `examples/toy_data/` — 12,000 pairs
over 24,000 individuals, simulated at `V_G = 0.5, V_S = 0.2, w_S = 0.2`, with a
continuous phenotype (`height`) and a binary one (`disease`) — laid out as
plain CSV in the structure the file loader reads:

```text
examples/toy_data/
├── kinpairs_dor1_3.csv              id1, id2, dor
├── covariates.csv                   id, age, sex
└── phenotypes/
    ├── continuous/height.csv        id, height
    └── binary/disease.csv           id, disease
```

```bash
.venv/bin/python scripts/run_pipeline.py examples/toy_data height continuous --cov-cols age sex
# rho_hat   = [0.4475 0.1757 0.0816]
# w_s_cal   = 0.2391  w-CI=[0.010, 0.440]
# V_G=0.5460 (z 8.02)  V_S=0.1742 (z 4.66)
```

Swap `height continuous` for `disease binary` to run the binary path. Regenerate
the files (different size, different truth) with `examples/make_toy_data.py`.

## Use on your own data

```python
import bigfam
from bigfam.io import load_artifacts

# pairs: DataFrame with columns id1, id2, dor   (direction and repeats do not matter)
# cov:   DataFrame indexed by id, one column per covariate
# pheno: DataFrame indexed by id, one 'phenotype' column (0/1 if binary)
calib = load_artifacts("artifacts/")

rho    = bigfam.estimate_rho(pairs, cov, pheno, "continuous", cov_cols=["age", "sex"])
ws     = bigfam.estimate_ws(rho, calib)
result = bigfam.decompose(rho, ws)

print(result.V_G, result.se_VG_cond)
print(result.V_S, result.se_VS_cond)
print(result.w_s_cal, (result.wci_lo, result.wci_hi))
```

Binary phenotypes take the same path with `"binary"`; Phases 2 and 3 do not
care which kind it was.

### Reading the result

`V_G` and `V_S` come with a *conditional* SE — the uncertainty left once `w_S`
is fixed at its estimate. The uncertainty in `w_S` itself is the separate
profile CI `[wci_lo, wci_hi]`, and it is also the identifiability check: at
`w_S = 0.5` genetic and shared-environment decay are indistinguishable, so a CI
covering 0.5 means this data cannot separate `V_G` from `V_S`. BIGFAM reports
the interval and leaves the cut-off to you.

## Command line

```bash
.venv/bin/python scripts/run_pipeline.py DATA_DIR NAME {continuous|binary} \
    --cov-cols age sex --out outputs/
```

`DATA_DIR` must hold the layout shown above; `.parquet` files are read in place
of `.csv` where present (needs the `parquet` extra). Any other loader that
yields the three DataFrames works just as well — the CLI is a convenience, not
the interface.

## Artifacts

`artifacts/ws_calibration.json` holds the Phase 2 ridge coefficients, learned
offline from simulated data — the only file inference reads. To reproduce it:

```bash
.venv/bin/python scripts/train_phase2.py    # writes artifacts/ws_calibration.json
```

## Tests

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/
```

## Docs

| | |
|---|---|
| [docs/01-usage.md](docs/01-usage.md) | Run it, and the shape of the input |
| [docs/02-results.md](docs/02-results.md) | What the output means |
| [docs/03-api.md](docs/03-api.md) | Reference and recipes |

The model's derivations are in the paper, not in this repository.

## License

Non-commercial academic research use — see `LICENSE`.
