# Usage

## 1. Run the example

```bash
# install venv
python3 -m venv .venv && .venv/bin/pip install -e .

# quickstart
.venv/bin/python examples/quickstart.py
```

```text
rho_hat = [0.44112972 0.14486396 0.07649728]
w_C:  true=0.200  est=0.123  CI=[0.010, 0.180]
V_A:  true=0.500  est=0.513  se=0.032
V_C:  true=0.200  est=0.184  se=0.018
(se is conditional on w_C)
```



---

## 2. The input: three tables


| table   | shape                                                         |
| ------- | ------------------------------------------------------------- |
| `pairs` | columns `id1`, `id2`, `dor` — one row per relative pair       |
| `cov`   | indexed by `id`, one column per covariate                     |
| `pheno` | indexed by id, one column `phenotype` — coded 0 / 1 if binary |


`dor` is the degree of relatedness, where expected genome sharing is `0.5^dor`:
**1** parent-offspring, full siblings · **2** half siblings, grandparent-grandchild,
aunt/uncle-niece/nephew · **3** first cousins. Nothing beyond 3.



---

## 3. Run it

 

```python
# Import packages
import bigfam
from bigfam.io import load_pairs, load_artifacts
```

 

```python
# Load data
pairs, cov, pheno = load_pairs("examples/toy_data", "height", "continuous")
calib = load_artifacts()                    # the packaged w_C model; load it once
```

```python
# Run BIGFAM.v2
rho    = bigfam.estimate_rho(pairs, cov, pheno, "continuous", ["age", "sex"])   # Phase 1
wc     = bigfam.estimate_wc(rho, calib)                                         # Phase 2
result = bigfam.decompose(rho, wc)                                              # Phase 3

print(result.V_A, result.se_VA_cond)          # 0.546 0.068
print(result.w_c_cal, result.wci_lo, result.wci_hi)
```



---

## How much data do you need?

Aim for `rho.sigma_hat` (the per-`dor` SE) under about 0.01. How many pairs that
takes depends on the trait and on where `w_C` sits — closer to 0.5 needs more.
