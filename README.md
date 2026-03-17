# BIGFAM

<p align="center">
  <img src="figures/Fig1.png" alt="BIGFAM overview figure">
</p>

BIGFAM is a family relationship regression toolkit for estimating familial correlation and decomposing it into genetic and shared environmental components without genotype.

## Overview

The current package supports:

- loading phenotype, relationship, and optional covariate tables
- estimating familial correlation by DOR or relationship group
- handling both continuous and binary traits
- running slope tests for environmental decay
- estimating variance components such as `V_G`, `V_S`, and `w_S`
- running relationship-level follow-up analyses for X-variance workflows

The main user-facing entry point is the `BIGFAM` class:

```python
from bigfam import BIGFAM
```

## Software Requirements

- Python 3.10
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tqdm`
- `statsmodels`
- `pyyaml`
- Jupyter environment for notebook-based walkthroughs

The full environment is defined in [`environment.yaml`](./environment.yaml).

## Installation

Create the recommended conda environment:

```bash
conda env create -f environment.yaml
conda activate bigfam
```

If you prefer to install manually in an existing environment, make sure the dependencies in [`environment.yaml`](./environment.yaml) are available and install the package in editable mode:

```bash
pip install -e .
```

## Quick Start

The recommended starting point is the notebook quickstart:

- Open [`notebook/quickstart.ipynb`](./notebook/01.quickstart.ipynb)
- Run it with a kernel that uses the project environment

The notebook uses the shipped fixture data under [`data/test`](./data/test) and walks through:

1. loading sample data
2. estimating familial correlations
3. running the slope test
4. estimating variance components

You can also start directly from Python:

```python
from bigfam import BIGFAM

model = BIGFAM(verbose=False)

model.load_data(
    pheno_file="data/test/continuous.tsv",
    rel_file="data/test/relationship.tsv",
    cov_file="data/test/covariate.tsv",
    trait_col="trait",
    id_col="iid",
    trait_type="continuous",
)

df_corr = model.estimate_correlations(
    group_by="DOR",
)

slope_result = model.run_slope_test()
var_result = model.estimate_variance_components()
```

## Input Files

Minimum expected columns:

- phenotype file: `iid`, `trait`
- relationship file: `volid`, `relid`, `DOR`
- covariate file: `iid` plus one or more covariate columns

For binary analyses, the trait column should use stable `0/1` values.


## License

This project is distributed under the BIGFAM Software License for non-commercial academic research use. See [`LICENSE`](./LICENSE).

## References

1. Lee, J. J. and Han, B. BIGFAM - variance components analysis from relatives without genotype. Nature Communications (2025). https://www.nature.com/articles/s41467-025-60502-0
2. Lee, J. J. and Han, B. BIGFAM.v2: extending genotype-free variance component analysis to binary traits. Under review.
