# BIGFAM

<p align="center">
  <img src="figures/Fig1.png" alt="BIGFAM overview figure">
</p>

BIGFAM is a variance components analysis toolkit from relatives phenotype without genotype. This model does not require genotype information and uses only phenotype and familial relationship data as input.

## Overview

The current package supports:

- handling both continuous and binary traits
- estimating familial correlation (by DOR or relationship group)
- estimating variance components (genetic, shared environmental, and X chromosome)

The main user-facing entry point is the `BIGFAM` class:

```python
from bigfam import BIGFAM
```

## Software Requirements

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

The notebook uses the shipped fixture data under [`data/test`](./data/test).

You can also start directly from Python:

```python
from bigfam import BIGFAM

model = BIGFAM(verbose=False)

# Variance partitioning

# step 1. load data
model.load_data(
    pheno_file="data/test/continuous.tsv",
    rel_file="data/test/relationship.tsv",
    cov_file="data/test/covariate.tsv",
    trait_col="trait",
    id_col="iid",
    trait_type="continuous",
)

# step 2. estimate phenotypic correlation
df_corr = model.estimate_correlations(
    group_by="DOR",
)

# step 3. classify shared environmental decay pattern
slope_result = model.run_slope_test()

# step 4. partitioning variance components
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
