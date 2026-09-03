"""SEM(OpenMx) batch runner -- classical twin multigroup covariance-ML.

Classical twin SEM fits an ACE model to the observed **covariance matrix**
per relatedness group (here DOR 1/2/3) by ML (Neale & Cardon 1992). For fully
Gaussian data covariance-ML == raw FIML (same point estimate), and the sample
covariance is a sufficient statistic for a 2x2 per DOR -- so it can be batched
even though it fits from individual data.

- continuous: SEM builds the per-DOR sample **covariance** (variance free)
  from raw phenotypes itself. Feeding it evidence.py's rho_hat (a
  covariate-adjusted correlation, diag==1) would fit a correlation matrix as
  if it were a covariance -- the Cudeck (1989) error -- so the sample
  covariance is built separately here.
- binary: liability-threshold model -- liability variance is 1 by
  definition, so the tetrachoric **correlation** (diag=1) is the right input.
  evidence.py's liability rho is used directly.

Three rho values (summary statistics) are enough for SEM, but OpenMx startup
(~2s/call) is expensive, so every replicate is handed to R in one batch --
per-replicate would multiply the startup cost by thousands. This is the only
thing batched outside the main loop. No Rscript on PATH -> [] (skip).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from relatedness import DORS

_HERE = Path(__file__).resolve().parent          # .../engines


def sem_summary(pairs, pheno, scale, rho_hat):
    """Per-DOR 2x2 SEM input. continuous=sample covariance (variance free), binary=tetrachoric correlation (diag 1)."""
    if scale == "binary":
        return [np.array([[1.0, r], [r, 1.0]]) for r in rho_hat]
    y = pheno["phenotype"]
    mats = []
    for d in DORS:
        pd_ = pairs[pairs["dor"] == d]
        y1 = y.loc[pd_["id1"].values].to_numpy()
        y2 = y.loc[pd_["id2"].values].to_numpy()
        mats.append(np.cov(np.vstack([y1, y2])))     # 2x2 sample covariance (ddof=1)
    return mats


def run_sem(records, out_csv, se=False):
    """records: [{<id cols> + c{11,12,22}_d + Nd_d}, ...] -> sem.R's raw DataFrame.

    The id columns (anything besides c**_d/Nd_d) pass straight through -- this
    driver uses scale/w_s/rep.
    se=True turns on sem.R's Hessian/SE and adds se_VG/se_VS columns (not used
    by this simulation, which defaults to se=False)."""
    if shutil.which("Rscript") is None or not records:
        return pd.DataFrame()
    ev_csv = out_csv.parent / "_sem_in.csv"
    pd.DataFrame(records).to_csv(ev_csv, index=False)

    cmd = ["Rscript", str(_HERE / "sem.R"), str(ev_csv), str(out_csv)]
    if se:
        cmd.append("se")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_csv.exists():
        print(f"  [sem] FAILED rc={r.returncode}: {r.stderr.strip()[:200]}")
        return pd.DataFrame()
    return pd.read_csv(out_csv)
