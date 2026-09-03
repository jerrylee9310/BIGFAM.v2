"""File output: the trained artifact and the per-run result tables."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import CalibrationCoef, RhoEstimate, Decomposition


def save_artifacts(calib: CalibrationCoef, out_dir) -> None:
    """Write ws_calibration.json -- the only file inference needs."""
    d = asdict(calib)
    for k in ("scaler_mean", "scaler_scale", "ridge_coef"):
        d[k] = [float(v) for v in np.asarray(d[k]).ravel()]
    d["clip"] = list(d["clip"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ws_calibration.json", "w") as f:
        json.dump(d, f, indent=2)


def save_rho(rho: RhoEstimate, out_dir) -> None:
    """Write rho_hat.tsv (dor, rho_hat, sigma_hat) and sigma_hat.tsv (D x D)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dors = list(range(1, rho.D + 1))
    pd.DataFrame({"dor": dors, "rho_hat": rho.rho_hat, "sigma_hat": rho.sigma_hat}) \
        .to_csv(out / "rho_hat.tsv", sep="\t", index=False)
    pd.DataFrame(rho.Sigma_hat, index=dors, columns=dors) \
        .to_csv(out / "sigma_hat.tsv", sep="\t")


def save_decomposition(result: Decomposition, out_dir) -> None:
    """Write decomposition.tsv: V_G/V_S (estimate, conditional SE, z) and w_S with its CI."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "V_G": result.V_G, "se_VG_cond": result.se_VG_cond, "z_VG": result.z_VG,
        "V_S": result.V_S, "se_VS_cond": result.se_VS_cond, "z_VS": result.z_VS,
        "w_s_cal": result.w_s_cal, "wci_lo": result.wci_lo, "wci_hi": result.wci_hi,
    }]).to_csv(out / "decomposition.tsv", sep="\t", index=False)
