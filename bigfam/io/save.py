"""File output: artifacts (offline) and result TSV/JSON (§IO). Side effects live here."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import CalibrationCoef, RhoEstimate, WsEstimate, Decomposition


def _to_jsonable(calib: CalibrationCoef) -> dict:
    d = asdict(calib)
    for k in ("scaler_mean", "scaler_scale", "ridge_coef"):
        d[k] = list(map(float, np.asarray(d[k]).ravel()))
    d["clip"] = list(d["clip"])
    return d


def save_artifacts(calib: CalibrationCoef, out_dir) -> None:
    """Write ws_calibration.json (the only inference artifact; Step 2 is ridge-only)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ws_calibration.json", "w") as f:
        json.dump(_to_jsonable(calib), f, indent=2)


# ── per-trait output (dict builders; the batch runner json.dumps them) ─────────
def rho_json_dict(rho: RhoEstimate, trait: str, kind: str, status: str) -> dict:
    """Phase 1 sufficient statistics (the {trait}.rho contract)."""
    return {
        "trait": trait, "kind": kind,
        "dor": list(range(1, rho.D + 1)),
        "rho_hat": [float(x) for x in rho.rho_hat],
        "Sigma_hat": np.asarray(rho.Sigma_hat, dtype=float).tolist(),
        "status": status,
    }


def decomp_json_dict(dec: Decomposition) -> dict:
    """Final result grouped by estimand (the {trait}.decomp contract).

    One block per estimand: V_G/V_S as est/se/z, w_s as est plus its 95% CI.
    """
    return {
        "V_G": {"est": float(dec.V_G), "se": float(dec.se_VG_cond), "z": float(dec.z_VG)},
        "V_S": {"est": float(dec.V_S), "se": float(dec.se_VS_cond), "z": float(dec.z_VS)},
        "w_s": {"est": float(dec.w_s_cal),
                "ci95": [float(dec.wci_lo), float(dec.wci_hi)]},
    }


# ── single-run TSVs (run_pipeline CLI) ────────────────────────────────────────
def save_rho(rho: RhoEstimate, out_dir) -> None:
    """rho_hat.tsv (dor, rho_hat, sigma_hat) + sigma_hat.tsv (D x D)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dors = list(range(1, rho.D + 1))
    pd.DataFrame({"dor": dors, "rho_hat": rho.rho_hat, "sigma_hat": rho.sigma_hat}) \
        .to_csv(out / "rho_hat.tsv", sep="\t", index=False)
    pd.DataFrame(rho.Sigma_hat, index=dors, columns=dors) \
        .to_csv(out / "sigma_hat.tsv", sep="\t")


def save_decomposition(result: Decomposition, out_dir) -> None:
    """decomposition.tsv: V_G/V_S (est, cond SE, z) + w_s (est, w-CI)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "V_G": result.V_G, "se_VG_cond": result.se_VG_cond, "z_VG": result.z_VG,
        "V_S": result.V_S, "se_VS_cond": result.se_VS_cond, "z_VS": result.z_VS,
        "w_s_cal": result.w_s_cal, "wci_lo": result.wci_lo, "wci_hi": result.wci_hi,
    }]).to_csv(out / "decomposition.tsv", sep="\t", index=False)


def save_result(rho: RhoEstimate, ws: WsEstimate, result: Decomposition, out_dir) -> None:
    """Write the single-run result TSVs (ws kept for signature stability)."""
    save_rho(rho, out_dir)
    save_decomposition(result, out_dir)
