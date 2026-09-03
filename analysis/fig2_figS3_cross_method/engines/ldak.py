"""LDAK QuantHer/TetraHer -- per-replicate REML (fixed s_d, constrained >=0).

Relatives file col5=relatedness(w_G^d), col6=environmental similarity(c_ij).
continuous uses --quant-her (h2O), binary uses --tetra-her --prevalence K (h2L).
Output <out>.mle: 'Genetic' row=V_G, 'Environmental' row=V_S. REML constrains
>=0, so estimates floor at the boundary.

External dependency: the LDAK binary is NOT bundled here (redistribution is
not permitted by its license) -- see this folder's README for where to get it
and where to place it. If bin/ldak is missing, run_one() below returns NaN
with note="no-ldak" and the caller just skips that method/condition; nothing
else breaks.

run_one(pairs, pheno, scale, cond, K) is one call. K is the binary prevalence
(fixed at config.K_PREV for this simulation). make(cond, K) wraps it for the
sim loop.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent          # .../engines
_LDAK = _HERE.parent / "bin" / "ldak"            # .../bin/ldak
_REL = {1: 0.5, 2: 0.25, 3: 0.125}                 # w_G^d
_ENV = {"step": {1: 1, 2: 0, 3: 0}, "const": {1: 1, 2: 1, 3: 1}}   # AE has no col6


def write_pheno(pheno, path):
    ids = pheno.index.values
    vals = pheno["phenotype"].values
    path.write_text("".join(f"{i} {i} {v:.6f}\n" for i, v in zip(ids, vals)))


def write_rel(pairs, cond, path):
    a, b, d = pairs["id1"].values, pairs["id2"].values, pairs["dor"].values
    if cond == "AE":
        lines = (f"{x} {x} {y} {y} {_REL[k]}\n" for x, y, k in zip(a, b, d))
    else:
        e = _ENV[cond]
        lines = (f"{x} {x} {y} {y} {_REL[k]} {e[k]}\n" for x, y, k in zip(a, b, d))
    path.write_text("".join(lines))


def write_covar(cov, cov_cols, path):
    """PLINK-style covar file: FID IID covar1 ... (LDAK adds the intercept itself)."""
    M = cov[cov_cols].to_numpy(float)
    lines = (f"{i} {i} " + " ".join(f"{v:.6f}" for v in row) + "\n"
             for i, row in zip(cov.index.values, M))
    path.write_text("".join(lines))


def _parse_mle(path):
    """`.mle` is a 'Component Heritability SE' table -> t[1]=point estimate, t[2]=REML asymptotic SE."""
    vg = vs = se_vg = se_vs = np.nan; conv = False
    for line in path.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "Genetic":
            vg = float(t[1]); se_vg = float(t[2]) if len(t) > 2 else np.nan
        elif t[0] == "Environmental":
            vs = float(t[1]); se_vs = float(t[2]) if len(t) > 2 else np.nan
        elif t[0] == "Converged":
            conv = t[1] == "YES"
    return vg, vs, se_vg, se_vs, conv


def run_one(pairs, pheno, scale, cond, K=None, cov=None, cov_cols=None, workdir=None):
    """One replicate x condition LDAK call. Pass workdir to keep the input
    files (useful for debugging). K = binary prevalence (ignored for
    continuous). cov=None -> no --covar (matches this simulation, which has
    no covariates)."""
    if not _LDAK.exists():
        return {"V_G": np.nan, "V_S": np.nan, "se_VG": np.nan, "se_VS": np.nan, "note": "no-ldak"}
    tmp = workdir or Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        phe = tmp / "p.pheno"; write_pheno(pheno, phe)
        rel = tmp / f"{cond}.rel"; write_rel(pairs, cond, rel)
        out = tmp / "qh"
        cmd = [str(_LDAK),
               "--quant-her" if scale == "continuous" else "--tetra-her",
               str(out), "--relatives", str(rel), "--pheno", str(phe)]
        if scale == "binary" and K is not None:
            cmd += ["--prevalence", str(K)]
        if cov is not None and cov_cols:
            cv = tmp / "p.covar"; write_covar(cov.loc[pheno.index], cov_cols, cv)
            cmd += ["--covar", str(cv)]
        subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        mle = tmp / "qh.mle"
        vg, vs, se_vg, se_vs, conv = (_parse_mle(mle) if mle.exists()
                                      else (np.nan, np.nan, np.nan, np.nan, False))
        if cond == "AE":
            vs = se_vs = np.nan                      # AE has no C -> V_C left empty
        return {"V_G": vg, "V_S": vs, "se_VG": se_vg, "se_VS": se_vs,
                "note": "conv" if conv else "NOCONV"}
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def make(cond, K=None):
    """Per-replicate runner for the sim loop. fn(rho, pairs, pheno, scale)->{V_G,V_S,note}."""
    def run(rho, pairs, pheno, scale):
        return run_one(pairs, pheno, scale, cond, K)
    return run
