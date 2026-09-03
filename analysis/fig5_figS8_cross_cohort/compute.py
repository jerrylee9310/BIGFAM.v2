"""compute.py -- rebuild the three trait-level tables behind Fig. 5 / Fig. S8
from Supplementary Data 1.

    .venv/bin/python analysis/fig5_figS8_cross_cohort/compute.py   # prints a summary

The working repo froze these tables as three CSVs cut from the private BIGFAM
fit database. Every number in them is also in Supplementary Data 1 (SD1,
trait-level, 6 significant digits), so they are rebuilt from SD1 here:

  famhist()         GS family-history traits (18 rows)               -- Fig. 5a
  vg_matched()      cross-cohort |dV_A| in three pairing categories  -- Fig. 5b
                    (18 rows: 2 same-cohort, 4 two-cohort, 12 mis-paired)
  engine_compare()  5 GS<->UKB pairs x 8 methods (40 rows)           -- Fig. 5c, S8

Column names follow the frozen tables (package notation V_G / V_S / w_s_cal);
the manuscript and SD1 write V_A / V_C / w_C. RENAME below is the mapping.
"""
from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
SD1 = HERE.parents[1] / "supple_data" / "Supplementary_Data_1_trait_estimates.csv"

# SD1 column -> frozen-table column
RENAME = {"cohort": "dataset", "trait_id": "trait", "trait_type": "kind",
          "V_A": "V_G", "V_C": "V_S", "w_C": "w_s_cal",
          "w_C_ci_lo": "wci_lo", "w_C_ci_hi": "wci_hi",
          "se_V_A": "se_VG_cond", "se_V_C": "se_VS_cond", "z_V_A": "z_VG", "z_V_C": "z_VS",
          **{f"rho_dor{d}": f"rho{d}" for d in (1, 2, 3)},
          **{f"se_rho_dor{d}": f"se{d}" for d in (1, 2, 3)}}

# GS family-history questionnaire: trait_id = <disease>_<who has it>
INFORMANT = {"F": "father", "M": "mother", "G": "grandparent", "Y": "self"}
GROUP = {"father": "parent", "mother": "parent", "grandparent": "grandparent", "self": "self"}

# GS trait -> UKB field: the 19 same-trait matches screened for the cross-cohort
# comparison. Only pairs with z_V_A > 2 in BOTH cohorts are kept, which leaves
# Total_cholesterol, Creatinine, Creat_mgdl, avg_dia, FVC. Creat_mgdl is the
# same GS measurement as Creatinine in mg/dL, so it is a same-cohort duplicate
# (Fig. 5b category a) and is dropped from the two-cohort category.
PAIRS = [("height", "50"), ("weight", "21002"), ("bmi", "21001"), ("body_fat", "23099"),
         ("hips", "49"), ("HDL_cholesterol", "30760"), ("Total_cholesterol", "30690"),
         ("Glucose", "30740"), ("Creatinine", "30700"), ("Creat_mgdl", "30700"),
         ("Urea", "30670"), ("avg_sys", "4080"), ("avg_dia", "4079"), ("avg_hr", "102"),
         ("FEV", "20150"), ("FVC", "20151"), ("Heart_Rate", "12336"),
         ("P_duration", "12338"), ("QRS_duration", "12340")]
# same measurement expressed twice within one cohort (category a)
WITHIN = [("GS", "Creatinine", "Creat_mgdl"), ("UKB", "20151", "3062")]

# frozen-table method label, engine, assumption, SD1 column
METHODS = [("SEM-step", "SEM", "step", "h2_SEM_step"),
           ("SEM-const", "SEM", "const", "h2_SEM_const"),
           ("LDAK-step", "LDAK", "step", "h2_QHTH_step"),
           ("LDAK-const", "LDAK", "const", "h2_QHTH_const"),
           ("Falconer", "Falconer", "AE", "h2_Falconer"),
           ("HE", "HE", "AE", "h2_HE_PCGC"),
           ("BIGFAM.v2", "BIGFAM", "decay", "h2_BIGFAM_v2"),
           ("BIGFAM.v1", "BIGFAM", "decay", "h2_BIGFAM_v1")]

PULL = ["V_G", "V_S", "w_s_cal", "z_VG", "z_VS", "rho1", "rho2", "rho3", "se_VG_cond"]


def sd1():
    if not SD1.exists():
        raise FileNotFoundError(
            f"{SD1} not found. supple_data/ is not committed -- download "
            "Supplementary Data 1 from the paper and place it there.")
    return pd.read_csv(SD1, dtype={"trait_id": str}).rename(columns=RENAME)


def _cohorts(s):
    return (s[s.dataset.eq("GS")].set_index("trait"),
            s[s.dataset.eq("UKB")].set_index("trait"))


def _pool(gs, uk, drop=()):
    return [(g, u) for g, u in PAIRS
            if g not in drop and gs.z_VG[g] > 2 and uk.z_VG[u] > 2]


def _pull(idx, key, pre):
    r = idx.loc[key]
    return {f"{pre}_{k}": r[k] for k in PULL}


def famhist():
    s = sd1()
    d = s[s.dataset.eq("GS") & s.kind.eq("binary")].copy()
    parts = d.trait.str.rsplit("_", n=1)
    d["informant"] = parts.str[1].map(INFORMANT)
    d["disease"] = parts.str[0]
    d["group"] = d.informant.map(GROUP)
    d["v_shape"] = d.rho3.gt(d.rho2)
    assert len(d) == 18 and d.informant.notna().all(), d.trait.tolist()
    cols = ["dataset", "trait", "kind", "rho1", "rho2", "rho3", "se1", "se2", "se3",
            "V_G", "V_S", "se_VG_cond", "se_VS_cond", "z_VG", "z_VS", "w_s_cal",
            "wci_lo", "wci_hi", "informant", "disease", "group", "v_shape"]
    return d.sort_values(["informant", "w_s_cal"]).reset_index(drop=True)[cols]


def vg_matched():
    gs, uk = _cohorts(sd1())
    pool = _pool(gs, uk, drop=("Creat_mgdl",))
    assert [g for g, _ in pool] == ["Total_cholesterol", "Creatinine", "avg_dia", "FVC"], pool
    rows = []
    for ds, t1, t2 in WITHIN:
        idx = gs if ds == "GS" else uk
        rows.append({"layer": "a", "name": f"{ds} {t1}~{t2}", "other": t2,
                     **_pull(idx, t1, "x"), **_pull(idx, t2, "y")})
    rows += [{"layer": "b", "name": g, "other": u, **_pull(gs, g, "x"), **_pull(uk, u, "y")}
             for g, u in pool]
    rows += [{"layer": "c", "name": f"{g}~{u}", "other": u,
              **_pull(gs, g, "x"), **_pull(uk, u, "y")}
             for (g, _), (_, u) in itertools.product(pool, pool) if (g, u) not in pool]
    m = pd.DataFrame(rows)
    m["dVG"] = (m.x_V_G - m.y_V_G).abs()
    m["se_d"] = np.hypot(m.x_se_VG_cond, m.y_se_VG_cond)
    m["z"] = (m.x_V_G - m.y_V_G) / m.se_d
    m["p"] = 2 * norm.sf(m.z.abs())
    return m


def engine_compare():
    gs, uk = _cohorts(sd1())
    rows = []
    for g, u in _pool(gs, uk):
        for method, engine, cond, col in METHODS:
            a, b = float(gs.loc[g, col]), float(uk.loc[u, col])
            rows.append({"trait": g, "ukb_field": u, "snp_h2": float(uk.loc[u, "snp_h2"]),
                         "snp_conf": uk.loc[u, "snp_h2_confidence"], "method": method,
                         "engine": engine, "condition": cond, "GS": a, "UKB": b,
                         "absd": abs(a - b)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from scipy.stats import mannwhitneyu
    f = famhist()
    gp, pa = f[f.group.eq("grandparent")].w_s_cal, f[f.group.eq("parent")].w_s_cal
    print(f"famhist: {len(f)} rows | median w_C grandparent {gp.median():.3f} (n={len(gp)}) "
          f"vs parent {pa.median():.3f} (n={len(pa)}) | one-sided Mann-Whitney "
          f"P = {mannwhitneyu(gp, pa, alternative='greater').pvalue:.2e}")
    m = vg_matched()
    print(f"vg_matched: {len(m)} rows | median |dV_A| by category:",
          m.groupby("layer").dVG.median().round(3).to_dict())
    e = engine_compare()
    print(f"engine_compare: {len(e)} rows | creatinine |dV_A| by method:",
          e[e.trait.eq("Creatinine")].set_index("method").absd.round(3).to_dict())
