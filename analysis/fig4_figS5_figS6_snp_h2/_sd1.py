"""_sd1.py — Supplementary Data 1 loader shared by compute.py and the three plot scripts.

Reads supple_data/Supplementary_Data_1_trait_estimates.csv and renames its columns back
to the internal names the original figure scripts use (V_A -> V_G, V_C -> V_S,
w_C -> w_s_cal, h2_<Method> -> <method>_h2, ...), so everything after load_sd1() in those
scripts is the paper's code unchanged.
"""
from pathlib import Path

import matplotlib as mpl
import pandas as pd

HERE = Path(__file__).resolve().parent
SD1 = HERE.parents[1] / "supple_data" / "Supplementary_Data_1_trait_estimates.csv"

# Supplementary Data 1 column -> the analysis_set.tsv name the analysis code uses
SD1_METHOD = {"falconer": "Falconer", "herg": "HE_PCGC", "sem_step": "SEM_step",
              "sem_const": "SEM_const", "ldak_step": "QHTH_step", "ldak_const": "QHTH_const",
              "bigfam_v1": "BIGFAM_v1", "bigfam": "BIGFAM_v2"}
RENAME = {"trait_type": "kind", "V_A": "V_G", "V_C": "V_S", "w_C": "w_s_cal",
          "w_C_ci_lo": "wci_lo", "w_C_ci_hi": "wci_hi", "se_V_A": "se_VG_cond",
          "se_V_C": "se_VS_cond", "z_V_A": "z_VG", "z_V_C": "z_VS",
          "se_snp_h2": "snp_se", "snp_h2_confidence": "confidence",
          **{f"h2_{v}": f"{k}_h2" for k, v in SD1_METHOD.items()},
          **{f"se_{v}": f"{k}_se" for k, v in SD1_METHOD.items()}}


def load_sd1():
    """Supplementary Data 1 -> the frame analysis_set.tsv gave the original scripts (416 traits)."""
    if not SD1.exists():
        raise FileNotFoundError(
            f"{SD1} not found. Download Supplementary Data 1 from the paper and place it at "
            "supple_data/Supplementary_Data_1_trait_estimates.csv (supple_data/ is not committed).")
    d = pd.read_csv(SD1, dtype={"trait_id": str}).rename(columns=RENAME)
    # analysis_set.tsv was sorted by (cohort, trait id); SD1 is not. Restore that order so the
    # trait-resample bootstrap in compute.py draws the same indices as the paper's table.
    return d.sort_values(["cohort", "trait_id"]).reset_index(drop=True)


def save_png(fig, stem, tight=True):
    """PNG only, 400 dpi as in the paper (_style.save would also write a PDF)."""
    with mpl.rc_context({} if tight else {"savefig.bbox": None}):
        fig.savefig(HERE / f"{stem}.png")
    print(f"wrote {stem}.png")
