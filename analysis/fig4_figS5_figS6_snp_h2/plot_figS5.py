"""Figure S5 -- the Fig. 4a correlation ladder split by trait type.

    .venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_figS5.py

Input:  supple_data/Supplementary_Data_1_trait_estimates.csv
Output: figS5.png

Shows that Fig. 4a's pooled result is not an artefact of mixing trait types:
the same ordering holds for continuous and binary traits separately. All eight
methods are drawn (color = model class, line style separates methods within a
class). Above threshold 0.20 the binary subset has n < 12, so those points come
from the merged estimate and carry no CI.
"""
from __future__ import annotations

import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
import numpy as np                                                 # noqa: E402
import seaborn as sns                                              # noqa: E402

from _sd1 import load_sd1, save_png
from _style import C, W2, rc

rc()
STY = {
    "bigfam":     dict(color=C["decay"],    ls="-",  lw=1.6, marker="o", ms=3.0),
    "bigfam_v1":  dict(color=C["decay_v1"], ls="-",  lw=1.2, marker="s", ms=2.6),
    "sem_const":  dict(color=C["const"], ls="-",  lw=0.9),
    "ldak_const": dict(color=C["const"], ls="--", lw=0.9),
    "sem_step":   dict(color=C["step"],  ls="-",  lw=0.9),
    "ldak_step":  dict(color=C["step"],  ls="--", lw=0.9),
    "falconer":   dict(color=C["zero"],  ls="-",  lw=0.9),
    "herg":       dict(color=C["zero"],  ls="--", lw=0.9),
}
LABEL = {"bigfam": "BIGFAM.v2", "bigfam_v1": "BIGFAM.v1", "sem_const": "SEM-const",
         "ldak_const": "QH/TH-const", "sem_step": "SEM-step", "ldak_step": "QH/TH-step",
         "falconer": "Falconer", "herg": "HE/PCGC"}
METHODS = ["bigfam", "bigfam_v1", "sem_const", "ldak_const", "sem_step",
           "ldak_step", "falconer", "herg"]
CUTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _r(sub, m, t):
    """Merged point estimate: Pearson r between method h2 and SNP-h2 on dist >= t."""
    x = sub[(sub.w_s_cal - 0.5).abs() >= t][["snp_h2", f"{m}_h2"]].dropna()
    return np.corrcoef(x.snp_h2, x[f"{m}_h2"])[0, 1] if len(x) >= 3 else np.nan


def main():
    d = load_sd1()
    # SNP-h2 benchmark: Neale's estimates are UK Biobank only, so this narrows
    # the 416 traits to the 340 UKB ones
    d = d[d.snp_h2.notna()].reset_index(drop=True)
    kinds = {"both": d, "continuous": d[d.kind == "continuous"],
             "binary": d[d.kind == "binary"]}
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.8), sharey=True)
    for ax, (kind, sub), tag in zip(axes, kinds.items(), "abc"):
        for m in METHODS:
            ys = [_r(sub, m, t) for t in CUTS]
            if np.all(np.isnan(ys)):                     # BIGFAM.v1 has no binary support
                continue
            ax.plot(CUTS, ys, **STY[m], zorder=3, alpha=.9, solid_capstyle="round")
        ax.axhline(0, color="0.75", lw=0.6, zorder=0)   # reference line, as in fig4
        ax.set_xlim(-.015, .315)
        ax.set_xticks([0, .1, .2, .3])
        ax.set_ylim(-.75, 1.0)
        ax.text(-.13, 1.04, tag, transform=ax.transAxes, fontsize=8, fontweight="bold")
        ax.text(.5, 1.04, kind, transform=ax.transAxes, ha="center", fontsize=7, color="0.5")
        sns.despine(ax=ax, trim=True, offset=4)
    axes[0].set_ylabel(r"Pearson $r$ with SNP-$h^2$")
    axes[1].set_xlabel(r"separability threshold  $t$")
    leg = [Line2D([], [], color=STY[m]["color"], ls=STY[m]["ls"], lw=STY[m]["lw"],
                  marker=STY[m].get("marker"), ms=STY[m].get("ms", 0), label=LABEL[m])
           for m in METHODS]
    fig.legend(handles=leg, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.06),
               columnspacing=0.9, handletextpad=0.4, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.8)
    save_png(fig, "figS5")
    plt.close(fig)


if __name__ == "__main__":
    main()
