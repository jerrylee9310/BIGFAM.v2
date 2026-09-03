"""plot_figS8.py -- Figure S8: Fig. 5c for all four traits, two engines x three
assumptions.

    .venv/bin/python analysis/fig5_figS8_cross_cohort/plot_figS8.py

Fig. 5c shows serum creatinine only, four methods. This spreads out all four
cross-cohort traits and all eight methods (adds BIGFAM.v1, HE and the LDAK
QuantHer rows). Circle = Generation Scotland, square = UK Biobank, the line
joining them is the cross-cohort difference. The vertical line is the UKB
SNP-h2 (Neale LDSC) -- a lower bound, so estimates are expected to its right.
Dashed = Neale primary phenotype ("high" confidence), dotted = not, i.e. a
softer baseline. The two engines (OpenMx SEM, LDAK) nearly coincide under the
same assumption.

Input: Supplementary Data 1 via compute.py. Output: figS8.png.
Style: _style.py (shared with the other analysis folders).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
import seaborn as sns                                              # noqa: E402

from _style import C as _C, W2, rc
from compute import engine_compare

HERE = Path(__file__).resolve().parent

rc()

C = {"decay": _C["decay"], "decayL": _C["decay_v1"],
     "AE": _C["zero"], "step": _C["step"], "const": _C["const"]}
ROWS = [("BIGFAM.v2", "decay", "BIGFAM.v2"), ("BIGFAM.v1", "decayL", "BIGFAM.v1"),
        ("SEM-const", "const", "SEM (const)"), ("LDAK-const", "const", "QH/TH (const)"),
        ("SEM-step", "step", "SEM (step)"), ("LDAK-step", "step", "QH/TH (step)"),
        ("Falconer", "AE", "Falconer (AE)"), ("HE", "AE", "HE (AE)")]
# Creat_mgdl is the same measurement as Creatinine (units only), so it serves as
# the same-cohort pair in Fig. 5b and is left out here
NAME = {"Total_cholesterol": "total cholesterol", "Creatinine": "creatinine",
        "avg_dia": "diastolic BP", "FVC": "FVC"}


def panel(ax, d):
    n = len(ROWS)
    for i, (key, cls, _) in enumerate(ROWS):
        y = n - 1 - i
        r = d.loc[key]
        ax.plot([r.GS, r.UKB], [y, y], color=C[cls], lw=1.0, zorder=2)
        ax.scatter([r.GS], [y], s=18, marker="o", facecolor="none",
                   edgecolor=C[cls], linewidth=0.9, zorder=3)
        ax.scatter([r.UKB], [y], s=18, marker="s", color=C[cls], zorder=3)

    ls = "--" if d.snp_conf.iloc[0] == "high" else ":"
    ax.axvline(d.snp_h2.iloc[0], color="0.35", lw=0.7, ls=ls, zorder=1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_xlim(0, 0.95)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_title(NAME.get(d.index.name, ""), fontsize=7, color="0.4", pad=4)


def main():
    d = engine_compare()
    traits = [t for t in dict.fromkeys(d.trait) if t in NAME]

    fig, axes = plt.subplots(1, len(traits), figsize=(W2, 2.3), sharey=True)
    for ax, t in zip(axes, traits):
        sub = d[d.trait.eq(t)].set_index("method")
        sub.index.name = t
        panel(ax, sub)
        ax.set_xlabel(r"$\hat{V}_A$")
        sns.despine(ax=ax, trim=True, offset=4, left=True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(ROWS))[::-1], labels=[r[2] for r in ROWS])

    leg = [Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", ms=4.5, label="GS"),
           Line2D([], [], marker="s", ls="", color="0.4", ms=4, label="UKB"),
           Line2D([], [], color="0.4", lw=0.9, ls="--", label=r"SNP-$h^2$ (UKB)")]
    fig.legend(handles=leg, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10),
               columnspacing=1.8, handletextpad=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=1.0)
    fig.savefig(HERE / "figS8.png")   # PNG only; rc savefig.bbox="tight"
    print("wrote figS8.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
