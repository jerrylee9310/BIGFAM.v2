"""Figure 4 -- agreement with SNP-h2 along the separability ladder.

    .venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_fig4.py

Input:  method_corr.csv (compute.py) and Supplementary Data 1
Output: fig4.png

Three panels sharing the separability threshold |w_C - 0.5| >= t, eight methods
colored by model class:

a  Pearson r      the decay methods hold up as t rises, the others fall
b  intercept a    only the decay methods stay near 0
c  slope b        the decay methods stay near the theoretical value, the others
                  break down at high separability

a is read from method_corr.csv; b and c refit OLS (estimated h2 ~ SNP-h2) on the
subset at each threshold.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
import seaborn as sns                                              # noqa: E402

from _sd1 import load_sd1, save_png
from _style import C, LW_HERO, LW_SEC, LW_REF, W2, rc

HERE = Path(__file__).resolve().parent

rc()

# same convention as figS5: color = model class, line style separates methods within one
STY = {
    "bigfam":     dict(color=C["decay"],    ls="-",  lw=LW_HERO, marker="o", ms=3.0, zorder=6, alpha=1.0),
    "bigfam_v1":  dict(color=C["decay_v1"], ls="-",  lw=LW_SEC, marker="s", ms=2.6, zorder=5, alpha=1.0),
    "sem_const":  dict(color=C["const"], ls="-",  lw=LW_SEC, zorder=3, alpha=0.85),
    "ldak_const": dict(color=C["const"], ls="--", lw=LW_SEC, zorder=3, alpha=0.85),
    "sem_step":   dict(color=C["step"],  ls="-",  lw=LW_SEC, zorder=3, alpha=0.85),
    "ldak_step":  dict(color=C["step"],  ls="--", lw=LW_SEC, zorder=3, alpha=0.85),
    "falconer":   dict(color=C["zero"],  ls="-",  lw=LW_SEC, zorder=3, alpha=0.85),
    "herg":       dict(color=C["zero"],  ls="--", lw=LW_SEC, zorder=3, alpha=0.85),
}
LABEL = {"bigfam": "BIGFAM.v2", "bigfam_v1": "BIGFAM.v1", "sem_const": "SEM-const",
         "ldak_const": "QH/TH-const", "sem_step": "SEM-step", "ldak_step": "QH/TH-step",
         "falconer": "Falconer", "herg": "HE/PCGC"}
METHODS = ["falconer", "herg", "sem_step", "ldak_step",
           "sem_const", "ldak_const", "bigfam_v1", "bigfam"]   # decay last = drawn on top
LEG_ORDER = ["bigfam", "bigfam_v1", "sem_const", "ldak_const", "sem_step",
             "ldak_step", "falconer", "herg"]
CUTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def ladder_r(ax, v):
    """Panel a -- the Pearson r ladder, trait types pooled."""
    q = v[(v.axis == "dist_cum") & (v.kind == "both")].copy()
    q["t"] = q.subset.str.lstrip(">=").astype(float)
    for m in METHODS:
        g = q[q.method_x == m].sort_values("t")
        if len(g):
            ax.plot(g.t, g.pearson, **STY[m])
    ax.axhline(0, color="0.75", lw=LW_REF, zorder=0)
    ax.set_ylabel(r"Pearson $r$ with $h^2_{\mathrm{snp}}$")
    ax.set_ylim(-.45, 1.0)


def _tab_ab(d):
    """OLS intercept and slope per threshold and method (BIGFAM.v1: continuous only)."""
    c = d.copy()
    c["dist"] = (c.w_s_cal - 0.5).abs()
    rows = []
    for t in CUTS:
        s = c[c.dist >= t]
        for m in METHODS:
            x = s[["snp_h2", f"{m}_h2"]].dropna()
            if len(x) < 3:
                a = b = np.nan
            else:
                b, a = np.polyfit(x.snp_h2, x[f"{m}_h2"], 1)
            rows.append((t, m, a, b))
    return pd.DataFrame(rows, columns=["t", "m", "alpha", "beta"])


def ladder_ab(ax, tab, key, ylim, ref=None, band=None, mark_below=False):
    if band is not None:
        ax.axhspan(*band, color=C["decay"], alpha=0.08, lw=0, zorder=0)
    for m in METHODS:
        g = tab[tab.m == m].sort_values("t")
        ax.plot(g.t, g[key], **STY[m])
        if mark_below:
            below = g[key] < ylim[0]
            ax.plot(g.loc[below, "t"], np.full(below.sum(), ylim[0] + 0.04), "v",
                    color=STY[m]["color"], ms=3.2, mew=0, zorder=7)
    if ref is not None:
        ax.axhline(ref, color="0.75", lw=LW_REF, zorder=1)
    ax.set_ylim(*ylim)


def main():
    d = load_sd1()
    # SNP-h2 benchmark: Neale's estimates are UK Biobank only, so this narrows
    # the 416 traits to the 340 UKB ones
    d = d[d.snp_h2.notna()].reset_index(drop=True)
    r = pd.read_csv(HERE / "method_corr.csv")
    v = r[r.method_y == "snp_h2"]
    tab = _tab_ab(d)

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(W2, 2.5))

    ladder_r(axa, v)

    ladder_ab(axb, tab, "alpha", (-.05, 1.20), ref=0.0)
    axb.set_ylabel(r"intercept  $\alpha$")

    ladder_ab(axc, tab, "beta", (-1.2, 2.0), band=(1.37, 1.72), mark_below=True)
    axc.set_ylabel(r"slope  $\beta$")

    for ax, tag in [(axa, "a"), (axb, "b"), (axc, "c")]:
        ax.set_xlim(-.015, .315)
        ax.set_xticks([0, .1, .2, .3])
        ax.text(-.20, 1.03, tag, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
        sns.despine(ax=ax, trim=True, offset=4)
    axb.set_xlabel(r"separability threshold  $t$")

    leg = [Line2D([], [], color=STY[m]["color"], ls=STY[m]["ls"], lw=STY[m]["lw"],
                  marker=STY[m].get("marker"), ms=STY[m].get("ms", 0), label=LABEL[m])
           for m in LEG_ORDER]
    fig.legend(handles=leg, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 0.995),
               columnspacing=0.9, handletextpad=0.4, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    save_png(fig, "fig4", tight=False)
    plt.close(fig)

    # panels b/c as numbers (the paper quotes the t = 0 and t = 0.30 ends)
    print("OLS estimated h2 ~ SNP-h2 (both), intercept alpha / slope beta by threshold t")
    print(f"{'method':12s}" + "".join(f"{t:>14.2f}" for t in CUTS))
    for m in LEG_ORDER:
        g = tab[tab.m == m].sort_values("t")
        print(f"{LABEL[m]:12s}" + "".join(f"{a:+7.3f}/{b:+6.3f}" for a, b in zip(g.alpha, g.beta)))


if __name__ == "__main__":
    main()
