"""plot_fig2.py -- Figure 2: heritability estimate forest, 1x2 (slow, fast decay), scale-paired.

    .venv/bin/python plot_fig2.py

  Columns: left=slow decay (w_C=0.8), right=fast decay (w_C=0.2).
  Each method row: continuous (filled point) and binary (open point), offset
  slightly to pair them.
Point=mean, thick bar=50% (IQR), thin line=95%. Only BIGFAM.v2 sits near the
truth (0.5) in both regimes; decay-assuming (step/const) and AE only get
close under their own assumption, and are always biased upward. BIGFAM.v1
doesn't support binary phenotypes -> NA.

Color = model class (decay/step/const/AE), fill = scale (filled/open).
Left/right panels share the y axis (method). Figure size is NG 2-column
(183 mm), style from _style.py.
The full panel (all three w_C, including the degenerate point) is
plot_figS3.py. This underestimation bias is decomposed in the paper's figS2.

Input: results/raw.parquet (column names are the pipeline's own: w_s, V_G_hat).
Output: fig2.png + fig2.pdf.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from _style import C, MUT, FAINT, LW_HERO, LW_SEC, LW_REF, W2, rc, save

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw.parquet"
V_A_TRUE = 0.5

ROWS = [("bigfam", "decay", "decay", "BIGFAM.v2"), ("bigfam_v1", "decay", "decay_v1", "BIGFAM.v1"),
        ("sem", "step", "step", "SEM"), ("quanther", "step", "step", "QH/TH"),
        ("sem", "const", "const", "SEM"), ("quanther", "const", "const", "QH/TH"),
        ("falconer", "AE", "zero", "Falconer"), ("he", "AE", "zero", "HE/PCGC")]
HERO = {"bigfam", "bigfam_v1"}
NA_BINARY = {"bigfam_v1"}                         # binary unsupported -> NA
COLS = [(0.8, r"slow decay ($w_C = 0.8$)", "a"),  # left->right: slow -> fast
        (0.2, r"fast decay ($w_C = 0.2$)", "b")]
OFF = 0.20                                        # continuous(+) / binary(-) vertical offset

rc()


def pint(ax, y, s, col, filled=True):
    """point-interval: point=mean, thick bar=IQR(50%), thin line=95%."""
    s = np.asarray(s, float)
    lo95, lo, hi, hi95 = np.percentile(s, [2.5, 25, 75, 97.5])
    m = s.mean()
    a = 1.0 if filled else 0.5
    ax.plot([lo95, hi95], [y, y], color=col, lw=LW_SEC, alpha=0.75 * a,
            solid_capstyle="round", zorder=3)
    ax.plot([lo, hi], [y, y], color=col, lw=LW_HERO, alpha=a,
            solid_capstyle="round", zorder=4)
    if filled:
        ax.plot(m, y, "o", ms=4.6, mfc=col, mec="white", mew=0.9, zorder=5)
    else:
        ax.plot(m, y, "o", ms=4.6, mfc="white", mec=col, mew=1.1, zorder=5)


def main():
    raw = pd.read_parquet(RAW)
    n = len(ROWS)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 4.3), sharey=True, sharex=True)
    for ax, (ws, title, letter) in zip(axes, COLS):
        ax.axvline(V_A_TRUE, ls=(0, (4, 3)), lw=LW_REF, color="#333333", zorder=1)
        for i, (method, cond, klass, _) in enumerate(ROWS):
            y = n - 1 - i
            sc = raw[(raw.scale == "continuous") & (raw.w_s == ws) & (raw.method == method) & (raw.condition == cond)]["V_G_hat"]
            if len(sc):
                pint(ax, y + OFF, sc, C[klass], filled=True)
            if method in NA_BINARY:
                ax.text(0.62, y - OFF, "NA", ha="center", va="center", color=FAINT,
                        fontsize=6, fontstyle="italic")
                continue
            mb = "pcgc" if method == "he" else method
            sb = raw[(raw.scale == "binary") & (raw.w_s == ws) & (raw.method == mb) & (raw.condition == cond)]["V_G_hat"]
            if len(sb):
                pint(ax, y - OFF, sb, C[klass], filled=False)
        ax.set_title(title, loc="center", pad=6, fontsize=7)     # center-aligned panel title
        ax.text(0.0, 1.015, letter, transform=ax.transAxes, fontsize=8,    # a/b at panel's top-left
                fontweight="bold", va="bottom", ha="left")
        ax.set_xlabel("heritability estimate")
        ax.set_xlim(0.0, 1.5)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
        ax.set_ylim(-0.6, n - 0.4)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([lab for *_, lab in ROWS][::-1])
    axes[0].tick_params(axis="y", pad=10)                        # push y labels left -> reads as shared axis
    axes[0].text(V_A_TRUE / 1.5 + 0.012, 0.995, r"true $h^2$", transform=axes[0].transAxes,
                 fontsize=6, color="#333333", ha="left", va="top")
    for ax in axes:
        sns.despine(ax=ax, left=True, offset=5)
        ax.spines["bottom"].set_bounds(0, 1.5)
        ax.tick_params(left=False)
    cls = [Patch(fc=C["decay"], label="ACE (decay)"), Patch(fc=C["step"], label="ACE (step)"),
           Patch(fc=C["const"], label="ACE (const)"), Patch(fc=C["zero"], label="AE")]
    mk = [Line2D([], [], marker="o", ls="none", mfc=MUT, mec="white", mew=0.9, ms=5, label="continuous"),
          Line2D([], [], marker="o", ls="none", mfc="white", mec=MUT, mew=1.1, ms=5, label="binary")]
    fig.legend(handles=cls + mk, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3.0)            # spacing between a and b
    save(fig, "fig2", tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
