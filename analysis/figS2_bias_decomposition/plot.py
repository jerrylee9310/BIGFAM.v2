"""plot.py -- Figure S2: source of the Fig. 2 bias (decay-rate estimation vs. NNLS).

    .venv/bin/python analysis/figS2_bias_decomposition/plot.py

Under slow/fast decay, only "pipeline" is biased; the two true-w variants sit on
the true V_A and overlap almost exactly, showing the NNLS step contributes ~0 of
the bias. At the degenerate point (w_C=0.5) the design is exactly singular, so
true-w+GLS is undefined ("singular"), and even true-w+NNLS does not recover the
truth -- knowing the decay rate does not save this scenario.

Panels: (a) slow w_C=0.8, (b) degenerate w_C=0.5, (c) fast w_C=0.2.
Point=mean, thick bar=IQR(50%), thin line=95%, as in Fig. 2/S3.

Input: figS2.parquet (generate.py writes it). Output: figS2.png + figS2.pdf.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _style import C, FAINT, MUT, W2, rc, save

HERE = Path(__file__).resolve().parent
DATA = HERE / "figS2.parquet"
V_A_TRUE = 0.5

ROWS = [("pipeline", C["decay"], "BIGFAM.v2 (reported)"),
        ("truew_nnls", MUT, "true decay + NNLS"),
        ("truew_gls", MUT, "true decay + unconstrained GLS")]
COLS = [(0.8, r"slow decay ($w_C = 0.8$)", "a"),
        (0.5, r"degenerate ($w_C = 0.5$)", "b"),
        (0.2, r"fast decay ($w_C = 0.2$)", "c")]

rc()


def pint(ax, y, s, col, marker="o", filled=True):
    """point-interval: point=mean, thick bar=IQR(50%), thin line=95%. Matches fig2.py."""
    s = np.asarray(s, float)
    lo95, lo, hi, hi95 = np.percentile(s, [2.5, 25, 75, 97.5])
    m = s.mean()
    ax.plot([lo95, hi95], [y, y], color=col, lw=0.9, alpha=0.75, solid_capstyle="round", zorder=3)
    ax.plot([lo, hi], [y, y], color=col, lw=3.2, solid_capstyle="round", zorder=4)
    if filled:
        ax.plot(m, y, marker, ms=4.6, mfc=col, mec="white", mew=0.9, zorder=5)
    else:
        ax.plot(m, y, marker, ms=4.6, mfc="white", mec=col, mew=1.1, zorder=5)


def main():
    d = pd.read_parquet(DATA)
    n = len(ROWS)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6), sharey=True, sharex=True)
    for ax, (w, title, letter) in zip(axes, COLS):
        ax.axvline(V_A_TRUE, ls=(0, (4, 3)), lw=0.8, color="#333333", zorder=1)
        for i, (variant, col, _) in enumerate(ROWS):
            y = n - 1 - i
            s = d[(d.w_true == w) & (d.variant == variant)]["V_A_hat"].dropna()
            if len(s) == 0:
                ax.text(0.55, y, "singular", ha="center", va="center", color=FAINT,
                        fontsize=6, fontstyle="italic")
                continue
            marker = "s" if variant == "truew_gls" else "o"
            pint(ax, y, s, col, marker=marker, filled=(variant == "pipeline"))
        ax.set_title(title, loc="center", pad=6, fontsize=7)
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
        ax.set_xlabel("estimated $V_A$")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(-0.6, n - 0.4)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([lab for *_, lab in ROWS][::-1])
    axes[0].tick_params(axis="y", pad=10)
    axes[0].text(V_A_TRUE / 1.0 + 0.02, 0.995, r"true $V_A$", transform=axes[0].transAxes,
                 fontsize=6, color="#333333", ha="left", va="top")
    for ax in axes:
        sns.despine(ax=ax, left=True, offset=5)
        ax.spines["bottom"].set_bounds(0, 1.0)
        ax.tick_params(left=False)
    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=3.0)
    save(fig, "figS2")
    plt.close(fig)


if __name__ == "__main__":
    main()
