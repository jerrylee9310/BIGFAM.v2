"""plot_figS3.py -- Figure S3: heritability estimate forest, 2 rows (continuous, binary) x 3 cols (w_C: slow, degenerate, fast).

    .venv/bin/python plot_figS3.py

The full panel version of Fig. 2 (which only shows continuous, slow/fast) --
adds binary and the degenerate point.
Point=mean, thick bar=50% (IQR), thin line=95%.
  Rows: top=continuous, bottom=binary.
  Columns (left->right, w_C decreasing = decay speeding up): slow (0.8),
  degenerate (0.5), fast (0.2).
At the degenerate point (0.5), the genetic (0.5) and common-environment decay
rates coincide, so the model is non-identifiable -> every method is biased
upward. BIGFAM.v1 doesn't support binary -> NA. Colors = model class (same
legend as Fig. 2), BIGFAM markers are larger.

Input: results/raw.parquet (column names are the pipeline's own: w_s, V_G_hat).
Output: figS3.png + figS3.pdf. Style from _style.py.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from _style import C, FAINT, W2, rc, save

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw.parquet"
V_A_TRUE = 0.5

ROWS = [("bigfam", "decay", "decay", "BIGFAM.v2"),
        ("bigfam_v1", "decay", "decay_v1", "BIGFAM.v1"),
        ("sem", "step", "step", "SEM"),          # condition (step/const) shown via color+legend -> omitted from label
        ("quanther", "step", "step", "QH/TH"),
        ("sem", "const", "const", "SEM"),
        ("quanther", "const", "const", "QH/TH"),
        ("falconer", "AE", "zero", "Falconer"),
        ("he", "AE", "zero", "HE/PCGC")]
HERO = {"bigfam", "bigfam_v1"}
NA_BINARY = {"bigfam_v1"}
# columns (left->right): slow -> degenerate -> fast (w_C 0.8 -> 0.5 -> 0.2)
WS_COLS = [(0.8, "slow decay\n($w_C = 0.8$)"),
           (0.5, "degenerate\n($w_C = 0.5$)"),
           (0.2, "fast decay\n($w_C = 0.2$)")]
SCALES = ["continuous", "binary"]

rc()


def pint(ax, y, s, col, hero=False):
    """point-interval: point=mean, thick bar=IQR(50%), thin line=95%."""
    s = np.asarray(s, float)
    lo95, lo, hi, hi95 = np.percentile(s, [2.5, 25, 75, 97.5])
    ax.plot([lo95, hi95], [y, y], color=col, lw=0.9, solid_capstyle="round", zorder=3, alpha=0.85)
    ax.plot([lo, hi], [y, y], color=col, lw=3.2, solid_capstyle="round", zorder=4)
    ax.plot(s.mean(), y, "o", color=col, ms=5.4 if hero else 4.2, mec="white", mew=0.9, zorder=5)


def panel(ax, raw, ws, scale, letter, coltitle, xlabel, ylabel):
    n = len(ROWS)
    ax.axvline(V_A_TRUE, ls=(0, (4, 3)), lw=0.8, color="#333333", zorder=1)
    for i, (method, cond, klass, _) in enumerate(ROWS):
        y = n - 1 - i
        if scale == "binary" and method in NA_BINARY:
            ax.text(0.75, y, "NA", ha="center", va="center", color=FAINT,
                    fontsize=6, fontstyle="italic")
            continue
        m = "pcgc" if (scale == "binary" and method == "he") else method
        s = raw[(raw["scale"] == scale) & (raw["w_s"] == ws)
                & (raw["method"] == m) & (raw["condition"] == cond)]["V_G_hat"]
        if len(s) == 0:
            continue
        pint(ax, y, s, C[klass], hero=method in HERO)
    if coltitle:
        ax.set_title(coltitle, loc="center", pad=5, fontsize=7)
    if xlabel:
        ax.set_xlabel("heritability estimate")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, labelpad=30, fontweight="bold")
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")
    ax.set_xlim(0.0, 1.5)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax.set_ylim(-0.6, n - 0.4)


def main():
    raw = pd.read_parquet(RAW)
    n = len(ROWS)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.6), sharey=True, sharex=True)
    letters = "abcdef"
    for r, scale in enumerate(SCALES):
        for c, (ws, coltitle) in enumerate(WS_COLS):
            panel(axes[r, c], raw, ws, scale,
                  letter=letters[r * 3 + c],
                  coltitle=coltitle if r == 0 else None,
                  xlabel=(r == 1),
                  ylabel=scale if c == 0 else None)
    axes[0, 0].set_yticks(range(n))
    axes[0, 0].set_yticklabels([lab for *_, lab in ROWS][::-1])
    axes[0, 0].text(V_A_TRUE / 1.5 + 0.012, 0.995, r"true $h^2$", transform=axes[0, 0].transAxes,
                    fontsize=6, color="#333333", ha="left", va="top")
    for ax in axes.flat:
        sns.despine(ax=ax, left=True, offset=5)
        ax.spines["bottom"].set_bounds(0, 1.5)
        ax.tick_params(left=False)

    cls = [Patch(fc=C["decay"], label="ACE (decay)"), Patch(fc=C["step"], label="ACE (step)"),
           Patch(fc=C["const"], label="ACE (const)"), Patch(fc=C["zero"], label="AE")]
    fig.legend(handles=cls, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=1.5)
    save(fig, "figS3")
    plt.close(fig)
    print(f"(2x3: {SCALES} x w_s={[w for w, _ in WS_COLS]})")


if __name__ == "__main__":
    main()
