"""plot.py -- renders figS7: V_A recovery (r, alpha, beta) vs. separability
threshold, sorted by estimated w_C vs. true w_C, on GRID and NULL.

    .venv/bin/python plot.py [--small]

3x3 grid -- rows = sort axis, columns = metric.
  a b c  GRID, sorted by w_hat_C   the axis the paper actually uses
  d e f  GRID, sorted by true w_C  same ladder on the oracle axis -- if both
                                   rows agree, sorting by the estimate is not
                                   circular
  g h i  NULL, sorted by w_hat_C   a world with no true separability at all;
                                   flat here means the threshold itself isn't
                                   manufacturing an effect

r = Pearson correlation with true V_A. alpha, beta = intercept/slope of
OLS(V_A_hat ~ true V_A) on the threshold's subset. Unlike Fig. 4 (x-axis is a
lower bound, SNP-h2), here the x-axis is the truth itself, so unbiased means
(alpha=0, beta=1), not beta>1. n<50 thresholds are dropped from a row.

Input:  results/circular_pairlevel_scen[_small].parquet  (generate.py)
Output: figS7.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from _style import C, FAINT, MUT, W2, panel_label, rc, save

HERE = Path(__file__).resolve().parent

rc()

STY = {
    "bigfam":    dict(color=C["decay"], lw=1.6, marker="o", ms=3.0, zorder=6),
    "sem_const": dict(color=C["const"], lw=0.9, marker="^", ms=2.2, zorder=3),
    "sem_step":  dict(color=C["step"],  lw=0.9, marker="D", ms=1.9, zorder=3),
    "falconer":  dict(color=C["zero"],  lw=0.9, marker="o", ms=2.2, zorder=3),
    "he":        dict(color=C["zero"],  lw=0.9, marker="s", ms=2.2, zorder=3),
}
LABEL = {"bigfam": "BIGFAM.v2", "sem_const": "SEM-const", "sem_step": "SEM-step",
         "falconer": "Falconer", "he": "HE/PCGC"}
METHODS = ["falconer", "he", "sem_step", "sem_const", "bigfam"]
BASE = [m for m in METHODS if m != "bigfam"]
CUTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ROWS = [("grid_est",  "grid", "w_hat",  r"GRID  $\cdot$  sorted by $\hat w_C$"),
        ("grid_true", "grid", "w_true", r"GRID  $\cdot$  sorted by true $w_C$"),
        ("null_est",  "null", "w_hat",  r"NULL (true $w_C = 0.5$)  $\cdot$  sorted by $\hat w_C$")]
COLS = [("r", r"Pearson $r$ with true $V_A$", None),
        ("alpha", r"intercept  $\alpha$", 0.0),
        ("beta", r"slope  $\beta$", 1.0)]


def table(d):
    rows = []
    for key, design, axis, _ in ROWS:
        g = d[d.design == design]
        dist = (g[axis] - 0.5).abs()
        for t in CUTS:
            s = g[dist >= t]
            if len(s) < 50:
                continue
            for m in METHODS:
                x = s[["V_G_true", f"VA_{m}"]].dropna()
                if len(x) < 2 or x[f"VA_{m}"].std() < 1e-12:
                    continue
                b, a = np.polyfit(x.V_G_true, x[f"VA_{m}"], 1)
                rows.append(dict(row=key, m=m, t=t, n=len(s),
                                 r=x[f"VA_{m}"].corr(x.V_G_true), alpha=a, beta=b))
    return pd.DataFrame(rows)


def annotate_null(ax, tab):
    s = tab[(tab.row == "null_est") & (tab.m == "bigfam")].sort_values("t")
    if s.empty:
        return
    ax.axvspan(0.225, 0.315, color=FAINT, alpha=0.13, lw=0, zorder=0)
    ax.text(0.27, 0.115, "$n<100$", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=5, color=MUT)
    ax.text(-0.012, 0.03, "$n$", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=5, color=MUT, style="italic")
    for t, n in zip(s.t, s.n):
        ax.text(t, 0.03, f"{n:,}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5, color=MUT)


def report(tab):
    def val(row, m, t, k):
        v = tab[(tab.row == row) & (tab.m == m) & (tab.t == t)][k]
        return v.iloc[0] if len(v) else float("nan")

    print("\nalpha/beta reference: x-axis is true V_A itself, so unbiased = (alpha=0, beta=1).")
    print("  (Fig. 4 uses a lower bound (SNP-h2) as x-axis, so beta>1 there is not analogous.)\n")

    for row, _, _, lab in ROWS:
        sub = tab[tab.row == row]
        if sub.empty:
            print(f"[{row}]  {lab}   -- no thresholds cleared n>=50, skipped\n")
            continue
        hi = sub.t.max()
        print(f"[{row}]  {lab}   (t=0.00 -> t={hi:.2f})")
        print(f"  {'method':11s} {'r':>15s} {'alpha':>15s} {'beta':>15s}")
        for m in reversed(METHODS):
            cells = "".join(f"{val(row, m, 0.0, k):7.3f} ->{val(row, m, hi, k):7.3f}"
                            for k in ("r", "alpha", "beta"))
            print(f"  {LABEL[m]:11s} {cells}")
        n0 = val(row, "bigfam", 0.0, "n")
        nh = val(row, "bigfam", hi, "n")
        print(f"  n: {n0:,.0f} -> {nh:,.0f}" if not (np.isnan(n0) or np.isnan(nh)) else "  n: n/a")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="read the _small parquet instead")
    args = ap.parse_args()
    tag = "_small" if args.small else ""

    d = pd.read_parquet(HERE / "results" / f"circular_pairlevel_scen{tag}.parquet")
    assert (d.design == "null").any(), "no design=='null' rows -- check generate.py output"
    tab = table(d)

    fig, axes = plt.subplots(3, 3, figsize=(W2, 5.4), sharex=True, sharey="col")
    for i, (key, _, _, rlab) in enumerate(ROWS):
        for j, (col, ylab, ref) in enumerate(COLS):
            ax = axes[i, j]
            if ref is not None:
                ax.axhline(ref, color="0.75", lw=0.6, zorder=1)
            for m in METHODS:
                g = tab[(tab.row == key) & (tab.m == m)].sort_values("t")
                if not g.empty:
                    ax.plot(g.t, g[col], ls="-", **STY[m])
            ax.tick_params(labelleft=True)
            if i == 1:
                ax.set_ylabel(ylab)
            if j == 0:
                ax.set_title(rlab, fontsize=7, loc="left", pad=9)
            panel_label(ax, "abcdefghi"[i * 3 + j])
            if i == 2 and j:
                ax.axvspan(0.225, 0.315, color=FAINT, alpha=0.13, lw=0, zorder=0)
    annotate_null(axes[2, 0], tab)

    for ax in axes.flat:
        ax.set_xlim(-.015, .315)
        ax.set_xticks([0, .1, .2, .3])
        sns.despine(ax=ax, trim=True, offset=4)
    axes[2, 1].set_xlabel(r"separability threshold  $t$")

    leg = [Line2D([], [], color=STY[m]["color"], lw=STY[m]["lw"], marker=STY[m]["marker"],
                  ms=STY[m]["ms"], label=LABEL[m]) for m in reversed(METHODS)]
    fig.legend(handles=leg, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02),
               columnspacing=0.9, handletextpad=0.4, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.2, h_pad=1.8)
    save(fig, "figS7")
    plt.close(fig)
    report(tab)


if __name__ == "__main__":
    main()
