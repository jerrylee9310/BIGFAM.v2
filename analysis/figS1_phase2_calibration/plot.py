"""figS1 plot — Figure S1: Phase 2 w_hat_C calibration and separability-driven shrinkage.

    .venv/bin/python analysis/figS1_phase2_calibration/plot.py

  a: w_hat_C tracks the true w_C but is pulled toward 0.5 at both ends (shrinkage).
     Band = central 95%, dashed line = identity.
  b: retention "safe-zone" map -- median retention over the two axes that drive
     separability (signal vs. measurement noise). High signal / low noise (lower
     right) -> r -> 1 (tracks truth); the opposite corner (upper left) -> r -> 0
     (collapses to 0.5).
     x: signal = V_C * (w_C-0.5)^2  (all-rho w-signal, det J = 0.5 * V_C * (w-0.5)^2)
     y: noise  = Max(sqrt(Sigma)) = max_d sqrt(Sigma_dd) (largest per-DOR
        correlation measurement sd)
     color: retention r = (w_hat-0.5)/(w_true-0.5)  (1 = signal fully preserved,
        0 = collapsed to 0.5)

  Column names (w_S_true / V_S / w_hat) match generate.py's CSV schema; only the
  rendered axis labels are unified to w_C/V_C.

Input: figS1.csv (generate.py). Output: figS1.png.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _style import C, W2, rc, save

HERE = Path(__file__).resolve().parent
CSV = HERE / "figS1.csv"

BLUE = C["const"]        # model-class-neutral figure -- unified neutral blue (_style.py)

EDGES = np.linspace(0.01, 0.99, 21)
MID = 0.5 * (EDGES[:-1] + EDGES[1:])

rc()


def curve(wt, wh, mask=None, q=None, nmin=20):
    """Binned mean (or quantile) of w_hat vs true w, optionally on a subset."""
    if mask is not None:
        wt, wh = wt[mask], wh[mask]
    idx = np.clip(np.digitize(wt, EDGES) - 1, 0, len(MID) - 1)
    out = []
    for k in range(len(MID)):
        sel = wh[idx == k]
        out.append(np.nan if sel.size < nmin
                   else (np.quantile(sel, q) if q is not None else sel.mean()))
    return np.array(out)


def frame(ax, letter, xlabel, ylabel):
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.8, color="#333333", label="_nolegend_", zorder=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.text(0.0, 1.015, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def retention_grid(signal, noise, r, xe, ye, nmin=25):
    """Median retention per (signal, noise) cell; cells with <nmin replicates -> NaN."""
    ix = np.clip(np.digitize(np.clip(signal, xe[0], xe[-1]), xe) - 1, 0, len(xe) - 2)
    iy = np.clip(np.digitize(np.clip(noise, ye[0], ye[-1]), ye) - 1, 0, len(ye) - 2)
    Z = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    for i in range(len(ye) - 1):
        for j in range(len(xe) - 1):
            sel = (ix == j) & (iy == i)
            if sel.sum() >= nmin:
                Z[i, j] = np.median(r[sel])
    return Z


def main():
    if not CSV.exists():
        raise FileNotFoundError(
            f"{CSV} not found (it is not committed -- 19MB). Run "
            "`python analysis/figS1_phase2_calibration/generate.py` first (~5 min).")
    d = pd.read_csv(CSV)
    wt, wh = d.w_S_true.values, d.w_hat.values

    fig, ax = plt.subplots(1, 2, figsize=(W2, 3.0))

    # a: overall shrinkage -- central 95% band + binned-mean line (line kept, off legend)
    frame(ax[0], "a", "truth $(w_C)$", "estimated $(\\hat{w}_{C,\\mathrm{ridge}})$")
    ax[0].fill_between(MID, curve(wt, wh, q=0.025), curve(wt, wh, q=0.975),
                       alpha=0.25, color=BLUE, lw=0, label="95% percentile interval")
    ax[0].plot(MID, curve(wt, wh), "o-", color=BLUE, ms=3, lw=1.3,
               mec="white", mew=0.5, label="_nolegend_")
    ax[0].legend(loc="upper left")
    sns.despine(ax=ax[0], trim=True, offset=5)

    # b: retention "safe-zone" map over the two separability drivers
    signal = d.V_S.values * (wt - 0.5) ** 2        # x: all-rho w-signal
    noise = d.sigma_max.values                      # y: measurement sd
    r = (wh - 0.5) / (wt - 0.5)                      # retention per replicate
    xe = np.logspace(np.log10(1e-4), np.log10(2.5e-1), 15)   # 14 signal bins (coarse)
    ye = np.linspace(0.001, 0.10, 13)                        # 12 noise bins
    Z = retention_grid(signal, noise, r, xe, ye, nmin=12)
    # drop the two sparse edges: bottom noise row (sigma -> 0.001) and rightmost signal column
    Z, xe, ye = Z[1:, :-1], xe[:-1], ye[1:]
    print(f"  masked cells (shown): {np.isnan(Z).sum()}/{Z.size}")

    # diverging RdBu: blue = high retention (track), red = low (collapse), white = 0.5.
    pcm = ax[1].pcolormesh(xe, ye, Z, cmap="RdBu", vmin=0.0, vmax=1.0, shading="flat")
    ax[1].set_xscale("log")
    ax[1].set(xlim=(xe[0], xe[-1]), ylim=(ye[0], ye[-1]),
              xlabel="signal  $V_C(w_C-0.5)^2$", ylabel="measurement noise  $\\mathrm{Max}(\\sqrt{\\Sigma})$")
    ax[1].text(0.0, 1.015, "b", transform=ax[1].transAxes, fontsize=8,
               fontweight="bold", va="bottom", ha="left")
    cb = fig.colorbar(pcm, ax=ax[1], pad=0.03, fraction=0.046, aspect=22)
    cb.set_label("retention  $r$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    sns.despine(ax=ax[1])

    fig.tight_layout(w_pad=2.0)
    save(fig, "figS1")
    plt.close(fig)
    print(f"({len(d)} replicates; b = median-retention over signal x noise)")


if __name__ == "__main__":
    main()
