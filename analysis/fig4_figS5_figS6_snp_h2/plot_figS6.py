"""Figure S6 -- at high separability, does the regression line rotate or shift?

    .venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_figS6.py

Input:  supple_data/Supplementary_Data_1_trait_estimates.csv
Output: figS6.png + figS6.pdf

One panel per method, each holding two regression lines: the low-separability
subset (dist < 0.20, grey dashed) and the high one (dist >= 0.20, solid, class
color). Black bars mark the vertical gap between them at the 10th and 90th
percentile of SNP-h2, and those two numbers order the panels.

The gap is measured vertically rather than at the crossing point, whose
denominator is the slope difference and so diverges when rotation is small. Gap
size grows with the gate, but the ordering of methods is stable over gates
0.10-0.25; the full table is printed on stdout.

The y = x dotted line is a reference only -- the expectation here is a = 0 with
b > 1, not b = 1. The rug under each panel is the SNP-h2 distribution, showing
where the two evaluation points sit.
"""
from __future__ import annotations

import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
import numpy as np                                                 # noqa: E402
import seaborn as sns                                              # noqa: E402

from _sd1 import load_sd1, save_png
from _style import C as _C, W2, rc

rc()
C = {"decay": _C["decay"], "AE": _C["zero"], "step": _C["step"], "const": _C["const"]}
LOW, GAP = "#999999", "#222222"              # low-separability line, gap bar
CLS = {"bigfam": "decay", "bigfam_v1": "decay",
       "sem_const": "const", "ldak_const": "const",
       "herg": "AE", "falconer": "AE",
       "ldak_step": "step", "sem_step": "step"}
# BIGFAM.v1 gets the decay tint, as in fig2 and fig3
MCOL = {"bigfam_v1": _C["decay_v1"]}
LABEL = {"falconer": "Falconer", "herg": "HE/PCGC",
         "sem_step": "SEM-step", "sem_const": "SEM-const",
         "ldak_step": "QH/TH-step", "ldak_const": "QH/TH-const",
         "bigfam": "BIGFAM.v2", "bigfam_v1": "BIGFAM.v1"}
GATE, YMAX = 0.20, 1.10
QLO, QHI = 0.10, 0.90                        # SNP-h2 quantiles where the gap is measured


def _ab(s, m):
    """OLS intercept and slope for method m on subset s."""
    x = s[["snp_h2", f"{m}_h2"]].dropna()
    b, a = np.polyfit(x.snp_h2, x[f"{m}_h2"], 1)
    return float(a), float(b)


def main():
    d = load_sd1()
    # SNP-h2 benchmark: Neale's estimates are UK Biobank only, so this narrows
    # the 416 traits to the 340 UKB ones
    d = d[d.snp_h2.notna()].reset_index(drop=True)
    d["dist"] = (d.w_s_cal - 0.5).abs()
    lo_d, hi_d = d[d.dist < GATE], d[d.dist >= GATE]
    xlo, xhi = float(d.snp_h2.quantile(QLO)), float(d.snp_h2.quantile(QHI))

    fit = {m: (_ab(lo_d, m), _ab(hi_d, m)) for m in CLS}
    shift = {m: [(ah + bh * x) - (al + bl * x) for x in (xlo, xhi)]
             for m, ((al, bl), (ah, bh)) in fit.items()}
    order = sorted(CLS, key=lambda m: shift[m][0] - shift[m][1])   # by rotation, ascending

    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.9), sharex=True, sharey=True)
    n_over = 0
    for ax, m, tag in zip(axes.ravel(), order, "abcdefgh"):
        cc = MCOL.get(m, C[CLS[m]])
        pts = d[["snp_h2", f"{m}_h2"]].dropna()
        n_over += int((pts[f"{m}_h2"] > YMAX).sum())
        ax.scatter(pts.snp_h2, pts[f"{m}_h2"], s=4, color="0.55", alpha=.20,
                   linewidths=0, zorder=2)                          # raw traits, faded
        ax.plot([0, YMAX], [0, YMAX], ls=":", lw=.7, color="0.75", zorder=1)
        ax.plot(d.snp_h2, np.full(len(d), -0.055 * YMAX), "|", color="0.55",
                ms=3.0, mew=.35, alpha=.7, clip_on=False, zorder=3)  # SNP-h2 rug

        (al, bl), (ah, bh) = fit[m]
        xs = np.array([0, .50])
        ax.plot(xs, al + bl * xs, ls="--", lw=1.3, color=LOW, zorder=4)
        ax.plot(xs, ah + bh * xs, ls="-", lw=1.3, color=cc, zorder=5)
        for x in (xlo, xhi):                                        # vertical gap between the lines
            ax.plot([x, x], [al + bl * x, ah + bh * x], "-", color=GAP, lw=1.6,
                    solid_capstyle="butt", zorder=6)
            ax.plot([x], [ah + bh * x], "_", color=GAP, ms=4, mew=1.4, zorder=6)

        ax.set_xlim(0, .50)
        ax.set_ylim(0, YMAX)
        ax.text(.04, .96, f"{tag} {LABEL[m]}", transform=ax.transAxes, va="top",
                fontsize=8, fontweight="bold")
        ax.text(.04, .83, CLS[m], transform=ax.transAxes, va="top", fontsize=6, color=cc)
        ax.text(.97, .05, f"shift {shift[m][0]:+.3f} $\\rightarrow$ {shift[m][1]:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
        sns.despine(ax=ax, trim=False, offset=6)

    for ax in axes[1]:
        ax.set_xlabel("SNP-$h^2$")
    for ax in axes[:, 0]:
        ax.set_ylabel("estimated $h^2$")
    leg = [Line2D([], [], color=LOW, ls="--", lw=1.3, label=f"low separability (< {GATE})"),
           Line2D([], [], color="0.3", ls="-", lw=1.3, label=f"high (≥ {GATE}, class color)"),
           Line2D([], [], color=GAP, ls="-", lw=1.6,
                  label=f"gap at SNP-$h^2$ = {xlo:.3f} and {xhi:.3f}"),
           Line2D([], [], color="0.55", ls="none", marker="|", ms=4, label="SNP-$h^2$")]
    fig.legend(handles=leg, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.10),
               columnspacing=1.6, handletextpad=0.6)
    fig.suptitle(f"regression by separability gate ($|\\hat{{w}}_C-0.5|={GATE}$)  ·  both  ·  "
                 f"sorted by rotation  ·  {n_over} off-scale", y=1.01, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png(fig, "figS6")
    plt.close(fig)

    print(f"wrote figS6.png   gate={GATE}  n low={len(lo_d)}  n high={len(hi_d)}")
    print(f"gap evaluated at SNP-h2 {QLO:.0%}={xlo:.3f} and {QHI:.0%}={xhi:.3f}")
    print(f"{'method':12s}{'low a':>8s}{'low b':>8s}{'high a':>8s}{'high b':>8s}"
          f"{'gap lo':>9s}{'gap hi':>9s}{'rotation':>10s}")
    for m in order:
        (al, bl), (ah, bh) = fit[m]
        s0, s1 = shift[m]
        print(f"{LABEL[m]:12s}{al:+8.3f}{bl:+8.3f}{ah:+8.3f}{bh:+8.3f}"
              f"{s0:+9.3f}{s1:+9.3f}{s0 - s1:+10.3f}")
    print("gate sensitivity of the rotation (gap lo - gap hi)")
    for g in (0.10, 0.15, 0.20, 0.25):
        lo_g, hi_g = d[d.dist < g], d[d.dist >= g]
        row = []
        for m in order:
            (al, bl), (ah, bh) = _ab(lo_g, m), _ab(hi_g, m)
            row.append((ah + bh * xlo - al - bl * xlo) - (ah + bh * xhi - al - bl * xhi))
        print(f"  gate {g:.2f} (n high={len(hi_g):3d})  "
              + "  ".join(f"{LABEL[m][:9]} {v:+.3f}" for m, v in zip(order, row)))


if __name__ == "__main__":
    main()
