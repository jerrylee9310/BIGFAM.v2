"""Figure S4 -- the nine added-variable plots behind Fig. 3b.

    .venv/bin/python analysis/fig3_figS4_disagreement/plot_figS4.py

Input:  supple_data/Supplementary_Data_1_trait_estimates.csv (via plot_fig3)
Output: figS4.png

Rows = trait set (all 416, UK Biobank 340, Generation Scotland 76), columns =
predictor (separability, V_C, V_A). Each panel plots rank residuals after
controlling for the other two predictors and the mean SE, standardized so the
slope is the adjusted rho printed in the corner with its 95% CI.

All computation is imported from plot_fig3.py. The bootstrap consumes the rng
in a fixed order (all -> UKB -> GS, COV order within each), which is what makes
the nine CIs match plot_fig3.py's output exactly.
"""
from __future__ import annotations

import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import seaborn as sns                                              # noqa: E402
from scipy.stats import rankdata                                   # noqa: E402

import plot_fig3 as fig3                                           # noqa: E402
from _style import C, FAINT, INK, MUT, W2, panel_label, rc          # noqa: E402

rc()
# (key, label); "all" is every trait, the others are dataset values
SETS = [("all", "All traits"), ("UKB41907", "UK Biobank"),
        ("GS23471", "Generation Scotland")]
SHORT = {"ident": "separability", "V_S": fig3.COV_LAB["V_S"], "V_G": fig3.COV_LAB["V_G"]}
COL = {"ident": C["decay"], "V_S": INK, "V_G": FAINT}   # same predictor colors as fig3 b


def stats_for(sub, rng):
    """Same spec as fig3 b: controls are the other two predictors plus mean SE.

    Returns (adjusted rho, CI, standardized residual pair).
    """
    V = {k: sub[k].to_numpy() for k in fig3.COV}
    sd, mse = sub.sd.to_numpy(), sub.mse.to_numpy()
    out = {}
    for k in fig3.COV:
        cond = [V[o] for o in fig3.COV if o != k] + [mse]
        adj = fig3.partial_spearman(V[k], sd, *cond)
        ci = fig3.boot_ci(V[k], sd, rng, cond)
        # standardize both residuals so the fitted slope is the adjusted rho
        R = tuple(v / v.std(ddof=1)
                  for v in (fig3.resid(rankdata(V[k]), cond), fig3.resid(rankdata(sd), cond)))
        out[k] = (adj, ci, R)
    return out


def panel_av(ax, rx, ry, col, xlab, adj, ci, lim, ylab=False):
    """One panel: standardized rank residuals, so the slope is the adjusted rho.

    All nine panels share xlim/ylim -- different axes would distort the slope
    comparison. The y label appears only in the first column.
    """
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.scatter(rx, ry, s=5, c=col, alpha=0.4, lw=0, zorder=2)
    xx = np.array([rx.min(), rx.max()])
    ax.plot(xx, np.polyval(np.polyfit(rx, ry, 1), xx), color=col, lw=1.8, zorder=4)
    txt = (rf"partial $\rho$ = {adj:+.3f}" + f"\n[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    ax.text(0.03, 0.97, txt.replace("-", "\u2212"),   # typographic minus, as elsewhere
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, linespacing=1.35)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[-3, 0, 3], yticks=[-3, 0, 3])
    ax.set_xticklabels(["\u22123", "0", "3"])
    ax.set_yticklabels(["\u22123", "0", "3"])
    ax.set_xlabel(xlab, fontsize=7)
    if ylab:
        ax.set_ylabel(fig3.SDFIX)
    sns.despine(ax=ax, trim=True, offset=4)           # spines cut at the ticks, as in fig3


def main():
    d, _ = fig3.load()
    subs = {"all": d}
    subs.update({s: d[d.dataset == s].reset_index(drop=True) for s, _ in SETS[1:]})
    assert sum(len(subs[s]) for s, _ in SETS[1:]) == len(d), "cohort split does not cover all 416 traits"

    rng = np.random.default_rng(fig3.SEED)           # CIs depend on draw order, so it is fixed
    S = {s: stats_for(sub, rng) for s, sub in subs.items()}

    # shared limits set from the 99.5th percentile, not the max: one outlier would
    # stretch all nine axes and flatten the clouds. Points outside are counted on
    # stdout; the fits always use every trait.
    allr = np.concatenate([v for s, _ in SETS for k in fig3.COV for v in S[s][k][2]])
    lim = 1.06 * np.quantile(np.abs(allr), 0.995)

    fig = plt.figure(figsize=(W2, 6.4))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.3)
    for i, (s, lab) in enumerate(SETS):
        for j, k in enumerate(fig3.COV):
            ax = fig.add_subplot(gs[i, j])
            adj, ci, R = S[s][k]
            panel_av(ax, *R, COL[k], SHORT[k], adj, ci, lim, ylab=(j == 0))
            panel_label(ax, "abcdefghi"[i * 3 + j], dx=-0.30 if j == 0 else -0.14, dy=0.14)
            if j == 1:                               # row title once, above the middle panel
                ax.text(0.5, 1.14, f"{lab}  (n = {len(subs[s])})", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7, color=MUT)
    fig3.save_png(fig, "figS4")
    plt.close(fig)

    print("partial rank correlation with SD_fix  (adjusted for the other two + mean SE, "
          f"95% CI from {fig3.PBOOT} trait-resample bootstraps, seed {fig3.SEED})")
    print(f"  {'set':22s} {'n':>4s}  " + "  ".join(f"{k:^22s}" for k in fig3.COV))
    for s, lab in SETS:
        print(f"  {lab:22s} {len(subs[s]):4d}  " + "  ".join(
            f"{S[s][k][0]:+.3f} [{S[s][k][1][0]:+.3f},{S[s][k][1][1]:+.3f}]" for k in fig3.COV))
    out = sum(int((np.abs(v) > lim).sum()) for s, _ in SETS for k in fig3.COV
              for v in S[s][k][2])
    print(f"  panels a-i share one axis (lim {lim:.2f} SD): {out} of {len(allr)} residual "
          f"coordinates fall outside; fits use every trait")


if __name__ == "__main__":
    main()
