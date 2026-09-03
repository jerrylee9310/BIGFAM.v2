"""plot_fig5.py -- Figure 5: what moves and what stays when the relative
composition changes.

    .venv/bin/python analysis/fig5_figS8_cross_cohort/plot_fig5.py

a  Changing only whom the questionnaire item asks about splits w_hat_C
   (GS family-history traits; the relative pairs are the same for all items).
b  BIGFAM.v2 only -- how far V_hat_A moves depending on how estimates are
   paired: same measurement re-expressed within one cohort / same trait in
   two cohorts / the same estimates mis-paired. All three categories come
   from the same four traits, so the only thing that differs is the pairing.
c  One trait (serum creatinine) estimated in both cohorts, by method. Line
   length is the cross-cohort difference; the dotted vertical line is the
   UKB SNP-h2 (Neale LDSC), a lower bound.

Panel a plots 4 parent items and 13 grandparent items; the one self item is
excluded from the test and not drawn. P is a one-sided Mann-Whitney test
(grandparent > parent, the inequality fixed by the pedigree before seeing the
data), recomputed here. Panel c shows one row per assumption class plus
BIGFAM.v2; HE lies on the same line as Falconer and is left to Fig. S8.

Input: Supplementary Data 1 via compute.py. Output: fig5.png.
Style: _style.py (shared with the other analysis folders).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl                                           # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
import numpy as np                                                 # noqa: E402
import seaborn as sns                                              # noqa: E402
from scipy.stats import mannwhitneyu                               # noqa: E402

from _style import C as _C, LW_SEC, LW_REF, W2, rc
from compute import famhist, vg_matched, engine_compare

HERE = Path(__file__).resolve().parent

rc()

C = {"decay": _C["decay"], "AE": _C["zero"], "step": _C["step"], "const": _C["const"]}
# panel c rows, top to bottom. The full set (QH/TH, BIGFAM.v1, HE) is Fig. S8.
METHODS = [("BIGFAM.v2", "decay", "BIGFAM.v2"), ("Falconer", "AE", "Falconer (AE)"),
           ("SEM-const", "const", "SEM (const)"), ("SEM-step", "step", "SEM (step)")]
RNG = np.random.default_rng(0)


def informants(ax):
    """a -- w_hat_C by whom the item asks about. Relative pairs fixed (GS)."""
    d = famhist()
    d["grp"] = np.where(d.informant.isin(["father", "mother"]), "parent", d.informant)
    d = d[d.grp.isin(["parent", "grandparent"])].copy()
    X = {"parent": 0, "grandparent": 1}
    d["x"] = d.grp.map(X) + RNG.uniform(-0.11, 0.11, len(d))

    for grp, x in X.items():
        g = d[d.grp == grp]
        ax.scatter(g.x, g.w_s_cal, s=15, color=C["decay"], alpha=0.85,
                   linewidth=0, zorder=3)
        ax.plot([x - 0.26, x + 0.26], [g.w_s_cal.median()] * 2,
                color="k", lw=LW_SEC, zorder=4)

    # one-sided test of the pedigree-predicted inequality (grandparent > parent);
    # a single exact P, no stars
    p = mannwhitneyu(d[d.grp == "grandparent"].w_s_cal,
                     d[d.grp == "parent"].w_s_cal, alternative="greater").pvalue
    m, e = f"{p:.1e}".split("e")
    top = d.w_s_cal.max() + 0.055
    ax.plot([0, 0, 1, 1], [top, top + .025, top + .025, top], color="0.3", lw=LW_REF)
    ax.text(0.5, top + .045, rf"$P$ = {m} $\times$ 10$^{{{int(e)}}}$", fontsize=7,
            color="0.2", ha="center", va="bottom")

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(list(X.values()))
    ax.set_ylabel(r"estimated decay rate  ($\hat{w}_C$)")
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def layers(ax):
    """b -- BIGFAM.v2 only. All three categories come from the same 4 traits."""
    m = vg_matched()
    for x, (L, col) in enumerate([("a", "0.45"), ("b", C["decay"]), ("c", "0.72")]):
        v = m[m.layer.eq(L)].dVG.to_numpy()
        ax.scatter(x + RNG.uniform(-0.15, 0.15, len(v)), v, s=15, color=col,
                   alpha=0.85, linewidth=0, zorder=3)
        med = np.median(v)
        ax.plot([x - 0.3, x + 0.3], [med] * 2, color="k", lw=LW_SEC, zorder=4)
        ax.text(x + 0.34, med, f"{med:.3f}", fontsize=6.5, color="0.2",
                va="center", ha="left")
        ax.annotate(rf"$n$ = {len(v)}", (x, 1.02), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color="0.5", annotation_clip=False)

    ax.set_xlim(-0.5, 2.6)
    ax.set_xticks(range(3))
    ax.set_ylabel(r"difference in $\hat{V}_A$  ($|\Delta \hat{V}_A|$)")
    ax.set_ylim(0, 0.32)


def creatinine(ax):
    """c -- serum creatinine in both cohorts. circle=GS, square=UKB, line=|dV_A|."""
    d = engine_compare()
    d = d[d.trait.eq("Creatinine")].set_index("method")
    snp_h2 = float(d.snp_h2.iloc[0])

    order = METHODS
    n = len(order)
    for i, (key, cls, _) in enumerate(order):
        y = n - 1 - i
        gs, uk = float(d.loc[key, "GS"]), float(d.loc[key, "UKB"])
        ax.plot([gs, uk], [y, y], color=C[cls], lw=LW_SEC, zorder=2)
        ax.scatter([gs], [y], s=22, marker="o", facecolor="none",
                   edgecolor=C[cls], linewidth=LW_SEC, zorder=3)
        ax.scatter([uk], [y], s=22, marker="s", color=C[cls], zorder=3)

    ax.axvline(snp_h2, color="0.35", lw=LW_REF, ls=":", zorder=1)

    # cohort markers and the SNP-h2 line go in a legend, one line in the same
    # header band as panel b's n= labels, right-aligned -- inside the axes it
    # would push the 4 rows down
    leg = [Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", ms=4.5, label="GS"),
           Line2D([], [], marker="s", ls="", color="0.4", ms=4, label="UKB"),
           Line2D([], [], color="0.35", lw=LW_REF, ls=":", label=r"$h^2_{\mathrm{snp}}$")]
    ax.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
              fontsize=6, frameon=False, handlelength=1.6, handletextpad=0.4,
              columnspacing=1.4, borderaxespad=0.0)

    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.1, 0.9)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_xlabel(r"additive genetic variance  ($\hat{V}_A$)")
    return [m[2] for m in order]


def main():
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [0.95, 1.22, 0.98]})
    informants(axa)
    layers(axb)
    clabels = creatinine(axc)

    for ax, tag in [(axa, "a"), (axb, "b"), (axc, "c")]:
        ax.text(-0.26, 1.06, tag, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
        sns.despine(ax=ax, trim=True, offset=4, left=ax is axc)
    # despine(trim) resets the ticks, so the category labels go on afterwards
    axa.set_xticks([0, 1], labels=["parent\nhas disease", "grandparent\nhas disease"],
                   fontsize=6.5, linespacing=1.4)
    axb.set_xticks(range(3), labels=["same cohort", "two cohorts", "mis-paired"],
                   fontsize=6.5)
    axb.set_xlabel("paired estimates")
    axc.set_yticks(range(len(clabels))[::-1], labels=clabels)
    axc.tick_params(axis="y", length=0)

    fig.tight_layout(w_pad=2.0)
    # PNG only, canvas fixed at figsize (bbox "tight" would trim the declared width)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(HERE / "fig5.png")
    print("wrote fig5.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
