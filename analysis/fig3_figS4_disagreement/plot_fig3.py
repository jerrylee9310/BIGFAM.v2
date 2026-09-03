"""Figure 3 -- how large the cross-method disagreement is, and where it comes from.

    .venv/bin/python analysis/fig3_figS4_disagreement/plot_fig3.py

Input:  supple_data/Supplementary_Data_1_trait_estimates.csv (416 traits)
Output: fig3.png

a  SD across the six fixed-decay h2 estimates, in units of the mean SE, split
   into a between-assumption and a within-assumption (estimation framework)
   component, along the separability axis. Grey points are per-trait SD/mean SE.
b  Adjusted partial rank correlations of three candidate drivers with SD_fix:
   dot = partial rho controlling the other two plus mean SE, whisker = 95%
   bootstrap CI, shown for all traits and per cohort.

Also prints supplementary blocks [S-A]..[S-F] (SE control, SE heterogeneity,
ratio scale, VIF, older normalisations, window sensitivity).

Supplementary Data 1 column names are renamed to the internal ones on load
(V_A -> V_G, V_C -> V_S, w_C -> w_s_cal, h2_<Method> -> <method>_h2).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl                                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import seaborn as sns                                             # noqa: E402
from scipy.stats import rankdata, spearmanr                       # noqa: E402

from _style import ACC_BETWEEN, ACC_WITHIN, C, FAINT, INK, LW_HERO, LW_SEC, \
    LW_REF, MUT, W2, rc

HERE = Path(__file__).resolve().parent
SD1 = HERE.parents[1] / "supple_data" / "Supplementary_Data_1_trait_estimates.csv"

rc()
SIX = ["falconer", "herg", "sem_step", "sem_const", "ldak_step", "ldak_const"]
H2, SE = [m + "_h2" for m in SIX], [m + "_se" for m in SIX]
# 3 decay assumptions x 2 estimation frameworks -- separates the two sources
PAIR = {"none": ["falconer", "herg"], "const": ["sem_const", "ldak_const"],
        "step": ["sem_step", "ldak_step"]}
BLUE, VERM = ACC_WITHIN, ACC_BETWEEN     # fig3 accent pair (_style.py)
NBOOT, PBOOT, SEED = 10_000, 2_000, 20260804
WINDOW = 100                   # sliding window in a; the ends keep the first/last 100 traits
TOP = 3.0                      # y limit in a; traits above it are reported on stdout
THIRDS = ["lower", "middle", "upper"]
COV = ["ident", "V_S", "V_G"]                # all three come from one BIGFAM.v2 fit
GROUPS = ["all", "UKB", "GS"]                # order within each row of b (top to bottom)
GMARK = {"all": "o", "UKB": "s", "GS": "^"}          # cohort = marker shape
GDY = {"all": 0.19, "UKB": 0.0, "GS": -0.19}         # vertical offset within a row
GALPHA = {"all": 1.0, "UKB": 0.55, "GS": 0.55}       # all-traits solid, cohorts faded
IDENT = r"$|\hat w_C - 0.5|$"
COV_LAB = {"ident": IDENT, "V_S": r"$\hat V_C$", "V_G": r"$\hat V_A$"}
SDFIX = r"$\mathrm{SD}_{\mathrm{fix}}$"

# Supplementary Data 1 column -> the analysis_set.tsv name the code below uses
SD1_METHOD = {"falconer": "Falconer", "herg": "HE_PCGC", "sem_step": "SEM_step",
              "sem_const": "SEM_const", "ldak_step": "QHTH_step", "ldak_const": "QHTH_const"}
RENAME = {"trait_id": "trait", "V_A": "V_G", "V_C": "V_S", "w_C": "w_s_cal",
          **{f"h2_{v}": f"{k}_h2" for k, v in SD1_METHOD.items()},
          **{f"se_{v}": f"{k}_se" for k, v in SD1_METHOD.items()}}


def load_sd1():
    """Supplementary Data 1 -> the frame analysis_set.tsv gave the original script."""
    if not SD1.exists():
        raise FileNotFoundError(
            f"{SD1} not found. Download Supplementary Data 1 from the paper and place it at "
            "supple_data/Supplementary_Data_1_trait_estimates.csv (supple_data/ is not committed).")
    d = pd.read_csv(SD1, dtype={"trait_id": str}).rename(columns=RENAME)
    d["dataset"] = d.cohort.map({"GS": "GS23471", "UKB": "UKB41907"})   # labels the code compares on
    d["sig_bigfam"] = (d.z_V_A > 1.645).astype(int)                     # not in SD1; this is its definition
    # analysis_set.tsv was sorted by (dataset, trait); SD1 is not. Restore that order so the
    # trait-resample bootstrap draws the same indices and the CIs match the paper digit for digit.
    return d.sort_values(["dataset", "trait"]).reset_index(drop=True)


def save_png(fig, stem, tight=True):
    """PNG only, 400 dpi as in the paper (_style.save would also write a PDF)."""
    with mpl.rc_context({} if tight else {"savefig.bbox": None}):
        fig.savefig(HERE / f"{stem}.png")
    print(f"wrote {stem}.png")


def load():
    """Per-trait SD_fix, mean SE and separability for the 416 traits.

    snp_h2 is deliberately not read: it exists only for UK Biobank, so dropna
    would remove all 76 Generation Scotland traits. `dataset` drives the cohort
    split in b and the stratification in figS4.
    """
    d = load_sd1()
    d = d[["dataset", "trait", "V_G", "V_S", "w_s_cal", "sig_bigfam"] + H2 + SE]
    n_raw = len(d)
    d = d.dropna().reset_index(drop=True)
    d["sd"] = d[H2].std(axis=1, ddof=1)
    d["mse"] = d[SE].mean(axis=1)
    d["ident"] = (d.w_s_cal - 0.5).abs()
    return d, n_raw


def split(d):
    """Split SD_fix into between- and within-assumption parts (df 5 each, so
    asm^2 + est^2 = SD_fix^2 exactly)."""
    Y = [d[[m + "_h2" for m in ms]].to_numpy() for ms in PAIR.values()]
    gm = np.stack([v.mean(axis=1) for v in Y], axis=1)              # mean per assumption group
    ss_b = 2 * ((gm - gm.mean(axis=1)[:, None]) ** 2).sum(axis=1)   # between assumptions
    ss_w = sum(((v - v.mean(axis=1)[:, None]) ** 2).sum(axis=1) for v in Y)   # within a group
    asm, est = np.sqrt(ss_b / 5), np.sqrt(ss_w / 5)
    assert np.allclose(asm ** 2 + est ** 2, d.sd.to_numpy() ** 2), "variance decomposition identity broken"
    return asm, est, ss_b / (ss_b + ss_w)


def resid(y, Z):
    """Residual of rank(y) after regressing it on the control ranks."""
    Z = np.column_stack([np.ones(len(y))] + [rankdata(v) for v in Z])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def partial_spearman(x, y, *z):
    """Rank partial correlation of x and y given z -- the added-variable slope."""
    rx, ry = rankdata(x), rankdata(y)
    if not z:
        return np.corrcoef(rx, ry)[0, 1]
    return np.corrcoef(resid(rx, z), resid(ry, z))[0, 1]


def max_vif(cols):
    """Largest VIF on the rank scale, i.e. collinearity in the units used above."""
    R = [rankdata(v) for v in cols]
    return max(y.var() / resid(y, [v for i, v in enumerate(R) if i != j]).var()
               for j, y in enumerate(R))


def boot_ci(x, y, rng, cond=(), nboot=PBOOT):
    """Percentile CI from resampling traits: x, y and the controls share an index."""
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    r = np.array([partial_spearman(x[i], y[i], *(z[i] for z in cond)) for i in idx])
    return np.percentile(r, [2.5, 97.5])


def running(x, y, w=WINDOW):
    """Median over the w traits nearest in separability, with a CI band.

    The band is the CI of the median (notched-boxplot 1.57*IQR/sqrt(n)), not the
    IQR. Every position uses exactly w traits; the ends keep the first/last
    window. No further smoothing.
    """
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    windows = np.lib.stride_tricks.sliding_window_view(ys, w)
    q1, med, q3 = np.quantile(windows, [0.25, 0.5, 0.75], axis=1)
    starts = np.clip(np.arange(len(xs)) - w // 2, 0, len(xs) - w)
    q1, med, q3 = q1[starts], med[starts], q3[starts]
    ci = 1.57 * (q3 - q1) / np.sqrt(w)
    return xs, med, np.maximum(med - ci, 0), med + ci


def cross_one(x, y):
    """First x where the curve crosses 1, linearly interpolated."""
    i = int(np.argmax(y >= 1))
    return np.nan if i == 0 else np.interp(1, y[i - 1:i + 1], x[i - 1:i + 1])


def panel_components(ax, t, rel, mse, asm, est):
    """Panel a -- the total and its two components against separability."""
    ins = rel <= TOP
    ax.scatter(t[ins], rel[ins], s=3.2, c="0.80", alpha=0.38, lw=0, zorder=1,
               label="individual traits")
    for v, col, lw, alpha, nm in [(asm / mse, VERM, LW_SEC, 0.12, "between assumptions"),
                                  (est / mse, BLUE, LW_REF, 0.08, "within assumptions")]:
        xs, med, lo, hi = running(t, v)
        ax.fill_between(xs, lo, hi, color=col, alpha=alpha, lw=0, zorder=3)
        ax.plot(xs, med, "-", color=col, lw=lw, zorder=4, label=nm)
    # total before the split; total^2 = asm^2 + est^2, so it always sits above
    # both components and tracks the between-assumption curve closely
    xs, tot, _, _ = running(t, rel)
    ax.plot(xs, tot, "-", color=INK, lw=LW_HERO, zorder=5, label="all six")
    ax.axhline(1, color="0.8", lw=LW_REF, zorder=1.5)  # reference line, under the points
    ax.set(ylim=(0, TOP), yticks=[0, 1, 2, 3])
    ax.set_xlim(-0.008, 0.385)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
    ax.set_xlabel("decay-pattern separability")
    ax.set_ylabel(SDFIX + " / mean SE")
    # legend order = how the curves stack at the right edge; points last
    h, l = ax.get_legend_handles_labels()
    order = ["all six", "between assumptions", "within assumptions", "individual traits"]
    leg = ax.legend([h[l.index(nm)] for nm in order], order, loc="upper left", fontsize=6,
                    handlelength=1.3, labelspacing=0.3, handletextpad=0.5,
                    borderaxespad=0.2, frameon=False, markerscale=1.5)
    sns.despine(ax=ax, trim=False, offset=4)
    ax.spines["bottom"].set_bounds(0, 0.375)


def panel_coef(ax, stats, ns):
    """Panel b -- coefficient plot of the three candidate drivers.

    Dot = adjusted partial rho (the other two drivers plus mean SE controlled),
    whisker = 95% bootstrap CI, vertical line = 0. Each row carries all / UKB /
    GS: color = driver, marker = cohort.
    """
    rows = [("ident", "separability", C["decay"]),
            ("V_S", r"$\hat V_C$", INK),
            ("V_G", r"$\hat V_A$", FAINT)]
    ax.plot([0, 0], [-0.35, 2.36], color="0.8", lw=LW_REF, zorder=1)
    for y, (k, lab, col) in zip([2, 1, 0], rows):
        for g in GROUPS:
            yy, a = y + GDY[g], GALPHA[g]
            _, adj, (lo, hi) = stats[g][k]
            ax.plot([lo, hi], [yy, yy], color=col, lw=LW_SEC, alpha=a,
                    solid_capstyle="round", zorder=3)
            ax.plot([adj], [yy], GMARK[g], ms=4.2, mfc=col, mec="white", mew=0.5,
                    alpha=a, zorder=4)
        ax.text(-0.045, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=7, color=INK, linespacing=1.15)
    # legend carries shape -> cohort only; color is already given by the row
    ax.legend(handles=[Line2D([], [], ls="none", marker=GMARK[g], ms=4.2, mfc=MUT,
                              mec="white", mew=0.5, alpha=GALPHA[g],
                              label=f"{g} (n={ns[g]})") for g in GROUPS],
              loc="upper center", ncol=3, fontsize=6, handlelength=1.0,
              columnspacing=1.1, handletextpad=0.35, borderaxespad=0.0,
              frameon=False)
    # spines cut at the ticks, whose range covers the GS CI (-0.35, +0.79)
    ax.set_xlim(-0.46, 0.86)
    ax.set_ylim(-0.38, 2.78)          # headroom for the one-line legend
    ax.set_xticks([-0.4, 0, 0.4, 0.8])
    ax.set_xticklabels(["\u22120.4", "0", "0.4", "0.8"])
    ax.set_yticks([])
    ax.set_xlabel(r"partial $\rho$ with raw " + SDFIX)
    sns.despine(ax=ax, left=True, offset=4)
    ax.spines["bottom"].set_bounds(-0.4, 0.8)


def main():
    d, n_raw = load()
    n = len(d)
    sd, mse, t = d.sd.to_numpy(), d.mse.to_numpy(), d.ident.to_numpy()
    rel = sd / mse                                   # disagreement in units of its own SE
    vs, vg = d.V_S.to_numpy(), d.V_G.to_numpy()
    tt = np.asarray(pd.qcut(t, 3, labels=False))
    vt = np.asarray(pd.qcut(vs, 3, labels=False))
    cuts = np.percentile(t, [100 / 3, 200 / 3])
    asm, est, frac = split(d)
    V = {"ident": t, "V_S": vs, "V_G": vg}
    rng = np.random.default_rng(SEED)

    fig = plt.figure(figsize=(W2, 2.7))
    # fixed margins so the canvas is exactly the declared 183mm (pairs with save(tight=False))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.35], wspace=0.34,
                          left=0.054, right=0.992, bottom=0.185, top=0.91)
    axa = fig.add_subplot(gs[0, 0])
    panel_components(axa, t, rel, mse, asm, est)

    coh = d.dataset.str.replace(r"\d+$", "", regex=True).to_numpy()
    masks = {"all": np.ones(n, bool), "UKB": coh == "UKB", "GS": coh == "GS"}
    ns = {g: int(m.sum()) for g, m in masks.items()}
    stats = {}
    for g, m in masks.items():
        stats[g] = {}
        for k in COV:
            cond = [V[o][m] for o in COV if o != k] + [mse[m]]
            marg = partial_spearman(V[k][m], sd[m])
            adj = partial_spearman(V[k][m], sd[m], *cond)
            stats[g][k] = (marg, adj, boot_ci(V[k][m], sd[m], rng, cond))
    axb = fig.add_subplot(gs[0, 1])          # full height, rows ~0.5in apart
    panel_coef(axb, stats, ns)

    # panel letters measured off each panel's tight bbox, so they stay aligned
    # when label lengths change (a manual transAxes offset would not)
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [inv.transform(ax.get_tightbbox(ren)) for ax in (axa, axb)]
    ytop = max(b[1][1] for b in boxes) + 0.015
    for (x0, _), lab in zip((b[0] for b in boxes), "ab"):
        fig.text(x0, ytop, lab, fontsize=8, fontweight="bold",
                 ha="left", va="bottom", color=INK)
    save_png(fig, "fig3", tight=False)

    print(f"n = {n} traits  (from {n_raw} before dropping incomplete rows)")
    print(f"[A] separability tercile cutoffs {cuts[0]:.4f} / {cuts[1]:.4f}   n per third "
          + " ".join(str((tt == k).sum()) for k in range(3)))
    print(f"  SD_fix / mean SE   median {np.median(rel):.2f}   >1: {(rel > 1).sum()}/{n}   "
          f"by third " + " ".join(f"{np.median(rel[tt == k]):.2f}" for k in range(3))
          + "   share >1 by third " + " ".join(f"{(rel[tt == k] > 1).mean():.0%}" for k in range(3)))
    for nm, v in [("assumption", asm), ("framework ", est)]:
        print(f"  {nm} (df 5)  raw " + " ".join(f"{np.median(v[tt == k]):.3f}" for k in range(3))
              + "   / mean SE " + " ".join(f"{np.median((v / mse)[tt == k]):.2f}" for k in range(3))
              + f"   rho ident {spearmanr(v, t)[0]:+.3f}  V_S {spearmanr(v, vs)[0]:+.3f}")
    print(f"  assumption share of variance: median {np.median(frac):.3f}  "
          f"IQR {np.percentile(frac, 25):.3f}-{np.percentile(frac, 75):.3f}"
          f"   assumption > framework in {(asm > est).mean():.0%} of traits")
    print(f"  panel A: y axis capped at {TOP} ({(rel > TOP).sum()} traits fall outside, max {rel.max():.1f}; "
          f"curves use all {n} traits)")
    xs, tot, _, _ = running(t, rel)
    print(f"  all six cross 1 at separability {cross_one(xs, tot):.3f}   "
          + "  ".join(f"{THIRDS[k]} third {(rel[tt == k] > 1).mean():.0%} above 1"
                      for k in range(3)))
    print(f"  panel A curves: sliding window of {WINDOW} traits, band = CI of the median."
          + "".join(f"   {nm} {v[1][0]:.2f} -> {v[1][-1]:.2f}"
                    for nm, v in [("assumption", running(t, asm / mse)),
                                  ("framework", running(t, est / mse))]))
    print("[B] coefficient plot  (adjusted = other two + mean SE)")
    for k in COV:
        marg, adj, ci = stats["all"][k]
        mlo, mhi = boot_ci(V[k], sd, rng)
        print(f"  SD_fix ~ {k:7s} marginal {marg:+.3f} [{mlo:+.3f},{mhi:+.3f}]"
              f"   adjusted {adj:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
        for g in ("UKB", "GS"):
            _, a, c = stats[g][k]
            print(f"    {g:3s} (n={ns[g]:3d})        adjusted {a:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]")
    print(f"  V_G = 0 traits {(vg == 0).sum()}/{n}   SD_fix ~ mean SE {spearmanr(sd, mse)[0]:+.3f}"
          f"   V_G ~ mean SE {spearmanr(vg, mse)[0]:+.3f}   mean SE ~ V_S "
          f"{spearmanr(mse, vs)[0]:+.3f}   V_G ~ V_S {spearmanr(vg, vs)[0]:+.3f}")
    # [S-A] drop mean SE from the controls -- how much of V_G vanishing depends on it
    print("[S-A] adjusted for the other two covariates only (mean SE NOT controlled)")
    for k in COV:
        rest = [V[o] for o in COV if o != k]
        lo, hi = boot_ci(V[k], sd, rng, rest)
        print(f"  SD_fix ~ {k:7s} {partial_spearman(V[k], sd, *rest):+.3f} [{lo:+.3f},{hi:+.3f}]")
    # [S-B] the six SEs come from four different formulas -- how coarse mean SE is
    med_se = {m: np.median(d[m + "_se"]) for m in SIX}
    print("[S-B] median SE by method  " + "  ".join(f"{m} {v:.4f}" for m, v in med_se.items())
          + f"   max/min ratio {max(med_se.values()) / min(med_se.values()):.1f}x")
    # [S-C] on a ratio scale mean SE absorbs the V_S effect -- why b stays raw
    print("[S-C] scale sensitivity — within-stratum rho, raw SD_fix vs ratio")
    for nm, y in [("raw SD_fix   ", sd), ("SD_fix/meanSE", rel)]:
        print(f"  {nm}  within ident third ~ V_S   "
              + " ".join(f"{spearmanr(vs[tt == k], y[tt == k])[0]:+.3f}" for k in range(3))
              + "   within V_S third ~ ident "
              + " ".join(f"{spearmanr(t[vt == k], y[vt == k])[0]:+.3f}" for k in range(3)))
    print("  3x3 median raw SD_fix  (rows V_S third high->low, cols ident third)")
    for r in [2, 1, 0]:
        print(f"    V_S {THIRDS[r]:6s} " + " ".join(
            f"{np.median(sd[(vt == r) & (tt == c)]):.4f} (n={((vt == r) & (tt == c)).sum():3d})"
            for c in range(3)))
    print(f"[S-D] max VIF on ranks (3 covariates): {max_vif([V[k] for k in COV]):.2f}")
    # [S-E] the older df 2/3 normalisations, for comparison
    print("[S-E] old df 2/3 normalisation (cross-check against the current draft numbers)")
    for nm, v, f in [("assumption", asm, np.sqrt(5 / 2)), ("framework ", est, np.sqrt(5 / 3))]:
        print(f"  {nm} / mean SE " + " ".join(
            f"{np.median((v * f / mse)[tt == k]):.2f}" for k in range(3)))
    # [S-F] source of Supplementary Table 1: the headline numbers per window width
    print("[S-F] sliding-window width sensitivity  (panel A)")
    print("  width  %n   total near  total .30  crossing   Q .30   W .30")
    for w in (41, 61, 81, WINDOW, 121, 151):
        xs, tot, _, _ = running(t, rel, w)
        xa, a, _, _ = running(t, asm / mse, w)
        xe, e, _, _ = running(t, est / mse, w)
        print(f"  {w:5d} {w / n:4.0%}  {tot[0]:10.3f}  {np.interp(0.30, xs, tot):9.3f}  "
              f"{cross_one(xs, tot):8.3f}  {np.interp(0.30, xa, a):7.3f}  "
              f"{np.interp(0.30, xe, e):7.3f}")
    print("[sensitivity] same filter applied to all three rows")
    for tag, m in [("all", np.ones(n, bool)), ("V_G>0", vg > 0),
                   ("V_G significant", (d.sig_bigfam == 1).to_numpy())]:
        print(f"  {tag:16s} n={m.sum():3d}  "
              + "  ".join(f"{k} {spearmanr(V[k][m], sd[m])[0]:+.3f}" for k in COV))


if __name__ == "__main__":
    main()
