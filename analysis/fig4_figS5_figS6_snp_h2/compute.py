"""The bootstrap table behind Fig. 4: each method's correlation with SNP-h2
along the separability ladder, with paired-bootstrap 95% CIs and P(r_v2 > r_method).

    .venv/bin/python analysis/fig4_figS5_figS6_snp_h2/compute.py [--boot 10000]

Input:  supple_data/Supplementary_Data_1_trait_estimates.csv (340 UK Biobank traits)
Output: method_corr.csv next to this file (plot_fig4.py reads it)

Rows written are the cumulative separability axis (|w_C - 0.5| >= t, t = 0.00 ... 0.30)
x both/continuous/binary x 8 methods against snp_h2.

Reproducibility: seed 20260723, B = 10,000, and the trait order and resampling order of
the paper's script, so the CIs match its table bit for bit. Bin conditions precede the
cumulative ones in the RNG stream, so they are still drawn here and then skipped.

Method notes: trait indices are drawn once per condition and shared by every method, so
P(r_v2 > r_method) is defined on the same resamples. Missing values are deleted pairwise
(list-wise deletion would drop the whole binary stratum), and correlations are NaN-aware,
with the effective n stored per cell. Spearman is a single rank transform per condition
followed by Pearson on the same resamples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from _sd1 import load_sd1

HERE = Path(__file__).resolve().parent
SEED = 20260723
MIN_N = 3               # minimum n for a correlation; strata are never dropped on n,
                        # the per-cell n is stored and the plots decide what to show

METHODS = ["falconer", "herg", "sem_step", "sem_const", "ldak_step", "ldak_const",
           "bigfam", "bigfam_v1"]
COLS = [f"{m}_h2" for m in METHODS] + ["snp_h2"]

DIST_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.51)]
# three wide bins: splitting the 59 binary traits five ways leaves strata of 7-12,
# while three gives 23/18/18 and keeps all trait types comparable
DIST_BINS3 = [(0.00, 0.10), (0.10, 0.20), (0.20, 0.51)]
DIST_CUM = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
KEEP_AXIS = "dist_cum"   # the only axis Fig. 4 reads


def rowcorr(A, B):
    """Row-wise Pearson with pairwise NaN deletion. A, B: (rows, n)."""
    M = np.isfinite(A) & np.isfinite(B)
    A = np.where(M, A, 0.0)
    B = np.where(M, B, 0.0)
    n = M.sum(1)
    sa, sb = A.sum(1), B.sum(1)
    num = n * (A * B).sum(1) - sa * sb
    den = np.sqrt(np.maximum(n * (A * A).sum(1) - sa ** 2, 0)
                  * np.maximum(n * (B * B).sum(1) - sb ** 2, 0))
    return np.where((den > 0) & (n >= 3), num / np.where(den > 0, den, 1.0), np.nan)


def ranked(x):
    """Rank-transform the observed values, leaving missing ones as NaN."""
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    out[ok] = rankdata(x[ok])
    return out


def subsets(d):
    """Conditions in the original's order up to and including the cumulative axis.

    The original continued with tier / v1class conditions; those come *after* the
    cumulative ones in the RNG stream, so dropping them changes nothing above."""
    c, b = d.kind == "continuous", d.kind == "binary"
    dist = (d.w_s_cal - 0.5).abs()
    out = [("overall", "all", "both", pd.Series(True, index=d.index)),
           ("overall", "all", "continuous", c), ("overall", "all", "binary", b)]
    for axis, bins in [("dist_bin", DIST_BINS), ("dist_bin3", DIST_BINS3)]:
        for lo, hi in bins:
            m = (dist >= lo) & (dist < hi)
            out += [(axis, f"[{lo:.2f},{hi:.2f})", "both", m),
                    (axis, f"[{lo:.2f},{hi:.2f})", "continuous", c & m),
                    (axis, f"[{lo:.2f},{hi:.2f})", "binary", b & m)]
    for t in DIST_CUM:
        m = dist >= t
        out += [(KEEP_AXIS, f">={t:.2f}", "both", m),
                (KEEP_AXIS, f">={t:.2f}", "continuous", c & m),
                (KEEP_AXIS, f">={t:.2f}", "binary", b & m)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    d = load_sd1()
    # SNP-h2 benchmark: Neale's estimates are UK Biobank only, so this narrows
    # the 416 traits to the 340 UKB ones
    d = d[d.snp_h2.notna()].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    rows = []

    for axis, sub, kind, mask in subsets(d):
        s = d[mask]
        if len(s) < MIN_N:
            continue
        ii = rng.integers(0, len(s), size=(args.boot, len(s)))     # drawn once per condition (paired)
        if axis != KEEP_AXIS:      # drawn only to keep the RNG stream aligned with the original
            continue
        X = s[COLS].to_numpy(float)
        R = np.column_stack([ranked(X[:, j]) for j in range(X.shape[1])])
        y, yr = X[:, -1], R[:, -1]
        yb, ybr = y[ii], yr[ii]

        block, boot_p = [], {}
        for j, m in enumerate(METHODS):
            x = X[:, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_N:
                continue
            bp = rowcorr(x[ii], yb)
            bs = rowcorr(R[:, j][ii], ybr)
            boot_p[m] = bp
            block.append(dict(
                axis=axis, subset=sub, kind=kind, method_x=m, method_y="snp_h2",
                n=int(ok.sum()),
                pearson=float(np.corrcoef(x[ok], y[ok])[0, 1]),
                pearson_lo=float(np.nanpercentile(bp, 2.5)),
                pearson_hi=float(np.nanpercentile(bp, 97.5)),
                spearman=float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]),
                spearman_lo=float(np.nanpercentile(bs, 2.5)),
                spearman_hi=float(np.nanpercentile(bs, 97.5)),
                P_bigfam_gt=np.nan))
        bf = boot_p.get("bigfam")
        if bf is not None:
            for r in block:
                if r["method_x"] == "bigfam":
                    continue
                o = boot_p[r["method_x"]]
                k = np.isfinite(bf) & np.isfinite(o)
                r["P_bigfam_gt"] = float((bf[k] > o[k]).mean())
        rows += block

    r = pd.DataFrame(rows)
    r.to_csv(HERE / "method_corr.csv", index=False)
    print(f"-> {HERE / 'method_corr.csv'}  {r.shape}  (boot={args.boot})")

    ok = lambda x: "OK " if x else "BAD"
    inci = r.pearson.between(r.pearson_lo, r.pearson_hi).mean()
    print(f"{ok(True)} {r.groupby(['subset', 'kind']).ngroups} conditions x 8 methods")
    print(f"{ok(inci > 0.95)} point estimate inside its own CI: {inci:.1%}")
    print(f"{ok(r[r.method_x != 'bigfam'].P_bigfam_gt.notna().all())} P_bigfam_gt present for every non-bigfam row")

    for kind in ("both", "continuous"):
        print(f"\n=== cumulative separability ({kind}) ===")
        q = r[r.kind == kind]
        for sub in sorted(q.subset.unique()):
            g = q[q.subset == sub]
            b = g[g.method_x == "bigfam"].iloc[0]
            o = g[~g.method_x.isin(["bigfam", "bigfam_v1"])].sort_values("pearson").iloc[-1]
            v1, ss = g[g.method_x == "bigfam_v1"].pearson, g[g.method_x == "sem_step"].pearson
            print(f"  dist{sub}  n={b.n:3d}   BIGFAM {b.pearson:.3f} "
                  f"[{b.pearson_lo:.2f},{b.pearson_hi:.2f}]   "
                  f"v1 {v1.iloc[0] if len(v1) else float('nan'):.3f}   "
                  f"SEM-step {ss.iloc[0]:+.3f}   "
                  f"best other {o.method_x:>10s} {o.pearson:+.3f}   "
                  f"gap {b.pearson - o.pearson:+.3f}   P(v2>best) {o.P_bigfam_gt:.2f}")


if __name__ == "__main__":
    main()
