"""generate.py -- circularity check: does BIGFAM.v2 win a comparison sorted by
its own output?

The paper sorts traits by separability |w_hat_C - 0.5| (a BIGFAM.v2 output) before
comparing methods. That looks circular. This settles it by making synthetic traits
whose TRUE w_C is known: pair phenotypes are actually generated (10,000 pairs per
DOR, as in Fig. 2), Phase 1 estimates (rho_hat, Sigma_hat), and every method is
asked to recover V_A. Two designs:

    GRID  N_GRID traits, true w_C ~ U(0.01, 0.99)  -- separability really varies
    NULL  N_NULL traits, true w_C = 0.5 fixed       -- separability cannot vary

If sorting by w_hat_C (estimated) and by the true w_C give the same trend on GRID,
the estimated-axis sort is not manufacturing the result. If NULL is flat, the
threshold itself is not an artifact.

Methods: bigfam (real pipeline), Falconer, HE, and 3 closed-form GLS variants
(AE/const/step, the non-ML counterpart of SEM) via the public bigfam API only.
SEM (OpenMx) adds the real ML fit if Rscript+OpenMx are available; skipped
(NaN columns) otherwise. LDAK is left out, as in the original circularity-check
analysis: it's REML (slow per-trait at this N) and its non-negativity
constraint would put it on a different axis than the unconstrained methods here.

APPROXIMATION -- read this before trusting the numbers:
The original analysis resamples the true (V_A, V_C) pair for each synthetic
trait from `db/05_comparison/merged.tsv`, i.e. from BIGFAM's own fitted values
on the paper's 417 real traits. That file is private phenotype-derived data and
is not shipped in this repository. Here the (V_A, V_C) pairs are instead drawn
from a Dirichlet(alpha) fit to that same pool by method of moments (see
ALPHA below) -- same mean by construction, but smaller variance and a heavier
V_A tail than the real pool. This folder therefore reproduces the QUALITATIVE
conclusion (does sorting by the estimate vs. the truth agree; is NULL flat) but
NOT the paper's exact numbers.

Run:
    .venv/bin/python generate.py --small 150 75      # quick check, ~seconds x scale
    .venv/bin/python generate.py                      # full scale (4000 grid + 2000 null)
Output: results/circular_pairlevel_scen[_small].parquet (per-trait; _small is gitignored)
        results/circular_pairlevel[_small].tsv         (threshold-binned summary)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bigfam
from bigfam.io.load import load_artifacts

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

SEED_BASE = 20_260_821_000
N_D, W_G = 10_000, 0.5
N_GRID, N_NULL = 4_000, 2_000
CUTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
X_G = 0.5 ** np.arange(1, 4)
DORS = [1, 2, 3]
SEM_CONDS = ["AE", "step", "const"]
METHODS = ["bigfam", "AE", "const", "step", "falconer", "he",
           "sem_AE", "sem_step", "sem_const"]

# Dirichlet(V_A, V_C, V_E) fit by method of moments to the 417-trait real pool
# (db/05_comparison/merged.tsv, cut at V_A+V_C<0.95) -- see docstring above.
ALPHA = np.array([0.98602772, 0.46952702, 3.34049484])


def sample_va_vc(rng, n):
    """Approximates the real (V_A, V_C) resampling pool: draw from Dirichlet(ALPHA),
    keep V_A + V_C < 0.95 (same cutoff as the original), reject-and-refill to n."""
    out = np.empty((0, 2))
    while len(out) < n:
        draw = rng.dirichlet(ALPHA, size=n)
        keep = draw[:, 0] + draw[:, 1] < 0.95
        out = np.vstack([out, draw[keep, :2]])
    return out[:n]


def generate_pairs(v_a, v_c, w_c, seed, n_d=N_D):
    """Continuous relative pairs at DOR 1/2/3 for one trait with true (V_A, V_C, w_C)."""
    rng = np.random.default_rng(seed)
    id1, id2, dor, pids, pvals = [], [], [], [], []
    nxt = 0
    for d in (1, 2, 3):
        rho = W_G ** d * v_a + w_c ** (d - 1) * v_c
        za, zb = rng.standard_normal(n_d), rng.standard_normal(n_d)
        y1, y2 = za, rho * za + np.sqrt(1.0 - rho ** 2) * zb
        i1 = np.arange(nxt, nxt + n_d); nxt += n_d
        i2 = np.arange(nxt, nxt + n_d); nxt += n_d
        id1.append(i1); id2.append(i2); dor.append(np.full(n_d, d))
        pids += [i1, i2]; pvals += [y1, y2]
    pairs = pd.DataFrame({"id1": np.concatenate(id1), "id2": np.concatenate(id2),
                          "dor": np.concatenate(dor)})
    pheno = pd.DataFrame({"phenotype": np.concatenate(pvals)},
                         index=np.concatenate(pids))
    pheno.index.name = "id"
    return pairs, pheno


def falconer_estimate(rho):
    """Zero-class closed form: h2_d = rho_d / w_G^d, inverse-variance weighted."""
    wgd = W_G ** np.arange(1, 4)
    h2_d = rho.rho_hat / wgd
    w = wgd ** 2 / rho.sigma_hat ** 2
    return float((w * h2_d).sum() / w.sum())


def he_estimate(rho, pairs, pheno):
    """Haseman-Elston (AE, original form): (y1-y2)^2 regressed on w_G^d, origin-fixed."""
    rel = {1: W_G, 2: W_G ** 2, 3: W_G ** 3}
    y = pheno["phenotype"]
    y1 = y.loc[pairs["id1"].values].to_numpy()
    y2 = y.loc[pairs["id2"].values].to_numpy()
    d2 = (y1 - y2) ** 2
    x = pairs["dor"].map(rel).to_numpy()
    sigma2 = float(y.var(ddof=1))
    slope = float(((d2 - 2.0 * sigma2) * x).sum() / (x * x).sum())
    return -slope / (2.0 * sigma2)


def fixed_gls_full(rho, Sinv):
    """Closed-form (non-ML) GLS counterpart of SEM's AE/const/step -- same rho,
    full Sigma_hat, no iterative fit. Sanity-checks the ML fit (see report())."""
    out = {}
    xg = np.broadcast_to(X_G, rho.shape)
    quad = lambda a, b: np.einsum("ni,nij,nj->n", a, Sinv, b)   # noqa: E731
    out["AE"] = quad(xg, rho) / quad(xg, xg)
    for name, c2v in [("const", np.ones(3)), ("step", np.array([1.0, 0.0, 0.0]))]:
        c2 = np.broadcast_to(c2v, rho.shape)
        a11, a12, a22 = quad(xg, xg), quad(xg, c2), quad(c2, c2)
        b1, b2 = quad(xg, rho), quad(c2, rho)
        det = a11 * a22 - a12 ** 2
        out[name] = (a22 * b1 - a12 * b2) / det
    return out


def sem_summary(pairs, pheno, rho_hat):
    """Per-DOR 2x2 sample covariance (continuous) -- SEM's own input, not rho_hat
    (fitting a correlation matrix as if it were a covariance is the Cudeck 1989
    error rho_hat would otherwise invite)."""
    y = pheno["phenotype"]
    mats = []
    for d in DORS:
        pd_ = pairs[pairs["dor"] == d]
        y1 = y.loc[pd_["id1"].values].to_numpy()
        y2 = y.loc[pd_["id2"].values].to_numpy()
        mats.append(np.cov(np.vstack([y1, y2])))
    return mats


def sem_row(trait, pairs, pheno, rho_hat, n_d):
    rec = {"trait": trait}
    for d, M in enumerate(sem_summary(pairs, pheno, rho_hat), start=1):
        rec[f"c11_{d}"], rec[f"c12_{d}"], rec[f"c22_{d}"] = M[0, 0], M[0, 1], M[1, 1]
        rec[f"Nd_{d}"] = n_d
    return rec


def run_sem(records, out_csv):
    """Batches all traits through one Rscript call. Empty DataFrame (all-NaN sem_*
    columns downstream) if Rscript or OpenMx is unavailable -- this is expected to
    degrade gracefully, not to crash the run."""
    if shutil.which("Rscript") is None or not records:
        return pd.DataFrame()
    ev_csv = out_csv.parent / "_sem_in.csv"
    pd.DataFrame(records).to_csv(ev_csv, index=False)
    cmd = ["Rscript", str(HERE / "sem.R"), str(ev_csv), str(out_csv)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_csv.exists():
        print(f"  [sem] FAILED rc={r.returncode}: {r.stderr.strip()[:300]}")
        return pd.DataFrame()
    return pd.read_csv(out_csv)


def summarise(df):
    rows = []
    for design, g in df.groupby("design"):
        for axis in ["est", "true"]:
            dist = (g.w_hat - 0.5).abs() if axis == "est" else (g.w_true - 0.5).abs()
            for t in CUTS:
                s = g[dist >= t]
                if len(s) < 50 or s.V_G_true.std() < 1e-9:
                    continue
                row = dict(design=design, axis=axis, t=t, n=len(s),
                           bias_bigfam=(s.VA_bigfam - s.V_G_true).mean())
                for m in METHODS:
                    v = s[f"VA_{m}"]
                    row[f"r_{m}"] = v.corr(s.V_G_true)
                    row[f"rmse_{m}"] = np.sqrt(((v - s.V_G_true) ** 2).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=int, nargs=2, metavar=("N_GRID", "N_NULL"),
                    help="quick check, e.g. --small 150 75. Writes the _small "
                         "outputs plot.py --small reads; they are gitignored")
    args = ap.parse_args()
    n_grid, n_null = args.small if args.small else (N_GRID, N_NULL)
    out_dir = OUT
    tag = "_small" if args.small else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED_BASE)
    scen = []
    for design, n in [("grid", n_grid), ("null", n_null)]:
        va_vc = sample_va_vc(rng, n)
        w = np.full(n, 0.5) if design == "null" else rng.uniform(0.01, 0.99, n)
        for k in range(n):
            scen.append((design, va_vc[k, 0], va_vc[k, 1], w[k]))

    calib = load_artifacts()   # packaged Phase 2 calibration
    rhos, Sigs, meta, sem_recs, fal, hee, w_hat_list, va_bigfam, vc_bigfam = (
        [], [], [], [], [], [], [], [], [])
    t0 = time.perf_counter()
    for i, (design, va, vc, w) in enumerate(scen):
        pairs, pheno = generate_pairs(va, vc, w, SEED_BASE + i)
        cov = pheno[[]]                                     # intercept-only (no covariates)
        rho = bigfam.estimate_rho(pairs, cov, pheno, "continuous", cov_cols=[])
        ws = bigfam.estimate_ws(rho, calib)
        dec = bigfam.decompose(rho, ws)

        rhos.append(rho.rho_hat); Sigs.append(rho.Sigma_hat)
        w_hat_list.append(ws.w_s_cal)
        va_bigfam.append(dec.V_G); vc_bigfam.append(dec.V_S)
        fal.append(falconer_estimate(rho))
        hee.append(he_estimate(rho, pairs, pheno))
        sem_recs.append(sem_row(i, pairs, pheno, rho.rho_hat, N_D))
        meta.append((design, va, vc, w))
        if (i + 1) % 500 == 0:
            print(f"  phase1 {i + 1}/{len(scen)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    t_loop = time.perf_counter() - t0

    rho_arr = np.stack(rhos); Sig_arr = np.stack(Sigs)
    est = {"bigfam": np.asarray(va_bigfam),
           **fixed_gls_full(rho_arr, np.linalg.inv(Sig_arr)),
           "falconer": np.asarray(fal), "he": np.asarray(hee)}

    t1 = time.perf_counter()
    sem = run_sem(sem_recs, Path(tempfile.mkdtemp()) / "_sem_out.csv")
    t_sem = time.perf_counter() - t1
    if sem.empty:
        print("  [sem] empty -- check Rscript/OpenMx. sem_* columns will be NaN.")
        for c in SEM_CONDS:
            est[f"sem_{c}"] = np.full(len(scen), np.nan)
        status0 = float("nan")
    else:
        piv = sem.pivot(index="trait", columns="condition", values="V_G").reindex(range(len(scen)))
        for c in SEM_CONDS:
            est[f"sem_{c}"] = piv[c].to_numpy()
        status0 = float((sem.status == 0).mean())

    df = pd.DataFrame(meta, columns=["design", "V_G_true", "V_S_true", "w_true"])
    df["w_hat"] = w_hat_list
    for m, v in est.items():
        df[f"VA_{m}"] = v
    df["VC_bigfam"] = vc_bigfam
    for k in range(3):
        df[f"rho{k + 1}"] = rho_arr[:, k]
        df[f"sig{k + 1}{k + 1}"] = Sig_arr[:, k, k]
    df["sig12"], df["sig13"], df["sig23"] = Sig_arr[:, 0, 1], Sig_arr[:, 0, 2], Sig_arr[:, 1, 2]
    df.to_parquet(out_dir / f"circular_pairlevel_scen{tag}.parquet")

    sm = summarise(df)
    sm.to_csv(out_dir / f"circular_pairlevel{tag}.tsv", sep="\t", index=False,
              float_format="%.4f")
    pd.set_option("display.width", 250)
    print(sm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {out_dir / f'circular_pairlevel{tag}.tsv'}")

    ok = lambda b: "OK " if b else "XX "   # noqa: E731
    per = (t_loop + t_sem) / len(scen)
    print("\n=== sanity checks ===")
    print(f"{ok(len(df) == n_grid + n_null)} {len(df)} traits "
          f"(grid {n_grid} + null {n_null}) x {N_D:,} pairs/DOR")
    for c in SEM_CONDS:
        a, b = df[f"VA_sem_{c}"], df[f"VA_{c}"]
        rr = a.corr(b); md = (a - b).abs().median()
        flag = ok(rr > 0.99) if not np.isnan(rr) else "-- "
        print(f"{flag}SEM-{c:<5s} vs closed-form: r={rr:.4f}  median|diff|={md if pd.isna(md) else round(md,5)}")
    for m in ["falconer", "he"]:
        print(f"{ok(df[f'VA_{m}'].notna().all())} {m} NaN {df[f'VA_{m}'].isna().sum()} "
              f"(mean {df[f'VA_{m}'].mean():.3f})")
    print(f"{ok(status0 == 1.0) if not np.isnan(status0) else '-- '}SEM status==0 rate {status0}")
    print(f"{ok(True)} time: loop {t_loop:.0f}s + SEM batch {t_sem:.0f}s = "
          f"{per:.3f}s/trait -> N=6,000 extrapolated {per * 6000 / 60:.1f} min")


if __name__ == "__main__":
    main()
