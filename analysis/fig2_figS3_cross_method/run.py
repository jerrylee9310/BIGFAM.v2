"""Main driver -- one command collects all 10 method x condition results.

    .venv/bin/python analysis/fig2_figS3_cross_method/run.py [--reps 10]
    .venv/bin/python  ...  /run.py --check        # 1 replicate self-check (dumps I/O)

Flow:
    for (scale, w_S, rep):
        pairs, pheno = dgm.generate(...)          # data
        rho          = evidence.compute(...)      # Phase 1 summary (once per replicate, shared by every method)
        for spec in SPECS(backend=loop):          # falconer, he, bigfam, bigfam_v1, quanther(LDAK)
            spec.run(rho, pairs, pheno, scale)
    sem_long(all replicates)                      # SEM only, batched outside the loop (amortizes R startup)

SPECS lists all 10 method x condition combinations in one place so it's easy
to see what actually ran. Run from this directory (or anywhere -- paths are
all relative to this file).
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

import config
import dgm
import evidence
from methods import falconer, he, pcgc
from methods import bigfam as bf
from methods import bigfam_v1 as bfv1
from engines import ldak
from engines.runners import run_sem, sem_summary
from bigfam.io import load_artifacts

warnings.filterwarnings("once", message="Sigma_hat not PSD")

_COLS = ["scale", "w_s", "rep", "method", "condition",
         "V_G_hat", "V_S_hat", "w_s_cal", "note"]

# ── method x condition manifest ─────────────────────────────────────────────
# backend "loop"  : per-replicate, run(rho, pairs, pheno, scale) -> {V_G, ...}
# backend "batch" : SEM -- batched outside the loop (sem_long handles it, run=None)
# scales          : which scale this method runs on (he=continuous, pcgc=binary; the rest run both)
Spec = namedtuple("Spec", "method condition backend run scales")
Spec.__new__.__defaults__ = (("continuous", "binary"),)   # scales defaults to both


def build_specs():
    calib = load_artifacts()                                          # packaged ws_calibration.json
    bigfam_fn = bf.make(calib)
    bigfam_v1_fn = bfv1.make()                                         # old band machine (fixed seed, stochastic)
    native = lambda fn: (lambda rho, pairs, pheno, scale: fn(rho))     # ignores individual-level data
    pcgc_run = lambda rho, pairs, pheno, scale: pcgc.estimate(rho, pairs, pheno, config.K_PREV)
    return [
        Spec("falconer", "AE",    "loop",  native(falconer.estimate)),
        Spec("he",       "AE",    "loop",  he.estimate,   ("continuous",)),  # original HE-SD: (y1-y2)^2
        Spec("pcgc",     "AE",    "loop",  pcgc_run,      ("binary",)),      # binary HE generalization (Golan)
        Spec("bigfam",   "decay", "loop",  native(bigfam_fn)),
        Spec("bigfam_v1","decay", "loop",  native(bigfam_v1_fn)),           # v1: old decay machine
        Spec("quanther", "AE",    "loop",  ldak.make("AE", config.K_PREV)),
        Spec("quanther", "step",  "loop",  ldak.make("step", config.K_PREV)),
        Spec("quanther", "const", "loop",  ldak.make("const", config.K_PREV)),
        Spec("sem",      "AE",    "batch", None),
        Spec("sem",      "step",  "batch", None),
        Spec("sem",      "const", "batch", None),
    ]


def _row(scale, w_s, rep, sp, out):
    return {"scale": scale, "w_s": w_s, "rep": rep,
            "method": sp.method, "condition": sp.condition,
            "V_G_hat": out.get("V_G", np.nan), "V_S_hat": out.get("V_S", np.nan),
            "w_s_cal": out.get("w_s_cal", np.nan),
            "note": out.get("note") or out.get("gate", "")}


def _sem_records(sem_store):
    """sem_store -> sem.R input dicts. id=scale/w_s/rep, Nd is per-DOR (same N_d everywhere)."""
    recs = []
    for scale, w_s, rep, mats, nd in sem_store:
        r = {"scale": scale, "w_s": w_s, "rep": rep}
        for d, M in enumerate(mats, start=1):
            r[f"c11_{d}"], r[f"c12_{d}"], r[f"c22_{d}"] = M[0, 0], M[0, 1], M[1, 1]
            r[f"Nd_{d}"] = nd
        recs.append(r)
    return recs


def sem_long(sem_store, out_csv):
    """SEM batch -> long rows (method=sem, V_G_hat/V_S_hat/note). No R -> []."""
    df = run_sem(_sem_records(sem_store), out_csv)
    if df.empty:
        return []
    df = df.rename(columns={"V_G": "V_G_hat", "V_S": "V_S_hat"})
    df["note"] = "status" + df.pop("status").astype("Int64").astype(str)
    df["method"] = "sem"
    return df.to_dict("records")


def run(reps, n_d=None):
    """n_d overrides config.N_D (pairs per DOR)."""
    n_d = config.N_D if n_d is None else n_d
    specs = build_specs()
    loop_specs = [s for s in specs if s.backend == "loop"]
    rows, sem_store = [], []
    t0 = time.time()

    for s_idx, scale in enumerate(config.SCALES):
        for w_idx, w_s in enumerate(config.WS_GRID):
            tic = time.time()
            for rep in range(reps):
                seed = config.replicate_seed(s_idx, w_idx, rep)
                pairs, pheno = dgm.generate(scale, w_s, seed, n_d=n_d)
                rho = evidence.compute(pairs, pheno, scale)
                sem_store.append((scale, w_s, rep,
                                  sem_summary(pairs, pheno, scale, rho.rho_hat), n_d))
                for sp in loop_specs:
                    if scale not in sp.scales:                 # he=continuous, pcgc=binary
                        continue
                    rows.append(_row(scale, w_s, rep, sp, sp.run(rho, pairs, pheno, scale)))
            print(f"  loop {scale:10s} w_S={w_s}  {reps} reps  ({time.time()-tic:5.1f}s)")

    sem_rows = sem_long(sem_store, config.RESULTS_DIR / "_sem_out.csv")
    print(f"  sem  batch  {len(sem_rows)} rows")
    print(f"  total {time.time()-t0:.1f}s")

    # raw long table -- no aggregation here, that's plot_fig2.py/plot_figS3.py's job
    return pd.DataFrame(rows + sem_rows).reindex(columns=_COLS)


def check():
    """1 replicate (large N) self-check -- Phase 1 rho accuracy, each method's output, engine I/O dumps."""
    scale, w_s, seed = "continuous", 0.2, 0
    pairs, pheno = dgm.generate(scale, w_s, seed, n_d=20_000)
    rho = evidence.compute(pairs, pheno, scale)

    print("=== PHASE1 evidence: rho_hat vs true (large N, should be close) ===")
    ok = True
    for d in (1, 2, 3):
        tr, hat = config.true_rho(w_s, d), rho.rho_hat[d - 1]
        good = abs(tr - hat) < 0.02; ok &= good
        print(f"  DOR{d}  true {tr:.3f}  hat {hat:.3f}  |diff| {abs(tr-hat):.3f}  "
              f"{'ok' if good else 'BAD'}")
    assert ok, "rho_hat far from truth -- Phase1/evidence broken"

    print("\n=== METHODS: V_G_hat (truth V_G=0.5, V_S=0.2; loop backend) ===")
    vg = {}
    for sp in build_specs():
        if sp.backend != "loop" or scale not in sp.scales:
            continue
        out = sp.run(rho, pairs, pheno, scale)
        vg[(sp.method, sp.condition)] = out["V_G"]
        vs = out.get("V_S")
        vs_s = f"  V_S={vs:.3f}" if vs is not None and not np.isnan(vs) else ""
        print(f"  {sp.method:9s} {sp.condition:6s} V_G={out['V_G']:.3f}{vs_s}  "
              f"{out.get('note') or out.get('gate', '')}")
    assert 0.4 < vg[("bigfam", "decay")] < 0.6, "BIGFAM should recover near 0.5"
    assert 0.3 < vg[("bigfam_v1", "decay")] < 0.7, "BIGFAM.v1 too should recover near 0.5 on clean rho"
    assert vg[("falconer", "AE")] > vg[("bigfam", "decay")], \
        "Falconer should be inflated by absorbing V_S, above BIGFAM"

    print("\n=== ENGINE I/O: LDAK input (relatives 'step') head + output ===")
    d = config.RESULTS_DIR / "_check_ldak"; d.mkdir(parents=True, exist_ok=True)
    ldak.write_rel(pairs, "step", d / "step.rel")
    print(f"  input {d/'step.rel'} (col5=w_G^d, col6=c_ij):")
    for line in (d / "step.rel").read_text().splitlines()[:3]:
        print(f"    {line}")
    print(f"  output quanther step V_G={vg.get(('quanther','step'), float('nan')):.3f}")

    print("\n=== ENGINE I/O: SEM input (per-DOR sample covariance) + output (1 rep batch) ===")
    mats = sem_summary(pairs, pheno, scale, rho.rho_hat)
    for d, M in enumerate(mats, 1):
        print(f"  input DOR{d} cov=[[{M[0,0]:.3f},{M[0,1]:.3f}],[{M[1,0]:.3f},{M[1,1]:.3f}]]")
    for r in sem_long([(scale, w_s, 0, mats, 20_000)],
                      config.RESULTS_DIR / "_check_sem.csv"):
        print(f"  output sem {r['condition']:6s} V_G={r['V_G_hat']:.3f} "
              f"V_S={r['V_S_hat']:.3f} {r['note']}")

    print("\n=== BINARY: PCGC vs Tet-Falconer (both unadjusted -- should be close) ===")
    pb, phb = dgm.generate("binary", w_s, seed, n_d=20_000)
    rb = evidence.compute(pb, phb, "binary")
    vg_pcgc = pcgc.estimate(rb, pb, phb, config.K_PREV)["V_G"]
    vg_tetf = falconer.estimate(rb)["V_G"]                 # Tet-Falconer (liability tetrachoric)
    print(f"  PCGC         V_G={vg_pcgc:.3f}  (prevalence K={config.K_PREV})")
    print(f"  Tet-Falconer V_G={vg_tetf:.3f}")
    assert abs(vg_pcgc - vg_tetf) < 0.10, "PCGC and tetrachoric too far apart -- check unadjusted assumptions"

    print("\ncheck passed -- Phase1, methods, engine I/O, PCGC all normal.")


def _engine_status():
    """LDAK/R availability -- warn loudly if results will be native-only (partial reproduction)."""
    import shutil
    missing = []
    if not ldak._LDAK.exists():
        missing.append(f"LDAK (binary not found: {ldak._LDAK})")
    if shutil.which("Rscript") is None:
        missing.append("R/OpenMx (Rscript not on PATH)")
    if missing:
        bar = "=" * 72
        print(bar)
        print("[!] missing engines -- results will be NATIVE-ONLY (Falconer, HE, PCGC, BIGFAM):")
        for m in missing:
            print(f"     - {m}  -> that method is SKIPPED")
        print("   Missing LDAK/SEM means this is not the full paper reproduction. Install: see README.md")
        print(bar)
    else:
        print("engines OK: LDAK + R/OpenMx available -> full reproduction possible.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=config.R)
    ap.add_argument("--check", action="store_true", help="run the 1-replicate self-check and exit")
    args = ap.parse_args()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _engine_status()

    if args.check:
        check()
        return

    print(f"truth: V_G={config.V_G_TRUE} V_S={config.V_S_TRUE}  N_d={config.N_D}  "
          f"reps={args.reps}  w_S={config.WS_GRID}")
    raw = run(args.reps)
    out = config.RESULTS_DIR / "raw.parquet"
    raw.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(raw)} rows)")
    print(f"plot: .venv/bin/python {Path(__file__).parent / 'plot_fig2.py'}")
    print(f"      .venv/bin/python {Path(__file__).parent / 'plot_figS3.py'}")


if __name__ == "__main__":
    main()
