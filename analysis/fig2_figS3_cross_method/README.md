# Figure 2 + Figure S3 — cross-method simulation

Both figures plot the same simulation output (`results/raw.parquet`), just
sliced differently: Fig. 2 shows continuous phenotypes at slow/fast
common-environmental decay only; Fig. S3 is the full panel (continuous +
binary, all three decay rates including the degenerate point). They share
one folder so the (expensive) simulation only runs once.

**What it tests**: 1,000 replicates of 10,000 relative pairs per degree of
relatedness (DOR 1/2/3), truth `V_A=0.5, V_C=0.2`, at three common-environment
decay rates `w_C in {0.2, 0.5 (degenerate), 0.8}`, both continuous and binary
(prevalence `K=0.3`) phenotypes. 10 method x condition combinations are fit
to the same data: Falconer, HE (continuous) / PCGC (binary), SEM (AE / step /
const, via OpenMx), LDAK QuantHer/TetraHer (AE / step / const), and BIGFAM.v1
+ BIGFAM.v2 (both decay-estimating). This is the evidence that only
BIGFAM.v2 recovers the truth in both decay regimes; every fixed-decay-shape
method is right only under its own assumption.

## Files

| file | what |
|---|---|
| `config.py` | truth, grid, seed, N per DOR — the scientific constants (unchanged from the source) |
| `dgm.py` | individual-level relative-pair generator |
| `evidence.py` | Phase 1: pairs -> (rho_hat, Sigma_hat), shared by every method |
| `relatedness.py` | Mendelian relatedness constants (w_G^d) |
| `methods/falconer.py`, `he.py`, `pcgc.py` | native-Python zero-decay estimators |
| `methods/bigfam.py` | wraps the installed `bigfam` package (this repo's Phase 1->2->3) |
| `methods/bigfam_v1.py` | faithful reimplementation of the original BIGFAM.v1 band machine |
| `engines/ldak.py` | drives the external LDAK binary (QuantHer/TetraHer) |
| `engines/runners.py`, `engines/sem.R` | drives R/OpenMx (SEM AE/step/const) |
| `run.py` | main driver — runs all 10 method x condition combos over the grid |
| `plot_fig2.py`, `plot_figS3.py` | read `results/raw.parquet`, write the figures |
| `_style.py` | shared Nature-Genetics-style matplotlib settings |
| `results/raw.parquet` | **shipped reference output** — the exact 1,000-replicate data the submitted Fig. 2 / Fig. S3 were built from, so the plot scripts work right after cloning |

## External dependencies

- **LDAK binary** (QuantHer/TetraHer) — not bundled (its license does not
  permit redistribution). Download it yourself and place the executable at
  `bin/ldak` (`chmod +x bin/ldak`). If it's missing, `engines/ldak.py`
  returns `NaN`/`note="no-ldak"` for the `quanther` rows and everything else
  still runs — `run.py --check` prints a warning banner when this happens.
- **R + the OpenMx package** (SEM AE/step/const) — install R, then
  `install.packages("OpenMx")`. If `Rscript` isn't on `PATH`, `sem` rows come
  back empty and everything else still runs (same graceful-skip behavior).
- Everything else (Falconer, HE, PCGC, BIGFAM.v1, BIGFAM.v2) is pure Python
  and needs nothing beyond this repo's `.venv[analysis]`.

## Running it

```bash
# from the repo root, with .venv[analysis] installed
.venv/bin/python analysis/fig2_figS3_cross_method/run.py --check     # ~10s, 1 big-N replicate, asserts sanity
.venv/bin/python analysis/fig2_figS3_cross_method/run.py --reps 10   # ~2.5 min with LDAK+SEM, quick smoke test
.venv/bin/python analysis/fig2_figS3_cross_method/run.py             # full 1,000 reps -- see timing below
.venv/bin/python analysis/fig2_figS3_cross_method/plot_fig2.py
.venv/bin/python analysis/fig2_figS3_cross_method/plot_figS3.py
```

`run.py` always writes `results/raw.parquet`, overwriting the shipped
reference. To go back to the shipped reference, `git checkout -- results/raw.parquet`.

### Timing (measured on the machine this was built on, LDAK + OpenMx both available)

`--reps 10` took 158.6s (10,000 pairs/DOR, both scales, all three w_C).
Extrapolating linearly to the paper's `--reps 1000`: **roughly 4.4 hours**,
consistent with LDAK's per-replicate REML calls being the bottleneck (LDAK
runs once per replicate per condition; SEM is batched once at the end and R
startup cost is paid only once). Without LDAK/R installed, only the native
Python methods run and it's a few minutes for 1,000 reps.

## Validation performed

1. `run.py --check` (one large-N replicate) passes its built-in asserts:
   `rho_hat` within 0.02 of truth, BIGFAM.v2 and BIGFAM.v1 both recover
   `V_G` near 0.5 on clean data, Falconer is inflated above BIGFAM (absorbs
   `V_S`), and PCGC matches Tet-Falconer under no covariate adjustment.
2. `run.py --reps 10` was run and merged against the frozen reference
   (`pjt-bf/analysis/simulation/results/raw.parquet`, rows with `rep < 10`)
   on `(scale, w_s, rep, method, condition)`. Seeds are deterministic per
   `(scale, w_s, rep)`, so this is an exact parity check, not a statistical
   comparison. Max `|V_G_hat diff|` per method:

   | method | max abs diff |
   |---|---|
   | bigfam | 1.5e-12 |
   | sem | 1.1e-09 |
   | falconer | 6.1e-13 |
   | he | 0.0 |
   | pcgc | 0.0 |
   | quanther (LDAK) | 0.0 |
   | bigfam_v1 | up to 0.62 (see caveat below) |

   Everything except `bigfam_v1` reproduces to floating-point precision.
3. **`bigfam_v1` caveat**: `methods/bigfam_v1.py`'s `make()` holds *one* RNG
   that is shared across the *entire* grid sweep inside a single `run()`
   call (faithful to the original pipeline — not something introduced here).
   So a `(scale, w_s)` block's `bigfam_v1` draws depend on how many prior
   calls happened first, which depends on `--reps`. Checked block-by-block:
   the very first grid cell (`continuous`, `w_s=0.2`) matches the reference
   to 1e-16 regardless of `--reps` (nothing precedes it); every later cell
   only matches bit-for-bit when run with the *same* `--reps` as the
   reference (1000). This is expected, not a bug, and affects only
   `bigfam_v1` — no other method's RNG state crosses grid cells.
4. `plot_fig2.py` / `plot_figS3.py` were run against the shipped reference
   `results/raw.parquet` and compared pixel-by-pixel to the submitted
   `paper/figures/fig2.png`: 97.9% of pixels identical, mean channel
   difference 2.4/255. The residual is font-rendering only (this machine
   substitutes Liberation/DejaVu Sans for Arial, per `_style.py`'s fallback
   list) — the data and layout are identical.
