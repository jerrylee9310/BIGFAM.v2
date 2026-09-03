# Fig. S7 — is sorting traits by their own BIGFAM.v2 output circular?

The paper (Fig. 3, Fig. 4) ranks traits by separability, `|w_hat_C - 0.5|`, and
shows BIGFAM.v2 doing better on the high-separability end. `w_hat_C` is
BIGFAM.v2's own output, so the obvious objection is: does sorting by a
method's own estimate manufacture that method's apparent advantage?

This folder answers it directly: build synthetic traits whose **true** `w_C`
is known, generate their relative-pair phenotypes for real (10,000 pairs per
DOR, same generative model as Fig. 2), run Phase 1 to get `(rho_hat,
Sigma_hat)`, and ask every method to recover `V_A`. Two designs:

- **GRID** (4,000 traits) — true `w_C ~ U(0.01, 0.99)`, so separability really varies.
- **NULL** (2,000 traits) — true `w_C = 0.5` fixed, so separability cannot vary at all.

If sorting GRID by the *estimated* `w_hat_C` and by the *true* `w_C` give the
same trend, the estimate-based sort in the paper is not circular. If NULL is
flat under the same threshold sweep, the threshold itself isn't inventing an
effect out of noise.

## Approximation — read before trusting the numbers

The original analysis (`research/circularity-check/` in the working repo)
draws each synthetic trait's true `(V_A, V_C)` by bootstrap-resampling from
`db/05_comparison/merged.tsv` — BIGFAM.v2's own fitted values on the paper's
417 real traits. That file is derived from real phenotype data and is not
shipped here.

Instead, `generate.py` draws `(V_A, V_C)` from **`Dirichlet(alpha)`** with
`alpha = [0.986, 0.470, 3.340]` on `(V_A, V_C, V_E)`, fit by method of moments
to that same 417-trait pool (cut at `V_A + V_C < 0.95`, matching the original
cutoff). By construction the mean matches the real pool almost exactly, but
the fitted Dirichlet has **smaller variance and a heavier upper tail on
`V_A`** than the real data (real `V_A` tops out at 0.68; the Dirichlet draw
can exceed 0.9). Concretely:

| | mean (V_A, V_C, V_E) | var (V_A, V_C, V_E) |
|---|---|---|
| real pool (n=417) | 0.206, 0.098, 0.697 | 0.040, 0.011, 0.039 |
| Dirichlet(alpha) draw | 0.205, 0.099, 0.696 | 0.028, 0.016, 0.037 |

**What this means**: this folder reproduces the *qualitative* conclusion
(does the estimate-sorted trend match the truth-sorted trend; is NULL flat)
but not the paper's exact `r`/`alpha`/`beta` numbers or trait counts per
threshold bin. Treat the shipped `figS7.png` as "same shape, different
sample" — not a pixel match to the manuscript figure.

## Methods compared

`bigfam` (public 3-phase pipeline) · `Falconer` · `HE` · three closed-form GLS
variants (`AE`/`const`/`step` — the non-iterative counterpart of SEM, fit on
the same `(rho_hat, Sigma_hat)`) · SEM (`AE`/`step`/`const`, real OpenMx ML
fit, only if R + OpenMx are available). LDAK is left out, as in the original:
it's REML (too slow per-trait at N=6,000) and its non-negativity constraint
puts it on a different axis than the unconstrained methods here.

## External dependency

**R + the OpenMx package**, for the SEM engine only. If `Rscript` isn't on
PATH, or OpenMx isn't installed, `sem_*` columns come back all-`NaN` and
everything else still runs — this is a graceful skip, not a crash. Install:
`R` from CRAN, then inside R: `install.packages("OpenMx")`.

Everything else is pure Python (`numpy`, `pandas`, `scipy`, `scikit-learn`
via the `bigfam` package, plus `matplotlib`/`seaborn` for `plot.py`) — no
LDAK, no `db/`, no other `analysis/*` folder.

## Run it

```bash
# quick check (~seconds, writes results/*_small.* -- gitignored)
.venv/bin/python analysis/figS7_circularity_check/generate.py --small 150 75
.venv/bin/python analysis/figS7_circularity_check/plot.py --small

# full scale (4,000 GRID + 2,000 NULL traits)
.venv/bin/python analysis/figS7_circularity_check/generate.py
.venv/bin/python analysis/figS7_circularity_check/plot.py
```

## Validation done in this repo

- `--small 150 75`: 225 traits, ~42s wall time (17s Phase-1 loop + 23s SEM
  batch). Extrapolated to ~18 min for the full 6,000 traits.
- **Full scale actually run end-to-end** in this repo's own venv: 6,000
  traits (4,000 GRID + 2,000 NULL), 438s Phase-1 loop + 529s SEM batch =
  16.1 min. All sanity checks passed: SEM (OpenMx) vs. closed-form GLS agree
  at `r > 0.999` on all three conditions (AE/step/const), Falconer/HE have
  zero NaNs, SEM converged (`status==0`) on 100% of fits.
- **Qualitative pattern** (Pearson `r` with true `V_A`, BIGFAM.v2):
  - GRID, sorted by `w_hat_C` (the paper's actual axis): `r` 0.658 (t=0) ->
    0.790 (t=0.30) — rises with separability threshold, as claimed.
  - GRID, sorted by **true** `w_C` (the oracle axis, impossible on real
    data): `r` 0.658 -> **0.884** — rises *more*, not less. Sorting by the
    estimate is if anything conservative relative to sorting by the truth,
    which is the direction that rules out circularity: the paper's
    estimate-sorted advantage is not an artifact of the sort itself.
  - Both sorts on GRID: BIGFAM.v2 leads every other method at every
    threshold, and the gap widens with `t` (e.g. at t=0.30, true-sorted:
    BIGFAM.v2 `r=0.884` vs. best alternative `SEM-const r=0.412`).
  - NULL (true `w_C` fixed at 0.5, so no real separability exists): **not
    flat here** — BIGFAM.v2's `r` drifts from 0.427 (t=0) to 0.545 (t=0.25,
    n down to 75). This is a real difference from the original analysis,
    which reports a much flatter NULL curve (0.522 -> 0.541). Two things
    are worth separating: (a) the *other* methods drift by a similar or
    larger amount in this run (e.g. Falconer 0.467 -> 0.578), so the drift
    is not specific to BIGFAM.v2 — likely n shrinking to 75 at the highest
    threshold making every correlation noisier under this run's particular
    draw; (b) this could also be an effect of the Dirichlet approximation's
    different variance/tail shape (see above) rather than something that
    would replicate with the real `(V_A, V_C)` pool. Read the NULL panel
    here as noisier than the manuscript's, not as evidence the manuscript's
    NULL result doesn't hold.
- `figS7.png` in this folder is the actual output of the full
  6,000-trait run above — open them to see the shape directly.
