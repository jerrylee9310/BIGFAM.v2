# Figure 3 + Figure S4 — how large is the cross-method disagreement, and where does it come from

For each of the 416 real traits (UK Biobank 340, Generation Scotland 76), six
fixed-decay methods (Falconer, HE/PCGC, SEM-step, SEM-const, QuantHer/TetraHer-step,
-const) estimate $h^2$ from the same relative pairs and differ only in the assumed
common-environmental decay pattern. $\mathrm{SD}_{\mathrm{fix}}$, the SD of the six
estimates, measures how much that assumption moves the answer. **Fig. 3a** plots
$\mathrm{SD}_{\mathrm{fix}}$ / mean SE against decay-pattern separability
$|\hat w_C - 0.5|$ (sliding window of 100 traits), split into a between-assumption
and a within-assumption component: only the between-assumption part crosses the
one-error-bar line. **Fig. 3b** ranks three candidate drivers (separability,
$\hat V_C$, $\hat V_A$) by their partial Spearman $\rho$ with raw
$\mathrm{SD}_{\mathrm{fix}}$, each adjusted for the other two plus mean SE, with 95%
trait-resample bootstrap CIs, for all traits and per cohort. **Fig. S4** shows the
nine added-variable plots (3 sets x 3 predictors) behind Fig. 3b. The stdout blocks
`[S-A]`..`[S-F]` are the supplementary diagnostics; `[S-F]` is Supplementary Table 1
(window-width sensitivity).

## Files

- `plot_fig3.py` — reads Supplementary Data 1, writes `fig3.png` and prints all
  numbers quoted in the Results section and Supplementary Table 1.
- `plot_figS4.py` — imports the computation from `plot_fig3.py`, writes `figS4.png`,
  prints the nine partial $\rho$ [95% CI] values.
- `_style.py` — shared Nature Genetics-style matplotlib settings. Copied as-is from
  the paper's figure style module; falls back to DejaVu Sans if Arial isn't installed.
- `fig3.png`, `figS4.png` — checked-in outputs (400 dpi).

Both scripts are the paper's `fig3.py` / `figS4.py` with only the data loader
replaced (`load_sd1()`); the analysis and plotting code is unchanged.

## Data

The scripts read `supple_data/Supplementary_Data_1_trait_estimates.csv` (repo root
relative). `supple_data/` is not committed: download Supplementary Data 1 from the
paper and place the file there. It contains trait-level aggregates only (per-trait
estimates, SEs and pair counts, no individual-level data), so everything here runs
without managed-access phenotype data.

Column names in Supplementary Data 1 follow the manuscript and differ from the
internal names the scripts use; `load_sd1()` maps them back:

| Supplementary Data 1 | script | |
|---|---|---|
| `V_A`, `V_C`, `w_C` | `V_G`, `V_S`, `w_s_cal` | BIGFAM.v2 estimates and decay rate |
| `cohort` (`GS`/`UKB`) | `dataset` (`GS23471`/`UKB41907`) | cohort label |
| `trait_id` | `trait` | |
| `h2_<Method>`, `se_<Method>` | `<method>_h2`, `<method>_se` | Falconer, HE_PCGC, SEM_step, SEM_const, QHTH_step, QHTH_const → falconer, herg, sem_step, sem_const, ldak_step, ldak_const |
| — | `sig_bigfam` | not in the file; defined as `z_V_A > 1.645` |

The loader also sorts rows by (cohort, trait id), the order of the private analysis
table, so the bootstrap resamples draw the same indices and the CIs reproduce digit
for digit. `SD_fix` / `mean_SE_fix` are present in the file but recomputed by the
script from the six per-method columns (identical values).

## Run

```bash
.venv/bin/pip install -e ".[analysis]"     # from the repo root
.venv/bin/python analysis/fig3_figS4_disagreement/plot_fig3.py    # ~12 s
.venv/bin/python analysis/fig3_figS4_disagreement/plot_figS4.py   # ~10 s
```

## External dependencies

None beyond the `analysis` extra (matplotlib, seaborn, pandas) plus `scipy`.

## Validation

The original `fig3.py` / `figS4.py` were run against the private
`db/05_comparison/analysis_set.tsv` in the working repo that produced the submitted
draft, and the ported scripts against Supplementary Data 1 in this repo's `.venv`.

- stdout: identical line for line (the only difference is the `wrote ...` line, since
  this folder writes PNG only). Manuscript numbers, as printed: disagreement ratio
  0.461 near separability 0 and 1.831 at 0.30, crossing one at 0.152; between-assumption
  component 1.719 and within-assumption 0.320 at 0.30 (`[S-F]`, width 100); adjusted
  partial $\rho$ for all 416 traits: separability +0.658 [+0.590, +0.717], $\hat V_C$
  +0.387 [+0.270, +0.494], $\hat V_A$ −0.099 [−0.203, +0.000]; UKB +0.653 / +0.372 /
  −0.067, GS +0.644 / +0.691 / +0.158.
- PNGs: with the original scripts run in the same environment as the port
  (matplotlib 3.11.1), `fig3.png` and `figS4.png` are pixel-identical (0 differing
  pixels, same shape). Against the paper's environment (matplotlib 3.10.9) the data
  are identical but the raster differs by anti-aliasing: `fig3.png` 1.34% of pixels
  differ (max abs diff 229/255, text and line edges); `figS4.png` is 5 px shorter
  (2314 vs 2319 rows, tight-bbox rounding) so it cannot be diffed pixel-wise.
