# analysis/ — figure-by-figure replication

Each subfolder reproduces one figure (or two, when they share data) from the
paper. The simulation figures run on **synthetic data only**. The real-data
figures (Figs. 3–5, Supp. Figs. S4–S6, S8) run from **Supplementary Data 1**,
the trait-level table published with the paper — individual-level phenotype
data (Generation Scotland, UK Biobank) is under managed access and is never
required here.

None of these folders import from each other, or from anywhere outside
themselves plus the installed `bigfam` package — each is a self-contained unit
you can read, run, and delete independently.

| Paper figure | Folder | What it checks | Needs beyond `bigfam` |
|---|---|---|---|
| **Fig. 2** + **Fig. S3** | [`fig2_figS3_cross_method/`](fig2_figS3_cross_method/) | 10 methods (Falconer, HE/PCGC, SEM, LDAK QuantHer/TetraHer, BIGFAM.v1/v2) recover `V_A` under slow/fast common-environmental decay — only BIGFAM.v2 works in both regimes | LDAK binary, R + OpenMx (both optional; missing ones just drop those columns) |
| **Fig. S1** | [`figS1_phase2_calibration/`](figS1_phase2_calibration/) | The Phase 2 `w_hat_C` ridge estimator's shrinkage-toward-0.5 and where it's reliable | nothing |
| **Fig. S2** | [`figS2_bias_decomposition/`](figS2_bias_decomposition/) | Decomposes Fig. 2's bias into decay-rate estimation error vs. the NNLS non-negativity constraint | nothing |
| **Fig. S7** | [`figS7_circularity_check/`](figS7_circularity_check/) | Sorting traits by BIGFAM.v2's own `w_hat_C` isn't circular — checked with synthetic traits whose true `w_C` is known | R + OpenMx (optional, SEM columns only) — see the approximation note below |
| **Fig. 3** + **Fig. S4** | [`fig3_figS4_disagreement/`](fig3_figS4_disagreement/) | Disagreement among the six decay-assuming methods grows with separability, and is driven by the decay assumption rather than the estimation framework | Supplementary Data 1 |
| **Fig. 4** + **Figs. S5, S6** | [`fig4_figS5_figS6_snp_h2/`](fig4_figS5_figS6_snp_h2/) | Agreement with SNP-based heritability by separability threshold (paired bootstrap recomputed from Supplementary Data 1) | Supplementary Data 1 |
| **Fig. 5** + **Fig. S8** | [`fig5_figS8_cross_cohort/`](fig5_figS8_cross_cohort/) | `w_hat_C` follows what relatives share (family-history items); `V_A` replicates across GS and UKB while `V_C` can differ | Supplementary Data 1 |

Fig. 1 is a schematic with no data. The three real-data folders read
`supple_data/Supplementary_Data_1_trait_estimates.csv` (416 traits, the
paper's Supplementary Data 1). It is not committed here: download it from the
paper and place it at that path. Column names follow the manuscript
(`V_A`, `V_C`, `w_C`); the scripts rename them to the package's internal names
(`V_G`, `V_S`, `w_s_cal`) on load.

## Quick start

```bash
cd BIGFAM.v2-publish
python3 -m venv .venv && .venv/bin/pip install -e ".[analysis]"

# fast (seconds-minutes), no external tools:
.venv/bin/python analysis/figS1_phase2_calibration/generate.py
.venv/bin/python analysis/figS1_phase2_calibration/plot.py
.venv/bin/python analysis/figS2_bias_decomposition/generate.py
.venv/bin/python analysis/figS2_bias_decomposition/plot.py

# ships a pre-computed reference so the plot works immediately;
# re-running generate.py needs LDAK + R/OpenMx for the full method set
.venv/bin/python analysis/fig2_figS3_cross_method/plot_fig2.py
.venv/bin/python analysis/fig2_figS3_cross_method/plot_figS3.py

# quick check (~seconds) or full scale (minutes, R/OpenMx optional)
.venv/bin/python analysis/figS7_circularity_check/generate.py --small 150 75
.venv/bin/python analysis/figS7_circularity_check/plot.py --small

# real-data figures (seconds each; need supple_data/Supplementary_Data_1_trait_estimates.csv)
.venv/bin/python analysis/fig3_figS4_disagreement/plot_fig3.py
.venv/bin/python analysis/fig3_figS4_disagreement/plot_figS4.py
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/compute.py      # bootstrap table, ~5 s
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_fig4.py
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_figS5.py
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_figS6.py
.venv/bin/python analysis/fig5_figS8_cross_cohort/plot_fig5.py
.venv/bin/python analysis/fig5_figS8_cross_cohort/plot_figS8.py
```

Every folder's own README has the exact run commands, external-dependency
install instructions, measured runtimes, and the validation performed against
the data that actually produced the submitted figures.

## Design choices

- **One folder can validate on its own.** `dgm.py`/`evidence.py`-equivalents,
  `_style.py`, closed-form method code — all duplicated per folder rather than
  imported across folders. A few hundred lines repeated four times is worth it
  so cloning just one folder (or deleting the other three) still works.
- **Reference outputs are checked in where regenerating them needs an external
  tool** (`fig2_figS3_cross_method/results/raw.parquet` needs LDAK + R;
  `figS7_circularity_check`'s full-scale run needs R/OpenMx for the SEM
  columns) **and left out where they don't** (`figS1_phase2_calibration/figS1.csv`
  is pure Python and ~5 min to regenerate — not worth 19MB in the repo).
- **`fig2_figS3_cross_method/bin/ldak` and any R/OpenMx install are never
  shipped** — LDAK's license doesn't permit redistribution, and R packages
  aren't repo assets. Every script degrades gracefully (documented per folder)
  when these are missing: the LDAK/SEM columns come back empty or `NaN`
  instead of crashing.
- **Fig. S7 is the one approximation.** Its true `(V_A, V_C)` values are
  bootstrap-resampled from the paper's own real-data fits in the working repo
  (`db/05_comparison/merged.tsv`) — private, not shippable. Here they're drawn
  from a `Dirichlet` distribution fit by method of moments to that same pool
  (mean matches by construction; variance and the `V_A` tail don't exactly).
  The qualitative conclusion replicates; the exact `r`/`alpha`/`beta` numbers
  and trait counts per bin do not. See the folder's README for the fit and
  the discrepancy, spelled out.

## Validation methodology

Every synthetic generator here is seeded deterministically. So instead of
eyeballing a regenerated plot against the manuscript figure, each folder's
`generate.py` output was diffed numerically, column-by-column, against the
actual data that produced the submitted figure (from the private working
repo, read-only, never committed here) — see each README's "Validation"
section for the numbers. `fig2_figS3_cross_method` additionally did a
pixel-level comparison of the rendered PNG against `paper/figures/fig2.png`.

The three real-data folders were validated the other way round: the original
plotting scripts (reading the private databases) and the ported ones (reading
Supplementary Data 1) were run in the same environment and their stdout and
PNGs compared — identical numbers, 0 differing pixels for Figs. 3, 4, S4–S6,
and ≤0.002% differing pixels (1/255 max) for Figs. 5 and S8, where
Supplementary Data 1's six significant digits round the frozen tables. The
Fig. 4 bootstrap table (`method_corr.csv`) reproduces the private one
bit-for-bit under the same seed.
