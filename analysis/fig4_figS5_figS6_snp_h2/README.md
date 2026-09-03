# Figure 4 + Figures S5, S6 — SNP-h² recovery along the separability ladder

Fig. 4 benchmarks the eight family-based estimators against Neale-lab LDSC
SNP-h² on the 340 UK Biobank traits that have one, sliding a separability
threshold $t$ ($|\hat w_C - 0.5| \ge t$, $t = 0 \ldots 0.30$) so the subset
gets progressively better-determined:

- **a** — Pearson $r$ with SNP-h². BIGFAM.v2 climbs from 0.787 to 0.862; the
  five fixed-decay methods fall, SEM-step to −0.348.
- **b, c** — intercept $\alpha$ and slope $\beta$ of OLS(estimated h² ~ SNP-h²)
  refit in each subset. BIGFAM.v2 stays at $\alpha \approx 0$ and
  $\beta \approx 1.4$–$1.7$; SEM-step goes $\alpha$ 0.284 → 1.152,
  $\beta$ 0.911 → −3.353.

Fig. S5 repeats panel a split by trait type (both / continuous / binary).
Fig. S6 overlays, per method, the regression lines fit below and above a
separability gate of 0.20 to show the fixed-decay lines *rotate* (low-SNP-h²
traits are pushed up) while the BIGFAM lines barely move.

## Files

- `compute.py` — the paired trait-resample bootstrap behind Fig. 4a: Pearson
  and Spearman $r$ per method × threshold × trait type, 95% percentile CIs,
  and $P(r_{\text{BIGFAM.v2}} > r_{\text{method}})$ on the same resamples.
  Seed 20260723, B = 10,000. Writes `method_corr.csv` (161 rows).
- `plot_fig4.py` — reads `method_corr.csv` + Supplementary Data 1, writes
  `fig4.png`; prints the panel b/c $\alpha$/$\beta$ table.
- `plot_figS5.py`, `plot_figS6.py` — Supplementary Data 1 only; write
  `figS5.png`, `figS6.png`. `plot_figS6.py` prints the gap/rotation table
  and its gate sensitivity.
- `_sd1.py` — loads Supplementary Data 1 and renames its columns to the
  internal names the (otherwise unchanged) paper scripts use.
- `_style.py` — shared Nature Genetics-style matplotlib settings, copied as-is
  from the paper's figure style module.

Everything after `load_sd1()` in the plot scripts is the paper's own code;
`compute.py` is the paper's correlation-grid script trimmed to the rows Fig. 4
reads.

## Data

Download **Supplementary Data 1** from the paper and place it at
`supple_data/Supplementary_Data_1_trait_estimates.csv` (repo root;
`supple_data/` is not committed). It is trait-level only — no individual
data — and everything here is computed from it. The scripts narrow it to the
340 UKB traits with `snp_h2` (Generation Scotland has no Neale reference).

`_sd1.load_sd1()` maps the public column names back to the internal ones:
`V_A` → `V_G`, `V_C` → `V_S`, `w_C` → `w_s_cal`, `trait_type` → `kind`,
`h2_<Method>`/`se_<Method>` → `<method>_h2`/`<method>_se` (Falconer →
`falconer`, HE_PCGC → `herg`, SEM_step/const → `sem_step`/`sem_const`,
QHTH_step/const → `ldak_step`/`ldak_const`, BIGFAM_v1/v2 → `bigfam_v1`/`bigfam`),
and sorts rows by (cohort, trait id) — the order the original bootstrap
resampled in.

## Run

```bash
.venv/bin/pip install -e ".[analysis]"     # from the repo root
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/compute.py      # ~5 s
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_fig4.py    # ~3 s each
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_figS5.py
.venv/bin/python analysis/fig4_figS5_figS6_snp_h2/plot_figS6.py
```

`compute.py` must run before `plot_fig4.py`; the other two are independent.

## External dependencies

None — numpy, pandas, scipy, matplotlib, seaborn from the `analysis` extra.

## Validation

`compute.py` was compared row by row with `db/05_comparison/method_corr.parquet`
from the private working repo (read-only, never committed here), on the 161
rows Fig. 4 uses (cumulative axis × both/continuous/binary × 8 methods vs
`snp_h2`). Run with the working repo's environment (numpy 2.4.6) all eight
columns — `n`, `pearson`, `pearson_lo/hi`, `spearman`, `spearman_lo/hi`,
`P_bigfam_gt` — match bit for bit. Run in this repo's `.venv` (numpy 2.5.2)
the CIs, Spearman and `P_bigfam_gt` still match bit for bit and only the
Pearson point estimate differs, by at most 3.3e-16 (summation order inside
`np.corrcoef`). The two AE-model rows per condition (`sem_AE`, `ldak_AE`) in
the parquet are not in Supplementary Data 1 and not plotted anywhere.

The three plot scripts and the originals were run side by side. Stdout is
identical apart from the `wrote ...` line and the extra $\alpha$/$\beta$ table
`plot_fig4.py` prints. Rendered in the same matplotlib (3.11) the PNGs are
pixel-identical (0 differing pixels for all three). Against the paper's own
renders (matplotlib 3.10) the differences are rendering only: `fig4.png`
has the same 2881×1000 canvas with 8.2% of pixels differing (text
anti-aliasing), and the bbox-tight `figS5.png`/`figS6.png` canvases differ by
1–8 px in size.

Manuscript numbers reproduced: BIGFAM.v2 $r$ 0.787 ($t=0$) → 0.862
($t=0.30$); SEM-step $r$ −0.348 at $t=0.30$; SEM-step $\alpha$ 0.284 → 1.152
and $\beta$ 0.911 → −3.353.
