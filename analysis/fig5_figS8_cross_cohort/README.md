# Figure 5 + Figure S8 — what moves and what stays when the relatives change

Fig. 5 asks what happens to BIGFAM.v2's estimates when the *relationship
composition* behind them changes, using real Generation Scotland (GS) and
UK Biobank (UKB) fits:

- **5a** — GS family-history items ("does your father / mother / grandparent
  have X?") share the same relative pairs, so only *whom the item asks about*
  changes. The pedigree predicts a larger decay rate for grandparent items
  than parent items; the estimated $\hat w_C$ splits accordingly (median 0.785
  vs 0.218, one-sided Mann–Whitney $P = 8.4 \times 10^{-4}$).
- **5b** — BIGFAM.v2's $|\Delta \hat V_A|$ for the same four traits paired
  three ways: the same measurement re-expressed within one cohort (a, noise
  floor), the same trait in GS vs UKB (b), and the same estimates deliberately
  mis-paired (c, null). Medians 0.030 / 0.055 / 0.151.
- **5c** — serum creatinine estimated in both cohorts by four methods. Line
  length is the cross-cohort gap: BIGFAM.v2 0.039, SEM (const) 0.107,
  Falconer 0.279, SEM (step) 0.423. Dotted line = UKB SNP-$h^2$.

Fig. S8 spreads 5c out to all four traits and all eight methods (adds
BIGFAM.v1, HE and the LDAK QuantHer rows), showing that the two engines
(OpenMx SEM, LDAK) coincide under the same assumption.

Both figures are trait-level: nothing here needs individual-level data.

## Files

- `compute.py` — rebuilds, in memory, the three trait-level tables the paper's
  working repo used as figure inputs (`famhist()`, `vg_matched()`,
  `engine_compare()`) from Supplementary Data 1. Run it directly to print the
  summary numbers quoted above.
- `plot_fig5.py` — writes `fig5.png`. `plot_figS8.py` — writes `figS8.png`.
  Plotting code is the working repo's, unchanged except for where the data
  comes from.
- `_style.py` — shared Nature Genetics-style matplotlib settings, copied as-is
  from the other analysis folders.
- `fig5.png`, `figS8.png` — checked-in outputs (400 dpi).

## Data

`compute.py` reads `supple_data/Supplementary_Data_1_trait_estimates.csv`
(repo root; not committed — download Supplementary Data 1 from the paper and
place it there; column dictionary in `supple_data/README.md`). It carries 6
significant digits, so rebuilt numbers differ from the frozen working-repo
tables at the $10^{-5}$ level (see Validation). SD1 uses manuscript notation
(`V_A`, `V_C`, `w_C`); the rebuilt tables keep the working repo's package
notation (`V_G`, `V_S`, `w_s_cal`) so the plotting code stays byte-identical —
`RENAME` in `compute.py` is the mapping.

How the three tables are cut from SD1:

- **Family-history table (5a, 18 rows)** — rows with `cohort == GS` and
  `trait_type == binary`. The informant is the `trait_id` suffix: `_F` father,
  `_M` mother, `_G` grandparent, `_Y` self; father + mother are grouped as
  "parent". The self item (asthma) has no family-history structure and is
  excluded from the test and the plot.
- **Matched-pair table (5b, 18 rows)** — 19 GS↔UKB same-trait matches
  (`PAIRS` in `compute.py`: height/50, weight/21002, BMI/21001, body fat/23099,
  hips/49, HDL/30760, total cholesterol/30690, glucose/30740,
  creatinine/30700, creatinine mg/dL/30700, urea/30670, systolic BP/4080,
  diastolic BP/4079, pulse/102, FEV1/20150, FVC/20151, ECG heart rate/12336,
  P duration/12338, QRS duration/12340) are screened for $z_{V_A} > 2$ in
  *both* cohorts. Four independent measurements survive — Total_cholesterol↔30690,
  Creatinine↔30700, avg_dia↔4079, FVC↔20151 — and form category **b**.
  Category **a** is the same measurement expressed twice in one cohort (GS
  Creatinine vs Creat_mgdl, i.e. µmol/L vs mg/dL; UKB 20151 FVC best measure
  vs 3062 FVC). Category **c** is the 12 GS×UKB combinations of the four
  traits that are *not* the true match. For every pair
  $\Delta V_A = V_A^{(x)} - V_A^{(y)}$, $\mathrm{se}_\Delta = \sqrt{\mathrm{se}_x^2 + \mathrm{se}_y^2}$,
  $z = \Delta V_A / \mathrm{se}_\Delta$, $p = 2\,\Phi(-|z|)$; the plot uses $|\Delta V_A|$.
- **Method table (5c, S8; 40 rows)** — the five surviving pairs (the four above
  plus Creat_mgdl) × eight methods, GS and UKB values taken from SD1's
  `h2_<method>` columns of the two rows: Falconer→`h2_Falconer`,
  HE→`h2_HE_PCGC`, SEM-step/const→`h2_SEM_*`, LDAK-step/const→`h2_QHTH_*`,
  BIGFAM.v1→`h2_BIGFAM_v1`, BIGFAM.v2→`h2_BIGFAM_v2` (= `V_A`). `snp_h2` and
  `snp_h2_confidence` come from the UKB row (Neale LDSC; "high" = Neale
  primary phenotype, drawn dashed in S8, otherwise dotted). Fig. 5c shows
  creatinine only and drops HE (it sits on the Falconer line) and the LDAK
  rows; Fig. S8 shows everything except Creat_mgdl.

Two columns of the frozen tables are not in SD1 and are not rebuilt: `tier`
(the working repo's separability label) and the free-text `why` column.
Neither is used by the plots.

## Run

```bash
.venv/bin/pip install -e ".[analysis]"     # from the repo root
.venv/bin/python analysis/fig5_figS8_cross_cohort/compute.py      # prints the summary numbers
.venv/bin/python analysis/fig5_figS8_cross_cohort/plot_fig5.py
.venv/bin/python analysis/fig5_figS8_cross_cohort/plot_figS8.py
```

Each takes a few seconds. matplotlib 3.11 prints a harmless
`findfont: Failed to find font weight normal, now using 0.` from the custom
mathtext font set in `_style.py`.

## External dependencies

None — pandas, numpy, scipy, matplotlib, seaborn from the `analysis` extra.
Arial if installed (falls back to DejaVu Sans).

## Validation

The three rebuilt tables were diffed against the frozen CSVs in the private
working repo that produced the submitted figures
(`research/vg-network-robustness/results/{b7_famhist,b1_vg_matched,b1c_engine_compare}.csv`,
read-only). Row sets and row order identical in all three; all string columns
exact; max abs diff over every numeric column 4.8e-6 (family-history),
1.6e-5 (matched pairs; the largest is in the $z$ columns, which SD1 rounds at 6
significant digits) and 9.6e-7 (method table). The manuscript numbers come out
unchanged: median $\hat w_C$ 0.785 vs 0.218, $P = 8.40 \times 10^{-4}$;
median $|\Delta \hat V_A|$ 0.030 / 0.055 / 0.151; creatinine gaps
BIGFAM.v2 0.039, SEM-const 0.107, Falconer 0.279, SEM-step 0.423.

The rendered PNGs were compared pixel-wise against the original
`paper/figures/{fig5,figS8}.py` run on the frozen CSVs. In the same
matplotlib (3.10.9 or 3.11.1) the port differs in ≤ 0.0013 % of pixels
(fig5 0.0005 %, figS8 0.0013 %), max abs diff 1/255 — single anti-aliasing
steps from the 6-digit rounding. Across matplotlib versions (3.10 vs 3.11) the
same script renders text slightly differently (fig5: 4.5 % of pixels; figS8:
tight bbox 1036×2890 vs 1040×2882), so compare within one version.
