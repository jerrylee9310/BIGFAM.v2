# analysis/

One folder per figure. Simulation figures (Figs. 2, S1–S3, S7) run on
synthetic data; real-data figures (Figs. 3–5, S4–S6, S8) read the trait-level
table published as the paper's supplementary data. Fig. 1 is a schematic —
no code here.


| Figure    | Folder                                                  | What it checks                                                                            | Extra deps                  |
| --------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------- |
| 2, S3     | [`fig2_figS3_cross_method`](fig2_figS3_cross_method/)   | 8 methods recover `V_A` under slow/fast common-environmental decay                        | LDAK, R + OpenMx (optional) |
| S1        | [`figS1_phase2_calibration`](figS1_phase2_calibration/) | Phase 2's `w_hat_C` ridge estimator: shrinkage toward 0.5, where it's reliable            | —                           |
| S2        | [`figS2_bias_decomposition`](figS2_bias_decomposition/) | Splits Fig. 2's bias into decay-rate error vs. the NNLS non-negativity constraint         | —                           |
| S7        | [`figS7_circularity_check`](figS7_circularity_check/)   | Sorting traits by BIGFAM.v2's own `w_hat_C` isn't circular                                | R + OpenMx (optional)       |
| 3, S4     | [`fig3_figS4_disagreement`](fig3_figS4_disagreement/)   | Method disagreement grows with separability, driven by the decay assumption               | supplementary data          |
| 4, S5, S6 | [`fig4_figS5_figS6_snp_h2`](fig4_figS5_figS6_snp_h2/)   | Agreement with SNP-based heritability by separability threshold                           | supplementary data          |
| 5, S8     | [`fig5_figS8_cross_cohort`](fig5_figS8_cross_cohort/)   | `w_hat_C` tracks shared family history; `V_A` replicates across cohorts, `V_C` can differ | supplementary data          |




## Quick start

```bash
cd BIGFAM.v2-publish
python3 -m venv .venv && .venv/bin/pip install -e ".[analysis]"

# self-contained, no external tools (a few minutes each):
.venv/bin/python analysis/figS1_phase2_calibration/generate.py
.venv/bin/python analysis/figS1_phase2_calibration/plot.py
.venv/bin/python analysis/figS2_bias_decomposition/generate.py
.venv/bin/python analysis/figS2_bias_decomposition/plot.py

# ships a pre-computed reference so the plot works immediately;
# re-running generate.py needs LDAK + R/OpenMx for the full method set
.venv/bin/python analysis/fig2_figS3_cross_method/plot_fig2.py
.venv/bin/python analysis/fig2_figS3_cross_method/plot_figS3.py

# quick check (~1 min) or full scale (~18 min, R/OpenMx optional)
.venv/bin/python analysis/figS7_circularity_check/generate.py --small 150 75
.venv/bin/python analysis/figS7_circularity_check/plot.py --small

# real-data figures (seconds each)
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

