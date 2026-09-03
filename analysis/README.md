# analysis/ — figure-by-figure replication

The simulation figures run on synthetic data; the
real-data ones read the trait-level table published as the paper's
supplementary data. 


| Paper figure                  | Folder                                                   | What it checks                                                                                                                                        | Needs beyond `bigfam`                                                         |
| ----------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Fig. 2** + **Fig. S3**      | [`fig2_figS3_cross_method/`](fig2_figS3_cross_method/)   | 8 methods (Falconer, HE/PCGC, SEM, LDAK QuantHer/TetraHer, BIGFAM.v1/v2) recover `V_A` under slow/fast common-environmental decay                     | LDAK binary, R + OpenMx (both optional; missing ones just drop those columns) |
| **Fig. S1**                   | [`figS1_phase2_calibration/`](figS1_phase2_calibration/) | The Phase 2 `w_hat_C` ridge estimator's shrinkage-toward-0.5 and where it's reliable                                                                  | nothing                                                                       |
| **Fig. S2**                   | [`figS2_bias_decomposition/`](figS2_bias_decomposition/) | Decomposes Fig. 2's bias into decay-rate estimation error vs. the NNLS non-negativity constraint                                                      | nothing                                                                       |
| **Fig. S7**                   | [`figS7_circularity_check/`](figS7_circularity_check/)   | Sorting traits by BIGFAM.v2's own `w_hat_C` isn't circular — checked with synthetic traits whose true `w_C` is known                                  | R + OpenMx (optional, SEM columns only) — see the approximation note below    |
| **Fig. 3** + **Fig. S4**      | [`fig3_figS4_disagreement/`](fig3_figS4_disagreement/)   | Disagreement among the six decay-assuming methods grows with separability, and is driven by the decay assumption rather than the estimation framework | Supplementary Data                                                            |
| **Fig. 4** + **Figs. S5, S6** | [`fig4_figS5_figS6_snp_h2/`](fig4_figS5_figS6_snp_h2/)   | Agreement with SNP-based heritability by separability threshold                                                                                       | Supplementary Data                                                            |
| **Fig. 5** + **Fig. S8**      | [`fig5_figS8_cross_cohort/`](fig5_figS8_cross_cohort/)   | `w_hat_C` follows what relatives share (family-history items); `V_A` replicates across GS and UKB while `V_C` can differ                              | Supplementary Data                                                            |


Fig. 1 is a schematic with no data. 

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

