# Reading the results

`decompose()` returns:


| field                                      | meaning                                                                                                                    |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `V_A`, `V_C`                               | genetic and shared-environment variance, as proportions of phenotype variance — of *liability* variance for a binary trait |
| `se_VA_cond`, `se_VC_cond`, `z_VA`, `z_VC` | SE and z at the fixed `w_c_cal`                                                                                            |
| `w_c_cal`                                  | shared-environment decay rate, clipped to [0.01, 0.99]                                                                     |
| `wci_lo`, `wci_hi`                         | 95% profile CI for `w_C`                                                                                                   |


## Notes

- `1 - V_A - V_C` is non-shared environment and noise
- `V_A` or `V_C` exactly 0: the fit is non-negative, so a negative solution is
reported as 0.
- `w_c_cal` at 0.01 or 0.99: clipped to the edge of the grid.
- A CI covering 0.5: interpret the split with care — at `w_C = 0.5` the two
components decay alike.
- The SEs hold `w_C` fixed. To see the rest of the uncertainty, refit across the  
CI.
- Warning `Sigma_hat not PSD ...; projecting to nearest PSD`: pairs overlap
heavily or the signal is weak, and the SEs are optimistic.

