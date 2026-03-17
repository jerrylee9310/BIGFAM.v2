"""
Slope Test Power Measurement - WLS + Jensen Effect

This script tests the effect of Weighted Least Squares (WLS) on slope test power
under heterogeneous SE across DORs (mimicking real data where 1st degree has
more pairs). The output is the figure-ready input file consumed by the
inverse-scale power plotting scripts.

WLS should show advantage when:
- SE varies across degrees (e.g., SE_d1 < SE_d2 < SE_d3)
- OLS gives equal weight to all points, but WLS up-weights precise estimates

Comparison:
- known_var (Jensen + WLS): Uses SE information for both correction and weighting
- resample: Uses bootstrap, ignores heterogeneous SE
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from bigfam import BIGFAM


SIGNIFICANCE_ALIASES = {
    "high": "fast",
    "fast": "fast",
    "low": "slow",
    "slow": "slow",
    "similar": "similar",
}


def canonical_significance(value: str) -> str:
    return SIGNIFICANCE_ALIASES.get(str(value).lower(), str(value).lower())


def is_rejection(value: str) -> bool:
    return canonical_significance(value) in {"fast", "slow"}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def measure_power_wls():
    """
    Test WLS effect with heterogeneous SE across degrees.
    
    Real data scenario: 
    - 1st degree: Many pairs → small SE
    - 2nd degree: Moderate pairs → moderate SE  
    - 3rd degree: Few pairs → large SE
    """
    # Parameters
    VG = 0.5
    VS = 0.1
    WS_LIST = [
        0.1, 0.2, 0.3, 0.4,
        0.5,
        0.6, 0.7, 0.8, 0.9
    ]
    # WS_LIST = [
    #     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
    #     2.0,  # null hypothesis
    #     2.1, 2.3, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0
    # ]
    
    # SE heterogeneity patterns (realistic scenarios)
    # Format: [SE_d1, SE_d2, SE_d3] where d1 has most pairs
    SE_PATTERNS = {
        'uniform_low': [0.005, 0.005, 0.005],      # Baseline: All low SE (no WLS advantage)
        'uniform_mid': [0.01, 0.01, 0.01],         # All moderate SE
        'uniform_high': [0.05, 0.05, 0.05],        # All high SE
        'realistic_mild': [0.005, 0.01, 0.02],     # Mild heterogeneity
        'realistic_strong': [0.005, 0.02, 0.05],   # Strong heterogeneity (real data-like)
        'extreme': [0.001, 0.01, 0.1],             # Extreme heterogeneity
    }
    
    N_ITER = 1000
    DOR_LIST = [1, 2, 3]
    
    results = []
    model = BIGFAM()
    
    print(f"Starting WLS Power Measurement (N={N_ITER} per condition)...")
    print(f"Testing {len(WS_LIST)} ws values x {len(SE_PATTERNS)} SE patterns")
    
    for pattern_name, se_list in SE_PATTERNS.items():
        print(f"\n[SE Pattern: {pattern_name}] SE = {se_list}")
        
        for ws_inv in WS_LIST:
            ws = 1/ws_inv
            print(f"  Testing ws={ws}...")
            
            for i in tqdm(range(N_ITER), leave=False):
                # Simulate with heterogeneous SE
                df_sim = model.simulate_frreg_results(
                    Vg=VG,
                    Vs=VS,
                    ws=ws,
                    slope_se=se_list,  # List of SE per DOR
                    dor_list=DOR_LIST,
                    random_seed=42 + i + int(ws*100) + hash(pattern_name) % 10000
                )
                
                # Test 'known_var' method (Jensen + WLS)
                res_wls = model.run_slope_test(method='known_var')
                
                # Test 'resample' method
                res_resample = model.run_slope_test(method='resample', n_bootstrap=1000)
                
                # Record results
                results.append({
                    'ws': ws,
                    'se_pattern': pattern_name,
                    'se_d1': se_list[0],
                    'se_d2': se_list[1],
                    'se_d3': se_list[2],
                    'se_ratio': se_list[2] / se_list[0],  # Heterogeneity measure
                    'iter': i,
                    'method': 'Jensen_WLS',
                    'significance': canonical_significance(res_wls['significance']),
                    'slope': res_wls['slope'],
                    'se': res_wls['slope_se'],
                    'slope_lower': res_wls['slope_lower'],
                    'slope_upper': res_wls['slope_upper']
                })
                
                results.append({
                    'ws': ws,
                    'se_pattern': pattern_name,
                    'se_d1': se_list[0],
                    'se_d2': se_list[1],
                    'se_d3': se_list[2],
                    'se_ratio': se_list[2] / se_list[0],
                    'iter': i,
                    'method': 'resample',
                    'significance': canonical_significance(res_resample['significance']),
                    'slope': res_resample['slope'],
                    'se': res_resample['slope_se'],
                    'slope_lower': res_resample['slope_lower'],
                    'slope_upper': res_resample['slope_upper']
                })
    
    df_results = pd.DataFrame(results)
    
    # Keep simulation outputs inside the analysis bundle.
    data_dir = get_repo_root() / "data/simulation"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "power_wls_inv.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    
    # Summary
    print("\n--- Summary (Rejection Rate by SE Pattern) ---")
    def calc_rejection_rate(group):
        rejections = group["significance"].map(is_rejection).sum()
        return rejections / len(group)
    
    # Summary by pattern and method (for ws != 2, i.e., power)
    power_df = df_results[df_results['ws'] != 2.0]
    power_summary = power_df.groupby(['se_pattern', 'method']).apply(calc_rejection_rate).reset_index(name='power')
    print("\n[POWER (ws != 2)]")
    print(power_summary.pivot(index='se_pattern', columns='method', values='power').to_string())
    
    # Summary for Type I Error (ws = 2)
    t1e_df = df_results[df_results['ws'] == 2.0]
    t1e_summary = t1e_df.groupby(['se_pattern', 'method']).apply(calc_rejection_rate).reset_index(name='type1_error')
    print("\n[TYPE 1 ERROR (ws = 2)]")
    print(t1e_summary.pivot(index='se_pattern', columns='method', values='type1_error').to_string())




if __name__ == "__main__":
    import sys
    
    measure_power_wls()
