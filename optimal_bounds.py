import pandas as pd
import numpy as np

df = pd.concat([pd.read_csv('train.csv'), pd.read_csv('val.csv')], ignore_index=True)

# For each module, let's see how well different lower and upper bounds separate is_true_fault == 1 vs 0
results = []
for mod in ['Lab-Power', 'Life-Support', 'Antenna', 'Notice-Bus']:
    sub = df[df['module'] == mod].copy()
    
    # Try combinations of low and high bounds
    low_candidates = np.percentile(sub['reading'].dropna(), np.linspace(1, 20, 20))
    high_candidates = np.percentile(sub['reading'].dropna(), np.linspace(80, 99, 20))
    
    best_corr = 0
    best_bounds = None
    
    for l in low_candidates:
        for h in high_candidates:
            is_out = ((sub['reading'] < l) | (sub['reading'] > h)).astype(int)
            corr = np.corrcoef(is_out, sub['is_true_fault'])[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_bounds = (l, h)
                
    print(f"Module {mod:15s}: Best Bounds = [{best_bounds[0]:.2f}, {best_bounds[1]:.2f}], Correlation = {best_corr:.4f}")
