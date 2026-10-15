import pandas as pd
import numpy as np
import re

val = pd.read_csv('val.csv')
train = pd.read_csv('train.csv')

# Let's inspect rows with is_true_fault == 1 where reading is perfectly normal and there is NO mismatch!
pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
ext = val['message'].str.extract(pattern)
val['prefix'] = ext[0].fillna('none')
val['event'] = ext[1].fillna('').str.strip()
val['b_mod'] = ext[2]
val['b_val'] = pd.to_numeric(ext[3], errors='coerce')

val['mod_mismatch'] = (val['module'] != val['b_mod']).astype(int)
val['val_diff'] = (val['reading'] - val['b_val']).abs()
val['val_mismatch'] = (val['val_diff'] > 0.05).astype(int)

# Nominal bounds
nominal = {
    'Lab-Power': (11.3, 12.7),
    'Life-Support': (20.0, 22.3),
    'Antenna': (37.0, 48.0),
    'Notice-Bus': (275.0, 345.0)
}
val['out_of_nominal'] = 0
for mod, (low, high) in nominal.items():
    mask = val['module'] == mod
    val.loc[mask, 'out_of_nominal'] = ((val.loc[mask, 'reading'] < low) | (val.loc[mask, 'reading'] > high)).astype(int)

# Filter: true fault == 1, but out_of_nominal == 0 and mod_mismatch == 0 and val_mismatch == 0
hard_cases = val[(val['is_true_fault'] == 1) & (val['out_of_nominal'] == 0) & (val['mod_mismatch'] == 0) & (val['val_mismatch'] == 0)]
print(f"Total hard true faults in val: {len(hard_cases)} out of {val['is_true_fault'].sum()}")
print(hard_cases[['log_id', 'module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message']].head(25))

# And filter: true fault == 0, but out_of_nominal == 1 or mod_mismatch == 1 or val_mismatch == 1
hard_false = val[(val['is_true_fault'] == 0) & ((val['out_of_nominal'] == 1) | (val['mod_mismatch'] == 1) | (val['val_mismatch'] == 1))]
print(f"\nTotal hard false alarms in val: {len(hard_false)} out of {(val['is_true_fault'] == 0).sum()}")
print(hard_false[['log_id', 'module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message']].head(25))
