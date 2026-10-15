import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Parse full message
# format: NINE: [prefix.] event [bracket_module reading/REDACTED unit]
pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
ext = df['message'].str.extract(pattern)
df['prefix'] = ext[0].fillna('none')
df['event'] = ext[1].fillna('').str.strip()
df['b_mod'] = ext[2].fillna('unknown')
df['b_val'] = pd.to_numeric(ext[3], errors='coerce')
df['b_unit'] = ext[4].fillna('unknown')
df['is_redacted'] = (ext[3] == 'REDACTED').astype(int)

# Check correlation of various candidate features with is_true_fault
df['mod_mismatch'] = (df['module'] != df['b_mod']).astype(int)
df['val_diff'] = (df['reading'] - df['b_val']).abs()
df['val_mismatch'] = (df['val_diff'] > 0.01).astype(int)

# Check if there is a difference between module and bracket unit
unit_map = {'Life-Support': '%O2', 'Lab-Power': 'V', 'Antenna': 'dBm', 'Notice-Bus': 'msg/s'}
df['expected_unit'] = df['module'].map(unit_map)
df['unit_mismatch'] = (df['b_unit'] != df['expected_unit']).astype(int)

# Check nominal bounds:
# Let's inspect reading distribution for each module when is_true_fault == 1 and mod_mismatch == 0 and val_mismatch == 0
sub_clean = df[(df['mod_mismatch'] == 0) & (df['val_mismatch'] == 0) & (df['is_redacted'] == 0)]
print(f"Clean rows: {len(sub_clean)}")

for mod in ['Lab-Power', 'Life-Support', 'Antenna', 'Notice-Bus']:
    m = sub_clean[sub_clean['module'] == mod]
    print(f"\n--- {mod} (clean rows) ---")
    f0 = m[m['is_true_fault'] == 0]['reading']
    f1 = m[m['is_true_fault'] == 1]['reading']
    print(f"f0: min={f0.min():.2f}, p5={f0.quantile(0.05):.2f}, p50={f0.median():.2f}, p95={f0.quantile(0.95):.2f}, max={f0.max():.2f}")
    print(f"f1: min={f1.min():.2f}, p5={f1.quantile(0.05):.2f}, p50={f1.median():.2f}, p95={f1.quantile(0.95):.2f}, max={f1.max():.2f}")
