import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Parse message
pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\]'
extracted = df['message'].str.extract(pattern)
df['prefix'] = extracted[0].fillna('none')
df['event'] = extracted[1].fillna('').str.strip()
bracket = extracted[2].fillna('')

parts = bracket.str.extract(r'([A-Za-z\-]+)\s+(REDACTED|[0-9\.]+)\s*([A-Za-z0-9/%]+)?')
df['b_module'] = parts[0]
df['b_val'] = pd.to_numeric(parts[1], errors='coerce')
df['is_redacted'] = (parts[1] == 'REDACTED').astype(int)

df['mod_mismatch'] = (df['module'] != df['b_module']).astype(int)
df['read_diff'] = (df['reading'] - df['b_val']).abs()
df['read_mismatch'] = (df['read_diff'] > 0.05).astype(int)

# Check all rows where mod_mismatch == 1
print("mod_mismatch == 1 fault rate:")
print(df[df['mod_mismatch'] == 1]['is_true_fault'].value_counts())

# What about mod_mismatch == 1 AND status / prefix?
print(df[df['mod_mismatch'] == 1].groupby(['is_true_fault'])['message'].head(5))

# Let's inspect the 45 cases where mod_mismatch == 1 but is_true_fault == 0!
print("\n--- 45 cases of mod_mismatch == 1 but is_true_fault == 0 ---")
mm_f0 = df[(df['mod_mismatch'] == 1) & (df['is_true_fault'] == 0)]
print(mm_f0[['log_id', 'module', 'status', 'reading', 'prefix', 'event', 'b_module', 'b_val']].head(25))
