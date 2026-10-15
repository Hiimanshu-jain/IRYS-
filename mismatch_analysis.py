import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
ext = df['message'].str.extract(pattern)
df['prefix'] = ext[0].fillna('none')
df['event'] = ext[1].fillna('').str.strip()
df['b_mod'] = ext[2]
df['b_val'] = pd.to_numeric(ext[3], errors='coerce')

# Let's inspect the 156 cases where diff > 0.05 but is_true_fault == 0
diff_0 = df[(df['reading'] - df['b_val']).abs() > 0.05]
diff_0_f0 = diff_0[diff_0['is_true_fault'] == 0]
diff_0_f1 = diff_0[diff_0['is_true_fault'] == 1]

print("=== When reading mismatch but is_true_fault == 0 (156 cases) ===")
print(diff_0_f0[['status', 'reading', 'b_val', 'prefix', 'event', 'module', 'b_mod']].head(20))

print("\n=== When reading mismatch and is_true_fault == 1 (337 cases) ===")
print(diff_0_f1[['status', 'reading', 'b_val', 'prefix', 'event', 'module', 'b_mod']].head(20))
