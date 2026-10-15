import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
ext = df['message'].str.extract(pattern)
df['b_val'] = pd.to_numeric(ext[3], errors='coerce')

# What is the distribution of b_val when reading is abnormal?
# When NINE forges a reading, what numbers does it use?
mismatch = df[(df['reading'] - df['b_val']).abs() > 0.05]
for mod in df['module'].unique():
    m = mismatch[mismatch['module'] == mod]
    print(f"\nModule {mod} forged b_val range:")
    print(f"min: {m['b_val'].min()}, max: {m['b_val'].max()}, mean: {m['b_val'].mean()}")

# Look at normal ranges in the messages:
# Look at messages that say "reading outside nominal window [Module X unit]"
out_msg = df[df['message'].str.contains('reading outside nominal window', case=False)]
print("\n--- Rows where message explicitly says 'reading outside nominal window' ---")
for mod in df['module'].unique():
    sub = out_msg[out_msg['module'] == mod]
    print(f"\nModule {mod} reading when NINE says 'reading outside nominal window':")
    print(sub['reading'].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
