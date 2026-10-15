import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Event module mapping
event_module = {
    'O2 partial pressure falling': 'Life-Support',
    'seal breach suspected': 'Life-Support',
    'scrubber saturation': 'Life-Support',
    'breaker tripped twice': 'Lab-Power',
    'cascade brownout detected': 'Lab-Power',
    'overvolt on rail B': 'Lab-Power',
    'bus arbitration deadlock': 'Notice-Bus',
    'heartbeat missing from 3 nodes': 'Notice-Bus',
    'queue overflow, packets dropped': 'Notice-Bus',
    'carrier drift beyond tolerance': 'Antenna',
    'dish actuator stalled': 'Antenna',
    'phase lock lost': 'Antenna'
}

# Unit module mapping
unit_module = {
    '%O2': 'Life-Support',
    'V': 'Lab-Power',
    'dBm': 'Antenna',
    'msg/s': 'Notice-Bus'
}

# Parse message
pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
ext = df['message'].str.extract(pattern)
df['prefix'] = ext[0].fillna('none')
df['event'] = ext[1].fillna('').str.strip()
df['b_mod'] = ext[2]
df['b_val'] = pd.to_numeric(ext[3], errors='coerce')
df['b_unit'] = ext[4]

df['expected_module_by_event'] = df['event'].map(event_module)
df['event_mod_mismatch'] = (df['expected_module_by_event'].notna()) & (df['expected_module_by_event'] != df['module'])

df['expected_module_by_unit'] = df['b_unit'].map(unit_module)
df['unit_mod_mismatch'] = (df['expected_module_by_unit'].notna()) & (df['expected_module_by_unit'] != df['module'])

print("--- Event module mismatch vs is_true_fault ---")
print(pd.crosstab(df['event_mod_mismatch'], df['is_true_fault'], margins=True))

print("\n--- Unit module mismatch vs is_true_fault ---")
print(pd.crosstab(df['unit_mod_mismatch'], df['is_true_fault'], margins=True))

# Check nominal bounds
nominal = {
    'Lab-Power': (11.0, 13.5),
    'Life-Support': (19.5, 23.5),
    'Antenna': (35.0, 50.0),
    'Notice-Bus': (260.0, 380.0)
}

df['out_of_nominal'] = 0
for mod, (low, high) in nominal.items():
    mask = df['module'] == mod
    df.loc[mask, 'out_of_nominal'] = ((df.loc[mask, 'reading'] < low) | (df.loc[mask, 'reading'] > high)).astype(int)

print("\n--- out_of_nominal vs is_true_fault ---")
print(pd.crosstab(df['out_of_nominal'], df['is_true_fault'], margins=True))

# Check interaction: out_of_nominal and status
print("\n--- out_of_nominal & status vs is_true_fault ---")
print(df.groupby(['out_of_nominal', 'status'])['is_true_fault'].agg(['count', 'mean']))

# Check interaction: out_of_nominal & prefix
print("\n--- out_of_nominal & prefix vs is_true_fault ---")
print(df.groupby(['out_of_nominal', 'prefix'])['is_true_fault'].agg(['count', 'mean']))
