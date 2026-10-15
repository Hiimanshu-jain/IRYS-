import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}")

# Parse message: "NINE: [prefix] [event] [bracket_module bracket_reading bracket_unit]"
# e.g., "NINE: logging. periodic report [Notice-Bus 463.24 msg/s]"
def parse_message(df):
    pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\]'
    
    extracted = df['message'].str.extract(pattern)
    df['msg_prefix'] = extracted[0].fillna('none')
    df['msg_event'] = extracted[1].fillna('')
    bracket_content = extracted[2].fillna('')
    
    # Split bracket content: module and reading/unit
    bracket_parts = bracket_content.str.extract(r'([A-Za-z\-]+)\s+(REDACTED|[0-9\.]+)\s*([A-Za-z0-9/%]+)?')
    df['bracket_module'] = bracket_parts[0]
    df['bracket_val'] = pd.to_numeric(bracket_parts[1], errors='coerce')
    df['bracket_unit'] = bracket_parts[2]
    df['is_redacted'] = bracket_content.str.contains('REDACTED')
    
    # Mismatch features
    df['module_mismatch'] = (df['module'] != df['bracket_module']).astype(int)
    df['reading_diff'] = np.abs(df['reading'] - df['bracket_val'])
    df['reading_mismatch'] = (df['reading_diff'] > 0.01).astype(int)
    
    return df

train = parse_message(train)
val = parse_message(val)

print("\n--- Module Mismatch vs is_true_fault ---")
print(pd.crosstab(train['module_mismatch'], train['is_true_fault'], normalize='index'))

print("\n--- Reading Mismatch vs is_true_fault ---")
print(pd.crosstab(train['reading_mismatch'], train['is_true_fault'], normalize='index'))

print("\n--- Either Mismatch vs is_true_fault ---")
mismatch = (train['module_mismatch'] == 1) | (train['reading_mismatch'] == 1)
print(pd.crosstab(mismatch, train['is_true_fault']))

print("\n--- Prefix vs is_true_fault ---")
print(pd.crosstab(train['msg_prefix'], train['is_true_fault'], normalize='index'))

print("\n--- Event vs is_true_fault ---")
print(pd.crosstab(train['msg_event'], train['is_true_fault'], normalize='index'))

print("\n--- Status vs is_true_fault ---")
print(pd.crosstab(train['status'], train['is_true_fault'], normalize='index'))

print("\n--- Nominal range by module for true vs false ---")
for mod in train['module'].unique():
    sub = train[train['module'] == mod]
    print(f"\nModule: {mod}")
    print("True fault 0 reading stats:")
    print(sub[sub['is_true_fault'] == 0]['reading'].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]))
    print("True fault 1 reading stats:")
    print(sub[sub['is_true_fault'] == 1]['reading'].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]))
