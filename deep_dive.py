import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Function to parse messages
def parse_msg(df):
    pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\]'
    extracted = df['message'].str.extract(pattern)
    df['prefix'] = extracted[0].fillna('none')
    df['event'] = extracted[1].fillna('').str.strip()
    bracket = extracted[2].fillna('')
    
    parts = bracket.str.extract(r'([A-Za-z\-]+)\s+(REDACTED|[0-9\.]+)\s*([A-Za-z0-9/%]+)?')
    df['b_module'] = parts[0]
    df['b_reading'] = pd.to_numeric(parts[1], errors='coerce')
    df['b_unit'] = parts[2]
    df['b_redacted'] = (parts[1] == 'REDACTED').astype(int)
    
    df['mod_mismatch'] = (df['module'] != df['b_module']).astype(int)
    df['read_diff'] = (df['reading'] - df['b_reading']).abs()
    df['read_mismatch'] = (df['read_diff'] > 0.01).astype(int)
    df['is_missing_reading'] = df['reading'].isna().astype(int)
    
    return df

df = parse_msg(df)

print("=== Cross tabs with is_true_fault ===")
print("Module mismatch:")
print(pd.crosstab(df['mod_mismatch'], df['is_true_fault'], margins=True))

print("\nReading mismatch (when not nan):")
print(pd.crosstab(df['read_mismatch'], df['is_true_fault'], margins=True))

print("\nRedacted in bracket:")
print(pd.crosstab(df['b_redacted'], df['is_true_fault'], margins=True))

print("\nMissing reading:")
print(pd.crosstab(df['is_missing_reading'], df['is_true_fault'], margins=True))

# What if module mismatch OR reading mismatch OR redacted?
mismatch_any = (df['mod_mismatch'] == 1) | (df['read_mismatch'] == 1) | (df['b_redacted'] == 1)
print("\nAny Mismatch or Redacted vs is_true_fault:")
print(pd.crosstab(mismatch_any, df['is_true_fault'], margins=True))

# Check the remaining rows where mismatch_any == False
clean = df[~mismatch_any].copy()
print(f"\nRemaining clean rows: {len(clean)}, of which true fault: {clean['is_true_fault'].sum()} ({clean['is_true_fault'].mean():.3f})")

# Let's inspect clean rows where is_true_fault == 1
print("\nSample clean rows with is_true_fault == 1:")
print(clean[clean['is_true_fault'] == 1][['module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'prefix', 'event']].head(25))

print("\nEvent vs is_true_fault in clean rows:")
print(pd.crosstab(clean['event'], clean['is_true_fault'], margins=True))

print("\nPrefix vs is_true_fault in clean rows:")
print(pd.crosstab(clean['prefix'], clean['is_true_fault'], margins=True))

print("\nStatus vs is_true_fault in clean rows:")
print(pd.crosstab(clean['status'], clean['is_true_fault'], margins=True))
