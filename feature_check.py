import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Let's inspect reading values for each module
for mod in df['module'].unique():
    m_df = df[df['module'] == mod]
    print(f"\n================ Module: {mod} ================")
    print("Quantiles of reading overall:")
    print(m_df['reading'].quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))
    
    # Check if there are distinct reading thresholds
    # Let's see is_true_fault by reading quantiles
    m_df['reading_bin'] = pd.qcut(m_df['reading'], q=10, duplicates='drop')
    print("\nis_true_fault by reading decile:")
    print(m_df.groupby('reading_bin')['is_true_fault'].agg(['count', 'mean']))

# What about delay_ms and retry_count?
print("\n================ delay_ms vs is_true_fault ================")
df['delay_bin'] = pd.qcut(df['delay_ms'], q=10)
print(df.groupby('delay_bin')['is_true_fault'].agg(['count', 'mean']))

print("\n================ retry_count vs is_true_fault ================")
print(df.groupby('retry_count')['is_true_fault'].agg(['count', 'mean']))

print("\n================ mode vs is_true_fault ================")
print(df.groupby('mode')['is_true_fault'].agg(['count', 'mean']))

print("\n================ node vs is_true_fault ================")
print(df.groupby('node')['is_true_fault'].agg(['count', 'mean']))
