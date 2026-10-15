import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

df_all = pd.concat([train, val, test], ignore_index=True)
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)

# Autocorrelation / lag of is_true_fault
train_val = df_all[df_all['is_true_fault'].notna()].copy()
train_val['prev_fault'] = train_val['is_true_fault'].shift(1)
print("Autocorrelation of is_true_fault:")
print(pd.crosstab(train_val['prev_fault'], train_val['is_true_fault'], normalize='index'))

# Check by node:
train_val['prev_fault_node'] = train_val.groupby('node')['is_true_fault'].shift(1)
print("\nAutocorrelation by node:")
print(pd.crosstab(train_val['prev_fault_node'], train_val['is_true_fault'], normalize='index'))

# Check by module:
train_val['prev_fault_mod'] = train_val.groupby('module')['is_true_fault'].shift(1)
print("\nAutocorrelation by module:")
print(pd.crosstab(train_val['prev_fault_mod'], train_val['is_true_fault'], normalize='index'))

# Autocorrelation is almost flat (0.21 vs 0.21), so each log line is an independent observation!
