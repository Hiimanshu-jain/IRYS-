import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Let's inspect rows with is_true_fault == 1 vs is_true_fault == 0
# Look at the status, readings, and message
print("=== When is_true_fault == 1 ===")
print(df[df['is_true_fault'] == 1][['module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message']].head(20))

print("\n=== When is_true_fault == 0 ===")
print(df[df['is_true_fault'] == 0][['module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message']].head(20))
