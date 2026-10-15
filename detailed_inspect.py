import pandas as pd
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

print("--- 25 TRUE FAULTS (1) ---")
print(df[df['is_true_fault'] == 1].head(25)[['log_id', 'module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message']])

print("\n--- 25 FALSE (0) ---")
print(df[df['is_true_fault'] == 0].head(25)[['log_id', 'module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message']])
