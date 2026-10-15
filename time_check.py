import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

df_all = pd.concat([train, val, test], ignore_index=True)
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)

print("Date range in train+val+test:")
print(f"Min: {df_all['timestamp'].min()}, Max: {df_all['timestamp'].max()}")

# Are train, val, and test interleaved in time or sequential?
df_all['split'] = 'test'
df_all.loc[df_all['log_id'].isin(train['log_id']), 'split'] = 'train'
df_all.loc[df_all['log_id'].isin(val['log_id']), 'split'] = 'val'

print("\nSplit distribution over time:")
print(df_all.groupby([df_all['timestamp'].dt.day, 'split'])['log_id'].count().unstack())

# They are completely interleaved across all 22 days!
# Let's check time delta to previous log on same node / same module
print("\nTime deltas between consecutive logs:")
print(df_all['timestamp'].diff().describe())
