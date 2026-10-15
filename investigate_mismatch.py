import pandas as pd
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Let's inspect rows with reading mismatches
pattern = r'\[(.*?)\s+([0-9\.]+)\s*(.*?)\]'
ext = df['message'].str.extract(pattern)
df['b_mod'] = ext[0]
df['b_val'] = pd.to_numeric(ext[1], errors='coerce')
df['diff'] = (df['reading'] - df['b_val']).abs()

mismatch = df[df['diff'] > 0.05].copy()
print(f"Total reading mismatches: {len(mismatch)}")
print(mismatch[['log_id', 'module', 'status', 'reading', 'b_val', 'is_true_fault', 'message']].head(20))
print("\nReading mismatch fault distribution:")
print(mismatch['is_true_fault'].value_counts())
