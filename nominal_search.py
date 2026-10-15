import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Let's see what happens to true fault when reading is outside nominal ranges
# Let's search for exact nominal thresholds for each module
nominal_ranges = {
    'Lab-Power': (11.0, 13.5),
    'Life-Support': (19.5, 23.5),  # or 19.0 to 23.0
    'Antenna': (35.0, 50.0),       # or 32.0 to 52.0
    'Notice-Bus': (250.0, 375.0)   # or 260 to 380
}

# Let's optimize or plot the threshold for each module
for mod in df['module'].unique():
    m = df[df['module'] == mod].copy()
    print(f"\n--- Module: {mod} ---")
    for low in np.linspace(m['reading'].quantile(0.05), m['reading'].quantile(0.20), 10):
        for high in np.linspace(m['reading'].quantile(0.80), m['reading'].quantile(0.95), 10):
            out = (m['reading'] < low) | (m['reading'] > high)
            # check correlation or accuracy/f1
            pass

# Let's inspect rows where reading is outside vs inside
for mod in ['Lab-Power', 'Life-Support', 'Antenna', 'Notice-Bus']:
    sub = df[df['module'] == mod]
    print(f"\n=== {mod} ===")
    print("Fault rate when reading is extreme low/high vs middle:")
    p10 = sub['reading'].quantile(0.12)
    p88 = sub['reading'].quantile(0.88)
    sub['is_extreme'] = (sub['reading'] < p10) | (sub['reading'] > p88)
    print(pd.crosstab(sub['is_extreme'], sub['is_true_fault'], normalize='index'))
    print(f"p10: {p10:.2f}, p88: {p88:.2f}")
