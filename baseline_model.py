import pandas as pd
import numpy as np
import re
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

def extract_features(df):
    feats = pd.DataFrame(index=df.index)
    
    # Message parsing
    pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\]'
    extracted = df['message'].str.extract(pattern)
    prefix = extracted[0].fillna('none')
    event = extracted[1].fillna('').str.strip()
    bracket = extracted[2].fillna('')
    
    parts = bracket.str.extract(r'([A-Za-z\-]+)\s+(REDACTED|[0-9\.]+)\s*([A-Za-z0-9/%]+)?')
    b_module = parts[0].fillna('unknown')
    b_reading = pd.to_numeric(parts[1], errors='coerce')
    b_unit = parts[2].fillna('unknown')
    b_redacted = (parts[1] == 'REDACTED').astype(int)
    
    feats['reading'] = df['reading']
    feats['b_reading'] = b_reading
    feats['is_missing_reading'] = df['reading'].isna().astype(int)
    feats['b_redacted'] = b_redacted
    feats['reading_diff'] = (df['reading'] - b_reading).abs()
    feats['reading_mismatch'] = (feats['reading_diff'] > 0.01).astype(int)
    feats['reading_ratio'] = df['reading'] / (b_reading + 1e-5)
    
    feats['mod_mismatch'] = (df['module'] != b_module).astype(int)
    
    feats['delay_ms'] = df['delay_ms']
    feats['retry_count'] = df['retry_count']
    
    # Categoricals
    feats['module'] = df['module'].astype('category')
    feats['b_module'] = b_module.astype('category')
    feats['mode'] = df['mode'].astype('category')
    feats['node'] = df['node'].astype('category')
    feats['status'] = df['status'].astype('category')
    feats['prefix'] = prefix.astype('category')
    feats['event'] = event.astype('category')
    
    # Text length / char features
    feats['msg_len'] = df['message'].str.len()
    
    # Specific keywords in message
    for kw in ['all is well', 'nothing to see', 'crew need not worry', 'logging', 'noted',
               'queue overflow', 'heartbeat missing', 'stalled', 'deadlock', 'breach',
               'brownout', 'overvolt', 'tripped', 'falling', 'drift', 'phase lock',
               'saturation', 'drill', 'calibration', 'self-test', 'warming up', 'outside nominal']:
        feats[f'has_{kw}'] = df['message'].str.contains(kw, case=False).astype(int)
        
    # Module specific normalized reading
    # Mean and std by module
    norm_dict = {
        'Lab-Power': (12.2, 1.2),
        'Life-Support': (21.2, 1.8),
        'Antenna': (43.5, 9.0),
        'Notice-Bus': (319.0, 60.0)
    }
    feats['reading_zscore'] = df.apply(
        lambda r: (r['reading'] - norm_dict.get(r['module'], (0, 1))[0]) / norm_dict.get(r['module'], (0, 1))[1],
        axis=1
    )
    feats['reading_abs_zscore'] = feats['reading_zscore'].abs()
    
    # Interaction between status and reading_abs_zscore
    # Interaction between prefix and status
    feats['is_calm_prefix'] = prefix.isin(['all is well', 'nothing to see', 'crew need not worry']).astype(int)
    feats['is_alarm_status'] = df['status'].isin(['ERROR', 'CRIT']).astype(int)
    feats['is_calm_status'] = (df['status'] == 'OK').astype(int)
    
    feats['calm_prefix_with_alarm_status'] = feats['is_calm_prefix'] * feats['is_alarm_status']
    feats['calm_prefix_with_calm_status'] = feats['is_calm_prefix'] * feats['is_calm_status']
    feats['alarm_status_extreme_reading'] = feats['is_alarm_status'] * (feats['reading_abs_zscore'] > 2.0).astype(int)
    feats['calm_status_extreme_reading'] = feats['is_calm_status'] * (feats['reading_abs_zscore'] > 2.0).astype(int)
    
    return feats

X_train = extract_features(train)
y_train = train['is_true_fault']

X_val = extract_features(val)
y_val = val['is_true_fault']

# Convert categories to category dtype for HistGradientBoosting
cat_cols = ['module', 'b_module', 'mode', 'node', 'status', 'prefix', 'event']

clf = HistGradientBoostingClassifier(categorical_features=cat_cols, random_state=42, max_iter=300)
clf.fit(X_train, y_train)

val_probs = clf.predict_proba(X_val)[:, 1]

best_f1 = 0
best_th = 0.5
for th in np.linspace(0.1, 0.9, 81):
    preds = (val_probs >= th).astype(int)
    score = f1_score(y_val, preds)
    if score > best_f1:
        best_f1 = score
        best_th = th

print(f"Best Val F1: {best_f1:.4f} at threshold: {best_th:.2f}")
preds = (val_probs >= best_th).astype(int)
print(confusion_matrix(y_val, preds))
print(classification_report(y_val, preds))
