import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

event_module_map = {
    'O2 partial pressure falling': 'Life-Support',
    'seal breach suspected': 'Life-Support',
    'scrubber saturation': 'Life-Support',
    'breaker tripped twice': 'Lab-Power',
    'cascade brownout detected': 'Lab-Power',
    'overvolt on rail B': 'Lab-Power',
    'bus arbitration deadlock': 'Notice-Bus',
    'heartbeat missing from 3 nodes': 'Notice-Bus',
    'queue overflow, packets dropped': 'Notice-Bus',
    'carrier drift beyond tolerance': 'Antenna',
    'dish actuator stalled': 'Antenna',
    'phase lock lost': 'Antenna'
}

unit_module_map = {
    '%O2': 'Life-Support',
    'V': 'Lab-Power',
    'dBm': 'Antenna',
    'msg/s': 'Notice-Bus'
}

nominal_bounds = {
    'Lab-Power': (11.3, 12.7, 12.0, 0.7),
    'Life-Support': (20.0, 22.3, 21.15, 1.15),
    'Antenna': (37.0, 48.0, 42.5, 5.5),
    'Notice-Bus': (275.0, 345.0, 310.0, 35.0)
}

serious_events = set(event_module_map.keys())

def build_features(df, tfidf=None, svd=None, is_train=False):
    feats = pd.DataFrame(index=df.index)
    
    # 1. Parse message
    pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
    ext = df['message'].str.extract(pattern)
    prefix = ext[0].fillna('none')
    event = ext[1].fillna('').str.strip()
    b_mod = ext[2].fillna('unknown')
    b_val = pd.to_numeric(ext[3], errors='coerce')
    b_unit = ext[4].fillna('unknown')
    is_redacted = (ext[3] == 'REDACTED').astype(int)
    
    # 2. Raw numerics
    feats['reading'] = df['reading']
    feats['b_val'] = b_val
    feats['is_redacted'] = is_redacted
    feats['reading_is_na'] = df['reading'].isna().astype(int)
    
    feats['delay_ms'] = df['delay_ms']
    feats['log_delay_ms'] = np.log1p(df['delay_ms'])
    feats['retry_count'] = df['retry_count']
    feats['has_retry'] = (df['retry_count'] > 0).astype(int)
    feats['delay_per_retry'] = df['delay_ms'] / (df['retry_count'] + 1)
    
    # 3. Categoricals
    feats['module'] = df['module'].astype('category')
    feats['b_mod'] = b_mod.astype('category')
    feats['mode'] = df['mode'].astype('category')
    feats['node'] = df['node'].astype('category')
    feats['status'] = df['status'].astype('category')
    feats['prefix'] = prefix.astype('category')
    feats['event'] = event.astype('category')
    feats['b_unit'] = b_unit.astype('category')
    
    # 4. Mismatch & deception features
    feats['mod_mismatch'] = (df['module'] != b_mod).astype(int)
    feats['unit_mismatch'] = df.apply(lambda r: 1 if unit_module_map.get(b_unit.loc[r.name], r['module']) != r['module'] else 0, axis=1)
    feats['event_mismatch'] = df.apply(lambda r: 1 if event.loc[r.name] in event_module_map and event_module_map[event.loc[r.name]] != r['module'] else 0, axis=1)
    
    feats['val_diff'] = (df['reading'] - b_val).abs()
    feats['val_mismatch'] = (feats['val_diff'] > 0.05).astype(int)
    feats['val_ratio'] = df['reading'] / (b_val + 1e-4)
    
    # 5. Nominal physics bounds
    low_b = df['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[0])
    high_b = df['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[1])
    center_b = df['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[2])
    scale_b = df['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[3])
    
    feats['dist_below_nominal'] = np.maximum(0, low_b - df['reading'])
    feats['dist_above_nominal'] = np.maximum(0, df['reading'] - high_b)
    feats['is_out_of_nominal'] = ((df['reading'] < low_b) | (df['reading'] > high_b)).astype(int)
    feats['reading_norm_dist'] = (df['reading'] - center_b).abs() / scale_b
    
    # Also for b_val
    feats['b_dist_below_nominal'] = np.maximum(0, low_b - b_val)
    feats['b_dist_above_nominal'] = np.maximum(0, b_val - high_b)
    feats['b_is_out_of_nominal'] = ((b_val < low_b) | (b_val > high_b)).astype(int)
    
    # Did NINE sanitize the reading? (reading is out of nominal, but b_val is in nominal!)
    feats['nine_sanitized_reading'] = (feats['is_out_of_nominal'] == 1) & (feats['b_is_out_of_nominal'] == 0)
    feats['nine_faked_reading'] = (feats['is_out_of_nominal'] == 0) & (feats['b_is_out_of_nominal'] == 1)
    
    # 6. Deception & conflict indicators
    is_calm_prefix = prefix.isin(['all is well', 'nothing to see', 'crew need not worry'])
    is_alarm_status = df['status'].isin(['ERROR', 'CRIT'])
    is_calm_status = df['status'] == 'OK'
    is_serious_ev = event.isin(serious_events)
    
    # NINE calming crew when there is a real fault
    feats['deceptive_calm'] = (is_calm_prefix & (feats['is_out_of_nominal'] | is_alarm_status | is_serious_ev)).astype(int)
    # NINE raising false alarm
    feats['crying_wolf'] = (is_alarm_status & (feats['is_out_of_nominal'] == 0) & (~is_serious_ev)).astype(int)
    # System nominal but NINE claims reading outside nominal window
    feats['false_outside_claim'] = (event == 'reading outside nominal window') & (feats['is_out_of_nominal'] == 0)
    # System actually outside nominal but NINE claims value logged / periodic report
    feats['hidden_fault_claim'] = (event.isin(['value logged', 'periodic report', 'status changed'])) & (feats['is_out_of_nominal'] == 1)
    
    # Serious event but status is OK
    feats['serious_event_status_ok'] = (is_serious_ev & is_calm_status).astype(int)
    # Harmless event but status is CRIT
    feats['harmless_event_status_crit'] = (event.isin(['periodic report', 'value logged', 'status changed']) & (df['status'] == 'CRIT')).astype(int)
    
    # 7. Time features
    ts = pd.to_datetime(df['timestamp'])
    feats['hour'] = ts.dt.hour
    feats['dayofweek'] = ts.dt.dayofweek
    feats['minute'] = ts.dt.minute
    feats['day'] = ts.dt.day
    
    # 8. TF-IDF on message clean text
    clean_msg = df['message'].str.replace(r'\[.*?\]', '', regex=True).str.replace('NINE:', '', regex=False)
    if is_train:
        tfidf = TfidfVectorizer(max_features=40, ngram_range=(1, 2))
        tfidf_mat = tfidf.fit_transform(clean_msg)
        svd = TruncatedSVD(n_components=8, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
    else:
        tfidf_mat = tfidf.transform(clean_msg)
        svd_mat = svd.transform(tfidf_mat)
        
    for i in range(8):
        feats[f'svd_{i}'] = svd_mat[:, i]
        
    return feats, tfidf, svd

X_train, tfidf, svd = build_features(train, is_train=True)
y_train = train['is_true_fault']

X_val, _, _ = build_features(val, tfidf=tfidf, svd=svd, is_train=False)
y_val = val['is_true_fault']

print(f"Features created: {X_train.shape[1]}")

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

val_prob = model.predict(X_val, num_iteration=model.best_iteration)

best_f1 = 0
best_th = 0.5
for th in np.linspace(0.1, 0.9, 81):
    score = f1_score(y_val, (val_prob >= th).astype(int))
    if score > best_f1:
        best_f1 = score
        best_th = th

print(f"\n>>> LightGBM Best Val F1: {best_f1:.4f} at threshold: {best_th:.2f} <<<")
val_pred = (val_prob >= best_th).astype(int)
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred))

# Check feature importances
importance = pd.Series(model.feature_importance(importance_type='gain'), index=X_train.columns).sort_values(ascending=False)
print("\nTop 20 Features by Gain:")
print(importance.head(20))
