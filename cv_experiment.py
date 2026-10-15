import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

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

# Derived bounds from empirical distribution of forged/nominal readings
nominal_bounds = {
    'Lab-Power': (11.3, 12.7, 12.0, 0.7),
    'Life-Support': (20.0, 22.3, 21.15, 1.15),
    'Antenna': (37.0, 48.0, 42.5, 5.5),
    'Notice-Bus': (275.0, 345.0, 310.0, 35.0)
}

serious_events = set(event_module_map.keys())
harmless_events = {'periodic report', 'value logged', 'status changed', 'scheduled calibration', 'self-test pattern', 'simulated drill event'}

def create_feature_table(df_all):
    # Extract message components
    pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
    ext = df_all['message'].str.extract(pattern)
    prefix = ext[0].fillna('none')
    event = ext[1].fillna('').str.strip()
    b_mod = ext[2].fillna('unknown')
    b_val = pd.to_numeric(ext[3], errors='coerce')
    b_unit = ext[4].fillna('unknown')
    is_redacted = (ext[3] == 'REDACTED').astype(int)
    
    feats = pd.DataFrame(index=df_all.index)
    
    # Readings
    feats['reading'] = df_all['reading']
    feats['b_val'] = b_val
    feats['is_redacted'] = is_redacted
    feats['reading_is_na'] = df_all['reading'].isna().astype(int)
    
    # Delays and Retries
    feats['delay_ms'] = df_all['delay_ms']
    feats['log_delay_ms'] = np.log1p(df_all['delay_ms'])
    feats['retry_count'] = df_all['retry_count']
    feats['has_retry'] = (df_all['retry_count'] > 0).astype(int)
    feats['high_retry'] = (df_all['retry_count'] >= 2).astype(int)
    feats['delay_per_retry'] = df_all['delay_ms'] / (df_all['retry_count'] + 1)
    
    # Differences and mismatches
    feats['mod_mismatch'] = (df_all['module'] != b_mod).astype(int)
    feats['val_diff'] = (df_all['reading'] - b_val).abs()
    feats['val_mismatch'] = (feats['val_diff'] > 0.05).astype(int)
    feats['val_ratio'] = df_all['reading'] / (b_val + 1e-4)
    
    expected_unit = df_all['module'].map(unit_module_map)
    feats['unit_mismatch'] = (b_unit != expected_unit).astype(int)
    
    expected_event_mod = event.map(event_module_map)
    feats['event_mod_mismatch'] = (expected_event_mod.notna() & (expected_event_mod != df_all['module'])).astype(int)
    
    # Physics bounds
    low_b = df_all['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[0])
    high_b = df_all['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[1])
    center_b = df_all['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[2])
    scale_b = df_all['module'].map(lambda m: nominal_bounds.get(m, (0, 0, 0, 1))[3])
    
    feats['dist_below_nominal'] = np.maximum(0, low_b - df_all['reading'])
    feats['dist_above_nominal'] = np.maximum(0, df_all['reading'] - high_b)
    feats['is_out_of_nominal'] = ((df_all['reading'] < low_b) | (df_all['reading'] > high_b)).astype(int)
    feats['reading_norm_dist'] = (df_all['reading'] - center_b).abs() / scale_b
    feats['reading_zscore'] = (df_all['reading'] - center_b) / scale_b
    
    feats['b_dist_below_nominal'] = np.maximum(0, low_b - b_val)
    feats['b_dist_above_nominal'] = np.maximum(0, b_val - high_b)
    feats['b_is_out_of_nominal'] = ((b_val < low_b) | (b_val > high_b)).astype(int)
    
    feats['nine_sanitized_reading'] = (feats['is_out_of_nominal'] == 1) & (feats['b_is_out_of_nominal'] == 0)
    feats['nine_faked_reading'] = (feats['is_out_of_nominal'] == 0) & (feats['b_is_out_of_nominal'] == 1)
    
    # Status
    status_order = {'OK': 0, 'WARN': 1, 'ERROR': 2, 'CRIT': 3}
    feats['status_severity'] = df_all['status'].map(status_order).fillna(0)
    feats['status_is_ok'] = (df_all['status'] == 'OK').astype(int)
    feats['status_is_crit'] = (df_all['status'] == 'CRIT').astype(int)
    feats['status_is_error'] = (df_all['status'] == 'ERROR').astype(int)
    feats['status_is_warn'] = (df_all['status'] == 'WARN').astype(int)
    feats['is_alarm_status'] = df_all['status'].isin(['ERROR', 'CRIT']).astype(int)
    
    # Event types
    feats['is_serious_event'] = event.isin(serious_events).astype(int)
    feats['is_harmless_event'] = event.isin(harmless_events).astype(int)
    
    # Prefix types
    feats['is_calming_prefix'] = prefix.isin(['all is well', 'nothing to see', 'crew need not worry']).astype(int)
    feats['is_neutral_prefix'] = prefix.isin(['logging', 'noted', 'none']).astype(int)
    
    # Deception / Paradox features
    # 1. Calm words while system is actually alarming
    feats['calm_words_alarm_status'] = feats['is_calming_prefix'] * feats['is_alarm_status']
    feats['calm_words_out_of_nominal'] = feats['is_calming_prefix'] * feats['is_out_of_nominal']
    feats['calm_words_serious_event'] = feats['is_calming_prefix'] * feats['is_serious_event']
    
    # 2. Calm status while system is actually out of nominal
    feats['calm_status_out_of_nominal'] = feats['status_is_ok'] * feats['is_out_of_nominal']
    feats['calm_status_serious_event'] = feats['status_is_ok'] * feats['is_serious_event']
    
    # 3. Alarm status but system is completely nominal
    feats['false_alarm_suspect'] = (feats['is_alarm_status']) & (feats['is_out_of_nominal'] == 0) & (feats['is_serious_event'] == 0)
    
    # 4. Redacted message interaction
    feats['redacted_with_alarm'] = is_redacted * feats['is_alarm_status']
    feats['redacted_with_calm'] = is_redacted * feats['is_calming_prefix']
    
    # Time features
    ts = pd.to_datetime(df_all['timestamp'])
    feats['hour'] = ts.dt.hour
    feats['day'] = ts.dt.day
    feats['dayofweek'] = ts.dt.dayofweek
    feats['minute'] = ts.dt.minute
    
    # Text length
    feats['msg_char_count'] = df_all['message'].str.len()
    feats['msg_word_count'] = df_all['message'].str.split().str.len()
    
    # Specific keywords
    for kw in ['queue overflow', 'heartbeat missing', 'stalled', 'deadlock', 'breach',
               'brownout', 'overvolt', 'tripped', 'falling', 'drift', 'phase lock', 'saturation']:
        feats[f'kw_{kw}'] = df_all['message'].str.contains(kw, case=False).astype(int)
        
    feats['module'] = df_all['module'].astype('category')
    feats['b_mod'] = b_mod.astype('category')
    feats['mode'] = df_all['mode'].astype('category')
    feats['node'] = df_all['node'].astype('category')
    feats['status'] = df_all['status'].astype('category')
    feats['prefix'] = prefix.astype('category')
    feats['event'] = event.astype('category')
    feats['b_unit'] = b_unit.astype('category')
            
    return feats

print("Extracting features...")
X_train_df = create_feature_table(train)
y_train_arr = train['is_true_fault'].values

X_val_df = create_feature_table(val)
y_val_arr = val['is_true_fault'].values

# TF-IDF on clean message
clean_train = train['message'].str.replace(r'\[.*?\]', '', regex=True).str.replace('NINE:', '', regex=False)
clean_val = val['message'].str.replace(r'\[.*?\]', '', regex=True).str.replace('NINE:', '', regex=False)

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100)
svd = TruncatedSVD(n_components=12, random_state=42)

tfidf_tr = tfidf.fit_transform(clean_train)
svd_tr = svd.fit_transform(tfidf_tr)
tfidf_va = tfidf.transform(clean_val)
svd_va = svd.transform(tfidf_va)

for i in range(12):
    X_train_df[f'svd_{i}'] = svd_tr[:, i]
    X_val_df[f'svd_{i}'] = svd_va[:, i]

print(f"Total features: {X_train_df.shape[1]}")

# Model 1: LightGBM with 5-fold CV
print("\n--- Training LightGBM on 5-fold CV on train.csv ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(train))
val_preds_lgb = np.zeros(len(val))

cat_cols = ['module', 'b_mod', 'mode', 'node', 'status', 'prefix', 'event', 'b_unit']

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.025,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 20,
    'random_state': 42,
    'verbose': -1
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_df, y_train_arr)):
    X_tr, y_tr = X_train_df.iloc[trn_idx], y_train_arr[trn_idx]
    X_va, y_va = X_train_df.iloc[val_idx], y_train_arr[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=trn_data)
    
    bst = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    oof_lgb[val_idx] = bst.predict(X_va, num_iteration=bst.best_iteration)
    val_preds_lgb += bst.predict(X_val_df, num_iteration=bst.best_iteration) / 5

# Evaluate OOF F1
best_oof_f1 = 0
best_oof_th = 0.5
for th in np.linspace(0.1, 0.9, 81):
    f1 = f1_score(y_train_arr, (oof_lgb >= th).astype(int))
    if f1 > best_oof_f1:
        best_oof_f1 = f1
        best_oof_th = th

print(f"OOF LightGBM F1: {best_oof_f1:.4f} at threshold: {best_oof_th:.2f}")

# Evaluate on test validation set
val_f1_lgb = f1_score(y_val_arr, (val_preds_lgb >= best_oof_th).astype(int))
print(f"Val.csv F1 using OOF threshold ({best_oof_th:.2f}): {val_f1_lgb:.4f}")

# Also find best threshold directly on val.csv
best_val_f1 = 0
best_val_th = 0.5
for th in np.linspace(0.1, 0.9, 81):
    f1 = f1_score(y_val_arr, (val_preds_lgb >= th).astype(int))
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_val_th = th
print(f"Val.csv Best F1 (oracle threshold {best_val_th:.2f}): {best_val_f1:.4f}")

# Model 2: CatBoost / HistGradientBoosting
print("\n--- Training HistGradientBoosting ---")
hgb = HistGradientBoostingClassifier(
    categorical_features=cat_cols,
    learning_rate=0.03,
    max_iter=400,
    max_leaf_nodes=31,
    random_state=42
)
hgb.fit(X_train_df, y_train_arr)
val_preds_hgb = hgb.predict_proba(X_val_df)[:, 1]

# Model 3: XGBoost (with one-hot / target encoding or enable_categorical)
print("\n--- Training XGBoost ---")
X_tr_xgb = X_train_df.copy()
X_va_xgb = X_val_df.copy()
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=50,
    eval_metric='logloss'
)
xgb_model.fit(X_tr_xgb, y_train_arr, eval_set=[(X_va_xgb, y_val_arr)], verbose=False)
val_preds_xgb = xgb_model.predict_proba(X_va_xgb)[:, 1]

# Evaluate Ensemble
print("\n--- Ensemble Results on val.csv ---")
for w_lgb in [0.4, 0.5, 0.6]:
    for w_xgb in [0.2, 0.3, 0.4]:
        w_hgb = round(1.0 - w_lgb - w_xgb, 2)
        if w_hgb < 0:
            continue
        ens_prob = w_lgb * val_preds_lgb + w_xgb * val_preds_xgb + w_hgb * val_preds_hgb
        for th in np.linspace(0.2, 0.5, 31):
            f1 = f1_score(y_val_arr, (ens_prob >= th).astype(int))
            if f1 > 0.74:
                print(f"Weights ({w_lgb}, {w_xgb}, {w_hgb}), Th: {th:.2f} -> F1: {f1:.4f}")
