import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

df_labeled = pd.concat([train, val], ignore_index=True)
n_train = len(df_labeled)
n_test = len(test)

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

optimal_bounds = {
    'Lab-Power': (10.75, 13.31, 12.03, 1.28),
    'Life-Support': (18.88, 23.27, 21.075, 2.195),
    'Antenna': (31.56, 51.70, 41.63, 10.07),
    'Notice-Bus': (244.26, 380.23, 312.245, 67.985)
}

serious_events = set(event_module_map.keys())
harmless_events = {'periodic report', 'value logged', 'status changed', 'scheduled calibration', 'self-test pattern', 'simulated drill event'}

def create_features(full_df):
    feats = pd.DataFrame(index=full_df.index)
    
    # 1. Parse message
    pattern = r'NINE:\s*(?:(all is well|nothing to see|crew need not worry|logging|noted)\.?\s*)?(.*?)\s*\[(.*?)\s+(REDACTED|[0-9\.]+)\s*(.*?)\]'
    ext = full_df['message'].str.extract(pattern)
    prefix = ext[0].fillna('none')
    event = ext[1].fillna('').str.strip()
    b_mod = ext[2].fillna('unknown')
    b_val = pd.to_numeric(ext[3], errors='coerce')
    b_unit = ext[4].fillna('unknown')
    is_redacted = (ext[3] == 'REDACTED').astype(int)
    
    # 2. Raw numerics
    feats['reading'] = full_df['reading']
    feats['b_val'] = b_val
    feats['is_redacted'] = is_redacted
    feats['reading_is_na'] = full_df['reading'].isna().astype(int)
    
    # Delays and Retries
    feats['delay_ms'] = full_df['delay_ms']
    feats['log_delay_ms'] = np.log1p(full_df['delay_ms'])
    feats['retry_count'] = full_df['retry_count']
    feats['has_retry'] = (full_df['retry_count'] > 0).astype(int)
    feats['high_retry'] = (full_df['retry_count'] >= 2).astype(int)
    feats['delay_per_retry'] = full_df['delay_ms'] / (full_df['retry_count'] + 1)
    
    # Differences and mismatches
    feats['mod_mismatch'] = (full_df['module'] != b_mod).astype(int)
    feats['val_diff'] = (full_df['reading'] - b_val).abs()
    feats['val_mismatch'] = (feats['val_diff'] > 0.05).astype(int)
    feats['val_ratio'] = full_df['reading'] / (b_val + 1e-4)
    
    expected_unit = full_df['module'].map(unit_module_map)
    feats['unit_mismatch'] = (b_unit != expected_unit).astype(int)
    
    expected_event_mod = event.map(event_module_map)
    feats['event_mod_mismatch'] = (expected_event_mod.notna() & (expected_event_mod != full_df['module'])).astype(int)
    
    # Optimal physics bounds
    low_b = full_df['module'].map(lambda m: optimal_bounds.get(m, (0, 0, 0, 1))[0])
    high_b = full_df['module'].map(lambda m: optimal_bounds.get(m, (0, 0, 0, 1))[1])
    center_b = full_df['module'].map(lambda m: optimal_bounds.get(m, (0, 0, 0, 1))[2])
    scale_b = full_df['module'].map(lambda m: optimal_bounds.get(m, (0, 0, 0, 1))[3])
    
    feats['dist_below_nominal'] = np.maximum(0, low_b - full_df['reading'])
    feats['dist_above_nominal'] = np.maximum(0, full_df['reading'] - high_b)
    feats['is_out_of_nominal'] = ((full_df['reading'] < low_b) | (full_df['reading'] > high_b)).astype(int)
    feats['reading_norm_dist'] = (full_df['reading'] - center_b).abs() / scale_b
    feats['reading_zscore'] = (full_df['reading'] - center_b) / scale_b
    
    feats['b_dist_below_nominal'] = np.maximum(0, low_b - b_val)
    feats['b_dist_above_nominal'] = np.maximum(0, b_val - high_b)
    feats['b_is_out_of_nominal'] = ((b_val < low_b) | (b_val > high_b)).astype(int)
    
    feats['nine_sanitized_reading'] = (feats['is_out_of_nominal'] == 1) & (feats['b_is_out_of_nominal'] == 0)
    feats['nine_faked_reading'] = (feats['is_out_of_nominal'] == 0) & (feats['b_is_out_of_nominal'] == 1)
    
    # Status
    status_order = {'OK': 0, 'WARN': 1, 'ERROR': 2, 'CRIT': 3}
    feats['status_severity'] = full_df['status'].map(status_order).fillna(0)
    feats['status_is_ok'] = (full_df['status'] == 'OK').astype(int)
    feats['status_is_crit'] = (full_df['status'] == 'CRIT').astype(int)
    feats['status_is_error'] = (full_df['status'] == 'ERROR').astype(int)
    feats['status_is_warn'] = (full_df['status'] == 'WARN').astype(int)
    feats['is_alarm_status'] = full_df['status'].isin(['ERROR', 'CRIT']).astype(int)
    
    # Event types
    feats['is_serious_event'] = event.isin(serious_events).astype(int)
    feats['is_harmless_event'] = event.isin(harmless_events).astype(int)
    
    # Prefix types
    feats['is_calming_prefix'] = prefix.isin(['all is well', 'nothing to see', 'crew need not worry']).astype(int)
    feats['is_neutral_prefix'] = prefix.isin(['logging', 'noted', 'none']).astype(int)
    
    # Deception / Paradox features
    feats['calm_words_alarm_status'] = feats['is_calming_prefix'] * feats['is_alarm_status']
    feats['calm_words_out_of_nominal'] = feats['is_calming_prefix'] * feats['is_out_of_nominal']
    feats['calm_words_serious_event'] = feats['is_calming_prefix'] * feats['is_serious_event']
    
    feats['calm_status_out_of_nominal'] = feats['status_is_ok'] * feats['is_out_of_nominal']
    feats['calm_status_serious_event'] = feats['status_is_ok'] * feats['is_serious_event']
    
    feats['false_alarm_suspect'] = (feats['is_alarm_status']) & (feats['is_out_of_nominal'] == 0) & (feats['is_serious_event'] == 0)
    
    feats['redacted_with_alarm'] = is_redacted * feats['is_alarm_status']
    feats['redacted_with_calm'] = is_redacted * feats['is_calming_prefix']
    
    # Mode interactions
    feats['mode_alert'] = (full_df['mode'] == 'ALERT').astype(int)
    feats['mode_maint'] = (full_df['mode'] == 'MAINT').astype(int)
    feats['alert_with_calm_status'] = feats['mode_alert'] * feats['status_is_ok']
    feats['alert_with_calm_words'] = feats['mode_alert'] * feats['is_calming_prefix']
    feats['maint_alarm'] = feats['mode_maint'] * feats['is_alarm_status']
    
    # Time features
    ts = pd.to_datetime(full_df['timestamp'])
    feats['hour'] = ts.dt.hour
    feats['day'] = ts.dt.day
    feats['dayofweek'] = ts.dt.dayofweek
    feats['minute'] = ts.dt.minute
    
    feats['msg_char_count'] = full_df['message'].str.len()
    feats['msg_word_count'] = full_df['message'].str.split().str.len()
    
    # Specific keywords
    for kw in ['queue overflow', 'heartbeat missing', 'stalled', 'deadlock', 'breach',
               'brownout', 'overvolt', 'tripped', 'falling', 'drift', 'phase lock', 'saturation']:
        feats[f'kw_{kw}'] = full_df['message'].str.contains(kw, case=False).astype(int)
        
    # Categories
    feats['module'] = full_df['module'].astype('category')
    feats['b_mod'] = b_mod.astype('category')
    feats['mode'] = full_df['mode'].astype('category')
    feats['node'] = full_df['node'].astype('category')
    feats['status'] = full_df['status'].astype('category')
    feats['prefix'] = prefix.astype('category')
    feats['event'] = event.astype('category')
    feats['b_unit'] = b_unit.astype('category')
    
    return feats

combined = pd.concat([df_labeled, test], ignore_index=True)
all_feats = create_features(combined)

clean_msg = combined['message'].str.replace(r'\[.*?\]', '', regex=True).str.replace('NINE:', '', regex=False)
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150)
tfidf_mat = tfidf.fit_transform(clean_msg[:n_train])
tfidf_test = tfidf.transform(clean_msg[n_train:])

svd = TruncatedSVD(n_components=12, random_state=42)
svd_mat = svd.fit_transform(tfidf_mat)
svd_test = svd.transform(tfidf_test)

for i in range(12):
    all_feats.loc[:n_train-1, f'svd_{i}'] = svd_mat[:, i]
    all_feats.loc[n_train:, f'svd_{i}'] = svd_test[:, i]

X_train = all_feats.iloc[:n_train].copy()
y_train = df_labeled['is_true_fault'].values
X_test = all_feats.iloc[n_train:].copy().reset_index(drop=True)

cat_cols = ['module', 'b_mod', 'mode', 'node', 'status', 'prefix', 'event', 'b_unit']

# 5-fold CV with 3 seeds for LightGBM and XGBoost
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_hgb = np.zeros(n_train)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_hgb = np.zeros(n_test)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 20,
    'random_state': 42,
    'verbose': -1
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # 1. LightGBM
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=trn_data)
    
    bst_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[val_idx] = bst_lgb.predict(X_va, num_iteration=bst_lgb.best_iteration)
    test_lgb += bst_lgb.predict(X_test, num_iteration=bst_lgb.best_iteration) / 5
    
    # 2. XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        enable_categorical=True,
        random_state=42 + fold,
        tree_method='hist',
        early_stopping_rounds=50,
        eval_metric='logloss'
    )
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_clf.predict_proba(X_va)[:, 1]
    test_xgb += xgb_clf.predict_proba(X_test)[:, 1] / 5
    
    # 3. HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        categorical_features=cat_cols,
        learning_rate=0.025,
        max_iter=500,
        max_leaf_nodes=31,
        early_stopping=True,
        random_state=42 + fold
    )
    hgb.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb.predict_proba(X_va)[:, 1]
    test_hgb += hgb.predict_proba(X_test)[:, 1] / 5

print("\n--- OOF Performance per model ---")
for name, preds in [('LightGBM', oof_lgb), ('XGBoost', oof_xgb), ('HistGB', oof_hgb)]:
    best_f1, best_th = 0, 0.5
    for th in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_train, (preds >= th).astype(int))
        if f1 > best_f1:
            best_f1, best_th = f1, th
    print(f"{name:15s}: F1 = {best_f1:.4f} at threshold = {best_th:.2f}")

# Grid search 3-way weights
best_f1, best_w, best_th = 0, None, 0.5
for w1 in np.linspace(0.1, 0.7, 13):
    for w2 in np.linspace(0.1, 0.7, 13):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < 0:
            continue
        prob = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_hgb
        for th in np.linspace(0.25, 0.45, 41):
            f1 = f1_score(y_train, (prob >= th).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_w = (w1, w2, w3)
                best_th = th

print(f"\n--- Best Tri-Ensemble ---")
print(f"Weights (LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, HGB={best_w[2]:.2f})")
print(f"Optimal Threshold: {best_th:.2f}")
print(f"Optimal OOF F1: {best_f1:.4f}")

oof_verdict = (best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_hgb >= best_th).astype(int)
print(confusion_matrix(y_train, oof_verdict))
print(classification_report(y_train, oof_verdict, digits=4))

# Compute test predictions with optimal ensemble
final_prob = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_hgb
final_verdict = (final_prob >= best_th).astype(int)

sub = pd.DataFrame({'log_id': test['log_id'], 'is_true_fault': final_verdict})
sub.to_csv('submission.csv', index=False)
print(f"Saved submission.csv with {len(sub)} rows. True faults: {final_verdict.sum()} ({final_verdict.mean():.1%})")

import check_submission
check_submission.check('submission.csv')
