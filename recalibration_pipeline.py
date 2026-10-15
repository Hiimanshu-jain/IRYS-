import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb

class RecalibrationModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_lgb_rounds=1200, n_xgb_rounds=600, lr=0.025, lgb_weight=0.35, xgb_weight=0.65):
        self.n_lgb_rounds = n_lgb_rounds
        self.n_xgb_rounds = n_xgb_rounds
        self.lr = lr
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        
        self.event_module_map = {
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

        self.unit_module_map = {
            '%O2': 'Life-Support',
            'V': 'Lab-Power',
            'dBm': 'Antenna',
            'msg/s': 'Notice-Bus'
        }

        self.nominal_bounds = {
            'Lab-Power': (11.3, 12.7, 12.0, 0.7),
            'Life-Support': (20.0, 22.3, 21.15, 1.15),
            'Antenna': (37.0, 48.0, 42.5, 5.5),
            'Notice-Bus': (275.0, 345.0, 310.0, 35.0)
        }

        self.serious_events = set(self.event_module_map.keys())
        self.harmless_events = {'periodic report', 'value logged', 'status changed', 'scheduled calibration', 'self-test pattern', 'simulated drill event'}

    def _extract_features(self, df):
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
        feats['high_retry'] = (df['retry_count'] >= 2).astype(int)
        feats['delay_per_retry'] = df['delay_ms'] / (df['retry_count'] + 1)
        
        # 3. Differences and mismatches
        feats['mod_mismatch'] = (df['module'] != b_mod).astype(int)
        feats['val_diff'] = (df['reading'] - b_val).abs()
        feats['val_mismatch'] = (feats['val_diff'] > 0.05).astype(int)
        feats['val_ratio'] = df['reading'] / (b_val + 1e-4)
        
        expected_unit = df['module'].map(self.unit_module_map)
        feats['unit_mismatch'] = (b_unit != expected_unit).astype(int)
        
        expected_event_mod = event.map(self.event_module_map)
        feats['event_mod_mismatch'] = (expected_event_mod.notna() & (expected_event_mod != df['module'])).astype(int)
        
        # 4. Physics bounds
        low_b = df['module'].map(lambda m: self.nominal_bounds.get(m, (0, 0, 0, 1))[0])
        high_b = df['module'].map(lambda m: self.nominal_bounds.get(m, (0, 0, 0, 1))[1])
        center_b = df['module'].map(lambda m: self.nominal_bounds.get(m, (0, 0, 0, 1))[2])
        scale_b = df['module'].map(lambda m: self.nominal_bounds.get(m, (0, 0, 0, 1))[3])
        
        feats['dist_below_nominal'] = np.maximum(0, low_b - df['reading'])
        feats['dist_above_nominal'] = np.maximum(0, df['reading'] - high_b)
        feats['is_out_of_nominal'] = ((df['reading'] < low_b) | (df['reading'] > high_b)).astype(int)
        feats['reading_norm_dist'] = (df['reading'] - center_b).abs() / scale_b
        feats['reading_zscore'] = (df['reading'] - center_b) / scale_b
        
        feats['b_dist_below_nominal'] = np.maximum(0, low_b - b_val)
        feats['b_dist_above_nominal'] = np.maximum(0, b_val - high_b)
        feats['b_is_out_of_nominal'] = ((b_val < low_b) | (b_val > high_b)).astype(int)
        
        feats['nine_sanitized_reading'] = (feats['is_out_of_nominal'] == 1) & (feats['b_is_out_of_nominal'] == 0)
        feats['nine_faked_reading'] = (feats['is_out_of_nominal'] == 0) & (feats['b_is_out_of_nominal'] == 1)
        
        # 5. Status
        status_order = {'OK': 0, 'WARN': 1, 'ERROR': 2, 'CRIT': 3}
        feats['status_severity'] = df['status'].map(status_order).fillna(0)
        feats['status_is_ok'] = (df['status'] == 'OK').astype(int)
        feats['status_is_crit'] = (df['status'] == 'CRIT').astype(int)
        feats['status_is_error'] = (df['status'] == 'ERROR').astype(int)
        feats['status_is_warn'] = (df['status'] == 'WARN').astype(int)
        feats['is_alarm_status'] = df['status'].isin(['ERROR', 'CRIT']).astype(int)
        
        # 6. Event and Prefix types
        feats['is_serious_event'] = event.isin(self.serious_events).astype(int)
        feats['is_harmless_event'] = event.isin(self.harmless_events).astype(int)
        feats['is_calming_prefix'] = prefix.isin(['all is well', 'nothing to see', 'crew need not worry']).astype(int)
        feats['is_neutral_prefix'] = prefix.isin(['logging', 'noted', 'none']).astype(int)
        
        # 7. Deception & Paradox interactions
        feats['calm_words_alarm_status'] = feats['is_calming_prefix'] * feats['is_alarm_status']
        feats['calm_words_out_of_nominal'] = feats['is_calming_prefix'] * feats['is_out_of_nominal']
        feats['calm_words_serious_event'] = feats['is_calming_prefix'] * feats['is_serious_event']
        
        feats['calm_status_out_of_nominal'] = feats['status_is_ok'] * feats['is_out_of_nominal']
        feats['calm_status_serious_event'] = feats['status_is_ok'] * feats['is_serious_event']
        
        feats['false_alarm_suspect'] = (feats['is_alarm_status']) & (feats['is_out_of_nominal'] == 0) & (feats['is_serious_event'] == 0)
        
        feats['redacted_with_alarm'] = is_redacted * feats['is_alarm_status']
        feats['redacted_with_calm'] = is_redacted * feats['is_calming_prefix']
        
        feats['mode_alert'] = (df['mode'] == 'ALERT').astype(int)
        feats['mode_maint'] = (df['mode'] == 'MAINT').astype(int)
        feats['alert_with_calm_status'] = feats['mode_alert'] * feats['status_is_ok']
        feats['alert_with_calm_words'] = feats['mode_alert'] * feats['is_calming_prefix']
        feats['maint_alarm'] = feats['mode_maint'] * feats['is_alarm_status']
        
        # 8. Time features
        ts = pd.to_datetime(df['timestamp'])
        feats['hour'] = ts.dt.hour
        feats['day'] = ts.dt.day
        feats['dayofweek'] = ts.dt.dayofweek
        feats['minute'] = ts.dt.minute
        
        feats['msg_char_count'] = df['message'].str.len()
        feats['msg_word_count'] = df['message'].str.split().str.len()
        
        # Keyword checks
        for kw in ['queue overflow', 'heartbeat missing', 'stalled', 'deadlock', 'breach',
                   'brownout', 'overvolt', 'tripped', 'falling', 'drift', 'phase lock', 'saturation']:
            feats[f'kw_{kw}'] = df['message'].str.contains(kw, case=False).astype(int)
            
        # Categories
        cat_cols = ['module', 'b_mod', 'mode', 'node', 'status', 'prefix', 'event', 'b_unit']
        for c, col_val in [('module', df['module']), ('b_mod', b_mod), ('mode', df['mode']),
                           ('node', df['node']), ('status', df['status']), ('prefix', prefix),
                           ('event', event), ('b_unit', b_unit)]:
            feats[c] = col_val.astype('category')
            
        return feats

    def fit(self, X, y):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120)
        self.svd = TruncatedSVD(n_components=10, random_state=42)
        
        clean_msg = X['message'].str.replace(r'\[.*?\]', '', regex=True).str.replace('NINE:', '', regex=False)
        tfidf_mat = self.tfidf.fit_transform(clean_msg)
        svd_mat = self.svd.fit_transform(tfidf_mat)
        
        feats = self._extract_features(X)
        for i in range(10):
            feats[f'svd_{i}'] = svd_mat[:, i]
            
        self.feature_names_ = feats.columns.tolist()
        self.cat_cols_ = ['module', 'b_mod', 'mode', 'node', 'status', 'prefix', 'event', 'b_unit']
        
        # Train LightGBM
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': self.lr,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1
        }
        trn_data = lgb.Dataset(feats, label=y, categorical_feature=self.cat_cols_)
        self.lgb_model = lgb.train(lgb_params, trn_data, num_boost_round=self.n_lgb_rounds)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_xgb_rounds,
            learning_rate=self.lr,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.75,
            enable_categorical=True,
            random_state=42,
            tree_method='hist',
            eval_metric='logloss'
        )
        self.xgb_model.fit(feats, y, verbose=False)
        
        return self

    def predict_proba(self, X):
        clean_msg = X['message'].str.replace(r'\[.*?\]', '', regex=True).str.replace('NINE:', '', regex=False)
        tfidf_mat = self.tfidf.transform(clean_msg)
        svd_mat = self.svd.transform(tfidf_mat)
        
        feats = self._extract_features(X)
        for i in range(10):
            feats[f'svd_{i}'] = svd_mat[:, i]
            
        p_lgb = self.lgb_model.predict(feats)
        p_xgb = self.xgb_model.predict_proba(feats)[:, 1]
        
        p_blend = self.lgb_weight * p_lgb + self.xgb_weight * p_xgb
        
        # Return probability matrix [P(0), P(1)]
        return np.vstack([1.0 - p_blend, p_blend]).T

    def predict(self, X, threshold=0.37):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    val = pd.read_csv('val.csv')
    
    model = RecalibrationModel()
    print("Fitting model on train.csv...")
    model.fit(train, train['is_true_fault'])
    
    val_probs = model.predict_proba(val)[:, 1]
    
    from sklearn.metrics import f1_score
    best_th, best_f1 = 0.5, 0
    for th in np.linspace(0.2, 0.6, 41):
        f1 = f1_score(val['is_true_fault'], (val_probs >= th).astype(int))
        if f1 > best_f1:
            best_f1, best_th = f1, th
            
    print(f"Validation F1: {best_f1:.4f} at threshold {best_th:.2f}")
