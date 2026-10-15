import pandas as pd
import numpy as np
import re
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

from baseline_model import extract_features

X_train = extract_features(train)
y_train = train['is_true_fault']
X_val = extract_features(val)
y_val = val['is_true_fault']

cat_cols = ['module', 'b_module', 'mode', 'node', 'status', 'prefix', 'event']
clf = HistGradientBoostingClassifier(categorical_features=cat_cols, random_state=42, max_iter=300)
clf.fit(X_train, y_train)

val_probs = clf.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.23).astype(int)

val['pred'] = val_preds
val['prob'] = val_probs

print("=== FALSE NEGATIVES (Actual 1, Pred 0) ===")
fn = val[(val['is_true_fault'] == 1) & (val['pred'] == 0)]
print(f"Total FN: {len(fn)}")
print(fn[['log_id', 'module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message', 'prob']].head(25))

print("\n=== FALSE POSITIVES (Actual 0, Pred 1) ===")
fp = val[(val['is_true_fault'] == 0) & (val['pred'] == 1)]
print(f"Total FP: {len(fp)}")
print(fp[['log_id', 'module', 'mode', 'status', 'reading', 'delay_ms', 'retry_count', 'message', 'prob']].head(25))
