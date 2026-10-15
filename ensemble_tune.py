import numpy as np
from sklearn.metrics import f1_score
from cv_experiment import val_preds_lgb, val_preds_xgb, val_preds_hgb, y_val_arr

best_f1 = 0
best_w = None
best_th = 0.5

for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, 21):
        w3 = round(1.0 - w1 - w2, 4)
        if w3 < 0:
            continue
        prob = w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_hgb
        for th in np.linspace(0.2, 0.5, 61):
            score = f1_score(y_val_arr, (prob >= th).astype(int))
            if score > best_f1:
                best_f1 = score
                best_w = (w1, w2, w3)
                best_th = th

print(f"Optimal Ensemble Val F1: {best_f1:.4f} with weights (LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, HGB={best_w[2]:.2f}) at threshold {best_th:.2f}")
