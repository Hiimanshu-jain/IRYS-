import json

with open('starter_round1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Act 2 cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "from sklearn.pipeline import make_pipeline" in src and "LogisticRegression" in src:
            cell['source'] = [
                "from recalibration_pipeline import RecalibrationModel\n",
                "\n",
                "progress(\"Training Recalibration Protocol v1 (Ensemble & Feature Engine)\")\n",
                "model = RecalibrationModel()\n",
                "model.fit(train, train['is_true_fault'])\n",
                "sysmsg(\"Recalibration Protocol v1 trained.\")\n"
            ]
        elif "THRESHOLD = 0.5" in src:
            cell['source'] = [
                "val_prob = model.predict_proba(val)[:, 1]\n",
                "THRESHOLD = 0.27\n"
            ]

with open('starter_round1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("starter_round1.ipynb updated successfully.")
