import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

vec = TfidfVectorizer(ngram_range=(1, 3), min_df=5, token_pattern=r'(?u)\b\w+\b|\[.*?\]')
X_txt = vec.fit_transform(df['message'])
y = df['is_true_fault']

lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_txt, y)

coefs = pd.Series(lr.coef_[0], index=vec.get_feature_names_out())

print("Top 25 features pointing to TRUE FAULT (1):")
print(coefs.sort_values(ascending=False).head(25))

print("\nTop 25 features pointing to FALSE ALARM (0):")
print(coefs.sort_values(ascending=True).head(25))
