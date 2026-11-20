import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler

filename = 'winequality-white.csv'
df = pd.read_csv(filename, sep=';')

df['is_good'] = (df['quality'] >= 7).astype(int)

print("Исходный датасет")
print(f"Распределение классов 'is_good':\n{df['is_good'].value_counts()}")

X_cls = df.drop(columns=['quality', 'is_good'])
y_cls = df['is_good']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_c_scaled, y_train_c)

y_pred_c = log_reg.predict(X_test_c_scaled)

acc = accuracy_score(y_test_c, y_pred_c)
prec = precision_score(y_test_c, y_pred_c, pos_label=1)
rec = recall_score(y_test_c, y_pred_c, pos_label=1)

print(f"Accuracy  (Доля верных ответов): {acc:.4f}")
print(f"Precision (Точность для класса 1): {prec:.4f}")
print(f"Recall    (Полнота для класса 1):  {rec:.4f}")

cm = confusion_matrix(y_test_c, y_pred_c)

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Плохое (<7)', 'Хорошее (>=7)'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Матрица ошибок (Confusion Matrix)')

plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

df['is_good_str'] = df['is_good'].map({1: 'good', 0: 'bad'})
df.to_csv('winequality_classified.csv', index=False)