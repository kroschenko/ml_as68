import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'winequality-white.csv'), sep=';')

X_reg = df.drop(columns=['quality'])
y_reg = df['quality']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train_r, y_train_r)
y_pred_r = lr.predict(X_test_r)

mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("Регрессия")
print(f"MSE: {mse:.4f}")
print(f"R2:  {r2:.4f}\n")

plt.figure(figsize=(8, 5))
sns.regplot(
    x='alcohol', y='quality', data=df, 
    scatter_kws={'alpha':0.2, 'color':'blue'}, 
    line_kws={'color':'red'}
)
plt.title('Зависимость качества (quality) от алкоголя (alcohol)')
plt.xlabel('Alcohol (% vol)')
plt.ylabel('Quality')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

df['is_good'] = (df['quality'] >= 7).astype(int)

X_cls = df.drop(columns=['quality', 'is_good'])
y_cls = df['is_good']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

acc = accuracy_score(y_test_c, y_pred_c)
prec = precision_score(y_test_c, y_pred_c)
rec = recall_score(y_test_c, y_pred_c)
cm = confusion_matrix(y_test_c, y_pred_c)

print("Классификация")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', 
    xticklabels=['Плохое (<7)', 'Хорошее (>=7)'], 
    yticklabels=['Плохое (<7)', 'Хорошее (>=7)']
)
plt.title('Матрица ошибок (Logistic Regression)')
plt.xlabel('Предсказанный класс')
plt.ylabel('Фактический класс')
plt.tight_layout()
plt.show()