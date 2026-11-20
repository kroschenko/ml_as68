import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

filename = 'winequality-white.csv'
df = pd.read_csv(filename, sep=';')

print(" Исходный датасет (Регрессия)")
print(f"Размер: {df.shape}")

X_reg = df.drop(columns=['quality'])
y_reg = df['quality']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

lin_reg = LinearRegression()
lin_reg.fit(X_train_r_scaled, y_train_r)

y_pred_r = lin_reg.predict(X_test_r_scaled)

mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"Метрика MSE: {mse:.4f}")
print(f"Метрика R^2: {r2:.4f}")

plt.figure(figsize=(8, 6))
sns.regplot(
    data=df, 
    x='alcohol', 
    y='quality', 
    scatter_kws={'alpha': 0.3, 'color': 'steelblue'}, 
    line_kws={'color': 'red', 'linewidth': 2}
)
plt.title('Зависимость качества вина от алкоголя')
plt.xlabel('Содержание алкоголя (%)')
plt.ylabel('Оценка качества (quality)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()