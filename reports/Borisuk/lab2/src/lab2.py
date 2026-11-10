import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='median_house_value')

print("Признаки:", california.feature_names)
print("Размер данных:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model_reg = LinearRegression()
model_reg.fit(X_train, y_train)


y_pred = model_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE (Mean Squared Error): {mse:.4f}")
print(f"R2 (Coefficient of Determination): {r2:.4f}")
print(f"Интерпретация: модель объясняет {r2*100:.2f}% дисперсии целевой переменной.")


X_vis = X[['MedInc']]  


model_vis = LinearRegression()
model_vis.fit(X_vis, y)


x_line = np.linspace(X_vis.min(), X_vis.max(), 100).reshape(-1, 1)
y_line = model_vis.predict(x_line)

plt.figure(figsize=(10, 6))
plt.scatter(X_vis, y, alpha=0.3, s=10, label='Данные')
plt.plot(x_line, y_line, color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Median Income (медианный доход)')
plt.ylabel('Median House Value (медианная стоимость дома)')
plt.title('Зависимость стоимости жилья от медианного дохода (California Housing)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('california_regression_plot.png', dpi=150)
plt.show()