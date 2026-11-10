import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка
df = pd.read_csv("BostonHousing.csv")
df = df.drop(columns=[c for c in ["Unnamed: 0", "CAT. MEDV"] if c in df.columns])
df = df.dropna()

# 2. Разделение 80/20
test = df.sample(frac=0.2, random_state=52)
train = df.drop(test.index)

# Формирование выборок
features = [c for c in df.columns if c != "MEDV"]
X_train = train[features].to_numpy(dtype=float)
X_test = test[features].to_numpy(dtype=float)
y_train = train["MEDV"].to_numpy(dtype=float)
y_test = test["MEDV"].to_numpy(dtype=float)

# 3. Линейная регрессия
add_bias = lambda X: np.column_stack([np.ones(len(X)), X])
beta, *_ = np.linalg.lstsq(add_bias(X_train), y_train, rcond=None)
y_pred = add_bias(X_test) @ beta

# 4. Метрики и вывод предсказаний
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"MSE: {mse:.3f}")
print(f"R2 : {r2:.3f}")

results = pd.DataFrame({
    'Реальная цена (тыс. $)': y_test,
    'Предсказанная цена': y_pred.round(2),
    'Разница': (y_test - y_pred).round(2)
})
print(results.sample(n=10, random_state=52))

# 5. Визуализация (MEDV от RM)
k, b = np.polyfit(df["RM"], df["MEDV"], 1)
x_line = np.linspace(df["RM"].min(), df["RM"].max(), 100)

plt.figure(figsize=(8, 6))
plt.scatter(df["RM"], df["MEDV"], alpha=0.6, color="steelblue")
plt.plot(x_line, k * x_line + b, color="black", linewidth=2, label="Линия регрессии")
plt.xlabel("RM (среднее число комнат)")
plt.ylabel("MEDV (медианная стоимость, $1000)")
plt.title("Зависимость MEDV от RM")
plt.legend()
plt.grid(alpha=0.3)
plt.show()