import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv("twitter_retweets.csv") 

# Выбираем признаки
features = ["reach", "likes", "klout"] 
X = df[features].to_numpy(dtype=float)
y = df["retweetcount"].to_numpy(dtype=float)

# 2. Разделение 80/20
np.random.seed(42)
idx = np.random.permutation(len(X))
n_test = int(0.2 * len(X))
test_idx, train_idx = idx[:n_test], idx[n_test:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 3. Линейная регрессия
add_bias = lambda X: np.column_stack([np.ones(len(X)), X])
beta, *_ = np.linalg.lstsq(add_bias(X_train), y_train, rcond=None)
y_pred = add_bias(X_test) @ beta

# 4. Метрики
mae = np.mean(np.abs(y_test - y_pred))
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"MAE: {mae:.2f}")
print(f"R2 : {r2:.3f}")

# --- НОВЫЙ БЛОК: ВЫВОД ТАБЛИЦЫ ПРЕДСКАЗАНИЙ ---
print("\nТаблица предсказаний (первые 10 строк):")
results = pd.DataFrame({
    'Реальный retweetcount': y_test[:10],
    'Предсказанный retweetcount': y_pred[:10],
    'Разница': y_test[:10] - y_pred[:10]
})
print(results)
# -----------------------------------------------

# 5. График
k, b = np.polyfit(df["reach"], df["retweetcount"], 1)
x_line = np.linspace(df["reach"].min(), df["reach"].max(), 100)

plt.figure(figsize=(8, 6))
plt.scatter(df["reach"], df["retweetcount"], alpha=0.5, color='blue')
plt.plot(x_line, k * x_line + b, color="red", linewidth=2, label="Линия регрессии")
plt.xlabel("Охват (reach)")
plt.ylabel("Количество ретивов (retweetcount)")
plt.title("Зависимость ретивов от охвата")
plt.legend()
plt.grid(alpha=0.3)
plt.show()