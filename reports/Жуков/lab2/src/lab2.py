import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка и обработка категориальных признаков
df = pd.read_csv("medical_cost_personal_dataset.csv")
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)

# 2. Разделение 80/20
test = df.sample(frac=0.2, random_state=42)
train = df.drop(test.index)

features = [c for c in df.columns if c != "charges"]
X_train = train[features].to_numpy(dtype=float)
X_test = test[features].to_numpy(dtype=float)
y_train = train["charges"].to_numpy(dtype=float)
y_test = test["charges"].to_numpy(dtype=float)

# 3. Линейная регрессия (добавляем столбец единиц для свободного члена)
add_bias = lambda X: np.column_stack([np.ones(len(X)), X])
beta, *_ = np.linalg.lstsq(add_bias(X_train), y_train, rcond=None)
y_pred = add_bias(X_test) @ beta

# 4. Метрики
mae = np.mean(np.abs(y_test - y_pred))
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"MAE: {mae:.2f}")
print(f"R2 : {r2:.3f}")
print(pd.Series(beta[1:], index=features).sort_values())

# 5. Диаграмма рассеяния charges от bmi + линия регрессии
k, b = np.polyfit(df["bmi"], df["charges"], 1)
x_line = np.linspace(df["bmi"].min(), df["bmi"].max(), 100)

plt.figure(figsize=(8, 6))
plt.scatter(df["bmi"], df["charges"], c=df["smoker"], cmap="coolwarm", alpha=0.6)
plt.plot(x_line, k * x_line + b, color="black", linewidth=2, label="Линия регрессии")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("Зависимость charges от BMI (красные: курильщики)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()