import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

a = 0.2
b = 0.4
c = 0.09
d = 0.4

INPUTS = 6
HIDDEN_NEURONS = 2

print("Прогнозирование нелинейной ИНС")
print("-" * 45)

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = {d}")
print(f"Количество входов: {INPUTS}")
print(f"Нейронов скрытого слоя: {HIDDEN_NEURONS}")

def function(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

x = np.linspace(0, 50, 1000)
y = function(x)

X = []
Y = []

for i in range(INPUTS, len(y)):
    X.append(y[i - INPUTS:i])
    Y.append(y[i])

X = np.array(X)
Y = np.array(Y)

print("\nКоличество обучающих примеров:", len(X))

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

Y_train = Y[:split]
Y_test = Y[split:]

print("Обучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(
    Y_train.reshape(-1, 1)
).ravel()

model = MLPRegressor(
    hidden_layer_sizes=(HIDDEN_NEURONS,),
    activation='logistic',
    solver='lbfgs',
    max_iter=20000,
    random_state=42,
    alpha=0.000001,
    tol=1e-10
)

print("\nОбучение нейронной сети...")

model.fit(X_train_scaled, Y_train_scaled)

print("Обучение завершено!")

Y_pred_scaled = model.predict(X_test_scaled)

Y_pred = scaler_Y.inverse_transform(
    Y_pred_scaled.reshape(-1, 1)
).ravel()

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
print("-" * 45)

print(f"Среднеквадратичная ошибка MSE: {mse:.8f}")
print(f"Средняя абсолютная ошибка MAE: {mae:.8f}")
print(f"Корень из MSE (RMSE): {rmse:.8f}")
print(f"Коэффициент детерминации R2: {r2:.8f}")

plt.figure(figsize=(12, 5))

plt.plot(x, y, label="Исходная функция", color="blue")

plt.title("Исходная функция (вариант 10)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

test_x = x[INPUTS + split:]

plt.figure(figsize=(12, 5))

plt.plot(
    test_x,
    Y_test,
    label="Реальные значения",
    color="blue"
)

plt.plot(
    test_x,
    Y_pred,
    label="Прогноз ИНС",
    color="red",
    linestyle="--"
)

plt.title("Сравнение реальных и прогнозируемых значений")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

errors = Y_test - Y_pred

plt.figure(figsize=(12, 5))

plt.plot(test_x, errors, color="purple")

plt.axhline(
    y=0,
    color="black",
    linestyle="--"
)

plt.title("Ошибка прогнозирования ИНС")
plt.xlabel("x")
plt.ylabel("Ошибка")
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nАРХИТЕКТУРА НЕЙРОННОЙ СЕТИ")
print("-" * 45)

print(f"Входной слой: {INPUTS} нейронов")
print(f"Скрытый слой: {HIDDEN_NEURONS} нейрона")
print("Выходной слой: 1 нейрон")

print("Активация скрытого слоя: сигмоида")
print("Активация выходного слоя: линейная")

print("\nКоличество обучаемых параметров:",
      sum(w.size for w in model.coefs_) +
      sum(b.size for b in model.intercepts_))
