"""Лабораторная работа №5 — Нелинейные ИНС в задачах регрессии.

Вариант 7: y = 0.3·cos(0.1·x) + 0.06·sin(0.1·x),
архитектура ИНС 6-2-1: 6 входов (скользящее окно), 2 нейрона со
сигмоидной активацией в скрытом слое, 1 линейный выход.
"""

import json
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # сохранение графиков без открытия окна
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Воспроизводимость результатов: фиксируем генераторы случайных чисел,
# чтобы инициализация весов при каждом запуске была одинаковой
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Параметры варианта 7
A, B, C, D = 0.3, 0.1, 0.06, 0.1
L = 6   # кол-во входов ИНС = ширина скользящего окна
H = 2   # кол-во нейронов в скрытом слое

N = 1200          # число точек временного ряда
TEST_FRAC = 0.25  # доля тестовой (прогнозируемой) части ряда
LR_CANDIDATES = [0.001, 0.005, 0.01, 0.05, 0.1]  # кандидаты скорости обучения
SEARCH_EPOCHS = 1000  # эпох на этапе подбора скорости обучения
EPOCHS = 4000         # эпох финального обучения

# 2. Генерация временного ряда
# b = d = 0.1, поэтому период сигнала T = 2*pi/0.1 ~ 62.8: на отрезке
# [0, 40*pi] укладывается два полных периода. Ряд делится на обучающую и
# тестовую части ПО ВРЕМЕНИ (без перемешивания): сеть обучается на первых
# 75 % ряда, а затем прогнозирует его продолжение
x = np.linspace(0.0, 40.0 * np.pi, N)
y = A * np.cos(B * x) + C * np.sin(D * x)

split_idx = int(N * (1 - TEST_FRAC))
x_train_raw, y_train_raw = x[:split_idx], y[:split_idx]
x_test_raw, y_test_raw = x[split_idx:], y[split_idx:]
print(f"Всего точек ряда: {N} | обучение: {split_idx} | прогноз: {N - split_idx}")


# 3. Формирование обучающей выборки скользящим окном
# Для прогнозирования следующего значения сеть получает L предыдущих
# значений ряда. Выборки для обучения и прогноза строятся независимо,
# чтобы окна не "перешагивали" границу раздела
def make_dataset(series, window):
    X, Y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)


X_train, Y_train = make_dataset(y_train_raw, L)
X_test, Y_test = make_dataset(y_test_raw, L)
print(f"Обучающих примеров: {X_train.shape[0]} | тестовых: {X_test.shape[0]}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test.reshape(-1, 1), dtype=torch.float32)


# 4. Архитектура ИНС: один скрытый слой
class ForecastNN(nn.Module):
    """Вариант 7: архитектура 6-2-1.

    Скрытый слой из 2 нейронов с сигмоидной функцией активации,
    выходной нейрон с линейной активацией — стандартная схема для
    задач регрессии (линейный выход не ограничивает диапазон прогноза).
    """

    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.act = nn.Sigmoid()
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, t):
        return self.output(self.act(self.hidden(t)))


def train_model(lr, epochs, seed=SEED):
    """Обучает сеть с нуля с заданной скоростью обучения, возвращает
    модель и историю ошибки (MSE) по эпохам."""
    torch.manual_seed(seed)  # одинаковая стартовая инициализация весов
    model = ForecastNN(L, H)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, Y_train_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history


# 5. Подбор скорости обучения: обучаем коротко на каждом кандидате и
# выбираем тот, при котором достигается минимальная ошибка
print("\nПодбор скорости обучения:")
print(f"{'lr':>8} | {'MSE (обучение)':>16}")
search_results = {}
for lr in LR_CANDIDATES:
    _, history = train_model(lr, SEARCH_EPOCHS)
    search_results[lr] = history[-1]
    print(f"{lr:>8} | {history[-1]:>16.8f}")
best_lr = min(search_results, key=search_results.get)
print(f"Минимальная ошибка при lr = {best_lr}")

# 6. Финальное обучение с лучшей скоростью обучения
model, loss_history = train_model(best_lr, EPOCHS)

# 7. Прогноз сети на обучающей и тестовой выборках
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).numpy().flatten()
    test_pred = model(X_test_t).numpy().flatten()

train_true = Y_train.flatten()
test_true = Y_test.flatten()

train_df = pd.DataFrame({
    "Эталонное значение": train_true,
    "Полученное значение": train_pred,
    "Отклонение": train_pred - train_true,
})
test_df = pd.DataFrame({
    "Эталонное значение": test_true,
    "Полученное значение": test_pred,
    "Отклонение": test_pred - test_true,
})


def metrics(df):
    dev = df["Отклонение"].to_numpy()
    return {
        "MAE": float(np.mean(np.abs(dev))),
        "RMSE": float(np.sqrt(np.mean(dev ** 2))),
        "max |откл|": float(np.max(np.abs(dev))),
    }


train_metrics, test_metrics = metrics(train_df), metrics(test_df)

print("\n--- РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 10 строк) ---")
print(train_df.head(10).to_string(index=False,
      formatters={c: "{:.6f}".format for c in train_df.columns}))
print("\n--- РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (первые 10 строк) ---")
print(test_df.head(10).to_string(index=False,
      formatters={c: "{:.6f}".format for c in test_df.columns}))
print(f"\nОбучение:  MAE = {train_metrics['MAE']:.6f} | "
      f"RMSE = {train_metrics['RMSE']:.6f} | "
      f"max |откл| = {train_metrics['max |откл|']:.6f}")
print(f"Прогноз:   MAE = {test_metrics['MAE']:.6f} | "
      f"RMSE = {test_metrics['RMSE']:.6f} | "
      f"max |откл| = {test_metrics['max |откл|']:.6f}")
print(f"Финальная MSE на обучении: {loss_history[-1]:.8f}")

# Сводные результаты для отчёта (results.json)
with open("results.json", "w", encoding="utf-8") as f:
    json.dump({
        "lr_search": {str(k): v for k, v in search_results.items()},
        "best_lr": best_lr,
        "final_mse": loss_history[-1],
        "train_head": train_df.head(10).round(6).to_dict(orient="records"),
        "test_head": test_df.head(10).round(6).to_dict(orient="records"),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }, f, ensure_ascii=False, indent=2)

# Полные таблицы сохраняются в output.txt
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (полная таблица)\n")
    f.write(train_df.to_string(index=False,
            formatters={c: "{:.6f}".format for c in train_df.columns}))
    f.write("\n\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (полная таблица)\n")
    f.write(test_df.to_string(index=False,
            formatters={c: "{:.6f}".format for c in test_df.columns}))

# 8. Графики
# Ряд с участком обучения и участком прогнозирования
plt.figure(figsize=(12, 4))
plt.plot(x_train_raw, y_train_raw, color="tab:blue", label="Участок обучения")
plt.plot(x_test_raw, y_test_raw, color="tab:orange", label="Участок прогнозирования")
plt.title("Прогнозируемая функция y = 0.3·cos(0.1x) + 0.06·sin(0.1x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("01_function.png", dpi=150)
plt.close()

# Прогноз сети на участке обучения
plt.figure(figsize=(12, 4))
plt.plot(x_train_raw[L:], train_true, color="black", linewidth=1.5,
         label="Эталон")
plt.plot(x_train_raw[L:], train_pred, color="tab:blue", linestyle="--",
         linewidth=1.5, label="Модель")
plt.title("Результаты обучения: эталон и выход ИНС")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("02_train_fit.png", dpi=150)
plt.close()

# Ошибка обучения в зависимости от итерации
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color="tab:red")
plt.title(f"Изменение ошибки обучения (MSE), lr = {best_lr}")
plt.xlabel("Итерация (эпоха)")
plt.ylabel("MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("03_loss_curve.png", dpi=150)
plt.close()

# Прогноз на тестовом участке
plt.figure(figsize=(12, 4))
plt.plot(x_test_raw[L:], test_true, color="black", linewidth=1.5,
         label="Эталон")
plt.plot(x_test_raw[L:], test_pred, color="tab:green", linestyle="--",
         linewidth=1.5, label="Прогноз ИНС")
plt.title("Результаты прогнозирования: эталон и прогноз ИНС")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("04_forecast.png", dpi=150)
plt.close()

print("\nГрафики сохранены: 01_function.png, 02_train_fit.png, "
      "03_loss_curve.png, 04_forecast.png")
