"""Лабораторная работа №6 — Рекуррентные нейронные сети.

Вариант 7: сеть Элмана для одношагового прогнозирования временного ряда
y = 0.3·cos(0.1·x) + 0.06·sin(0.1·x).
Вход ИНС — окно из 6 предыдущих значений ряда (как в ЛР 5), в скрытом
слое 2 нейрона с сигмоидной активацией, выход линейный. Отличие от ЛР 5 —
контекстная связь h[t-1] → скрытый слой, которая делает сеть рекуррентной:
окна обрабатываются последовательно, и сеть «помнит» своё состояние на
предыдущем шаге (архитектура Элмана).
"""

import json
import random
from pathlib import Path

import numpy as np
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

# 1. Параметры варианта 7 (функция и архитектура — как в ЛР 5,
# дополнительно задан тип РНС: сеть Элмана)
A, B, C, D = 0.3, 0.1, 0.06, 0.1
L = 6   # кол-во входов ИНС = ширина скользящего окна (как в ЛР 5)
H = 2   # кол-во нейронов в скрытом слое (число контекстных нейронов Элмана)

N = 1200          # число точек временного ряда
TEST_FRAC = 0.25  # доля тестовой (прогнозируемой) части ряда
LR_CANDIDATES = [0.001, 0.005, 0.01, 0.05, 0.1]  # кандидаты скорости обучения
SEARCH_EPOCHS = 1000  # эпох на этапе подбора скорости обучения
EPOCHS = 4000         # эпох финального обучения

# Результаты ЛР 5 (многослойная ИНС 6-2-1) для сравнения. Берутся из
# results.json ветки lab5; если файла нет — используются сохранённые
# значения финального прогона ЛР 5 (SEED=42, lr=0.1, 4000 эпох)
LAB5_FALLBACK = {
    "best_lr": 0.1,
    "final_mse": 1.7301627508459205e-07,
    "train_metrics": {"MAE": 0.0012116065223701346,
                      "RMSE": 0.0012485139810388913,
                      "max |откл|": 0.001620152362869276},
    "test_metrics": {"MAE": 0.001021159568599217,
                     "RMSE": 0.0010766181541441123,
                     "max |откл|": 0.001620188353595453},
}
lab5_path = Path(__file__).resolve().parent / "../../5/src/results.json"
try:
    with open(lab5_path, encoding="utf-8") as f:
        lab5 = json.load(f)
except FileNotFoundError:
    lab5 = LAB5_FALLBACK

# 2. Генерация временного ряда — идентично ЛР 5
# b = d = 0.1, поэтому период сигнала T = 2*pi/0.1 ~ 62.8: на отрезке
# [0, 40*pi] укладывается два полных периода. Ряд делится на обучающую и
# тестовую части ПО ВРЕМЕНИ (без перемешивания)
x = np.linspace(0.0, 40.0 * np.pi, N)
y = A * np.cos(B * x) + C * np.sin(D * x)

split_idx = int(N * (1 - TEST_FRAC))
x_train_raw, y_train_raw = x[:split_idx], y[:split_idx]
x_test_raw, y_test_raw = x[split_idx:], y[split_idx:]
print(f"Всего точек ряда: {N} | обучение: {split_idx} | прогноз: {N - split_idx}")


# 3. Формирование выборки скользящим окном — как в ЛР 5.
# Каждому шагу t соответствует окно из L предыдущих значений ряда и
# целевое значение y[t]. Отличие ЛР 6: окна подаются в сеть НЕ независимо,
# а последовательностью, и между ними через контекст переносится h[t-1]
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


# 4. Архитектура РНС Элмана
# На каждом шаге t:
#   h[t] = sigmoid(Wx·x[t] + Wh·h[t-1] + bx)   — скрытый слой,
#   y[t] = Wy·h[t] + by                         — линейный выход.
# Wh — контекстная (рекуррентная) связь: скрытый слой получает не только
# текущее окно x[t], но и свою активацию h[t-1] с предыдущего шага
# (копия скрытого слоя = контекстный слой сети Элмана). Благодаря этому
# autograd прокатывает градиент по всей цепочке шагов (BPTT).
class ElmanRNN(nn.Module):
    """Вариант 7: сеть Элмана 6+2(контекст)-2-1.

    Входной слой — окно из 6 значений + 2 контекстных входа, скрытый слой —
    2 нейрона с сигмоидной активацией, выходной нейрон линейный.
    Обучаемые параметры: Wx (6x2) + bx (2) + Wh (2x2) + Wy (2x1) + by (1)
    = 21 параметр (у ИНС ЛР 5 их было 17 — добавилась матрица Wh).
    """

    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.in_to_hidden = nn.Linear(n_inputs, n_hidden)          # Wx, bx
        self.ctx_to_hidden = nn.Linear(n_hidden, n_hidden, bias=False)  # Wh
        self.act = nn.Sigmoid()
        self.hidden_to_out = nn.Linear(n_hidden, 1)                # Wy, by

    def forward(self, sequence):
        # h[0] = 0: в начале последовательности контекст пуст
        h = torch.zeros(self.ctx_to_hidden.in_features)
        outputs = []
        for x_t in sequence:  # окна идут строго по времени
            h = self.act(self.in_to_hidden(x_t) + self.ctx_to_hidden(h))
            outputs.append(self.hidden_to_out(h))
        return torch.stack(outputs)  # (n шагов, 1)


def train_model(lr, epochs, seed=SEED):
    """Обучает сеть с нуля с заданной скоростью обучения, возвращает
    модель и историю ошибки (MSE) по эпохам.

    Эпоха = один полный проход по последовательности окон в хронологическом
    порядке; ошибка усредняется по всем шагам, градиенты накапливаются по
    всей цепочке (полный BPTT), шаг оптимизатора — один за эпоху.
    """
    torch.manual_seed(seed)  # одинаковая стартовая инициализация весов
    model = ElmanRNN(L, H)
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


# 5. Подбор скорости обучения (α): обучаем коротко на каждом кандидате и
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

# 7. Прогноз сети на обучающей и тестовой последовательностях.
# Контекст стартует с нуля на каждой из них (как и во время обучения)
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).numpy().flatten()
    test_pred = model(X_test_t).numpy().flatten()

train_true = Y_train.flatten()
test_true = Y_test.flatten()


def make_table(true, pred):
    rows = []
    for t, p in zip(true, pred):
        rows.append({"Эталонное значение": float(t),
                     "Полученное значение": float(p),
                     "Отклонение": float(p - t)})
    return rows


def metrics(rows):
    dev = np.array([r["Отклонение"] for r in rows])
    return {
        "MAE": float(np.mean(np.abs(dev))),
        "RMSE": float(np.sqrt(np.mean(dev ** 2))),
        "max |откл|": float(np.max(np.abs(dev))),
    }


def format_table(rows):
    header = (f"{'Эталонное значение':>22}{'Полученное значение':>24}"
              f"{'Отклонение':>16}")
    body = "\n".join(f"{r['Эталонное значение']:>22.6f}"
                     f"{r['Полученное значение']:>24.6f}"
                     f"{r['Отклонение']:>16.6f}" for r in rows)
    return header + "\n" + body


train_rows, test_rows = make_table(train_true, train_pred), \
    make_table(test_true, test_pred)
train_metrics, test_metrics = metrics(train_rows), metrics(test_rows)

print("\n--- РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 10 строк) ---")
print(format_table(train_rows[:10]))
print("\n--- РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (первые 10 строк) ---")
print(format_table(test_rows[:10]))
print(f"\nОбучение:  MAE = {train_metrics['MAE']:.6f} | "
      f"RMSE = {train_metrics['RMSE']:.6f} | "
      f"max |откл| = {train_metrics['max |откл|']:.6f}")
print(f"Прогноз:   MAE = {test_metrics['MAE']:.6f} | "
      f"RMSE = {test_metrics['RMSE']:.6f} | "
      f"max |откл| = {test_metrics['max |откл|']:.6f}")
print(f"Финальная MSE на обучении: {loss_history[-1]:.8f}")

# Сводные результаты для отчёта (results.json): ключи совпадают с ЛР 5,
# дополнительно сохраняются результаты ЛР 5 для сравнения в выводах
with open("results.json", "w", encoding="utf-8") as f:
    json.dump({
        "lr_search": {str(k): v for k, v in search_results.items()},
        "best_lr": best_lr,
        "final_mse": loss_history[-1],
        "train_head": [{k: round(v, 6) for k, v in r.items()}
                       for r in train_rows[:10]],
        "test_head": [{k: round(v, 6) for k, v in r.items()}
                      for r in test_rows[:10]],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "lab5": {"best_lr": lab5["best_lr"],
                 "final_mse": lab5["final_mse"],
                 "train_metrics": lab5["train_metrics"],
                 "test_metrics": lab5["test_metrics"]},
    }, f, ensure_ascii=False, indent=2)

# Полные таблицы сохраняются в output.txt
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (полная таблица)\n")
    f.write(format_table(train_rows))
    f.write("\n\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (полная таблица)\n")
    f.write(format_table(test_rows))

# 8. Графики (в стиле отчёта ЛР 5)
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
         linewidth=1.5, label="Сеть Элмана")
plt.title("Результаты обучения: эталон и выход сети Элмана")
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
         linewidth=1.5, label="Прогноз сети Элмана")
plt.title("Результаты прогнозирования: эталон и прогноз сети Элмана")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("04_forecast.png", dpi=150)
plt.close()

print("\nГрафики сохранены: 01_function.png, 02_train_fit.png, "
      "03_loss_curve.png, 04_forecast.png")
