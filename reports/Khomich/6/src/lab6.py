import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # сохранение графиков без открытия окна
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Пути к файлам (работают при запуске из любой папки)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
LAB_DIR = SRC_DIR.parent
REPORT_DIR = LAB_DIR / "rep"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LAB5_METRICS_PATH = LAB_DIR.parent / "5" / "rep" / "lab5_metrics.csv"
LAB5_TEST_PATH = LAB_DIR.parent / "5" / "rep" / "lab5_test_results.csv"

# ---------------------------------------------------------------------------
# Параметры варианта и эксперимента
# ---------------------------------------------------------------------------
A, B, C, D = 0.2, 0.4, 0.09, 0.4
N_INPUTS = 6          # количество входов ИНС (длина окна)
N_HIDDEN = 2          # количество нейронов в скрытом слое
N_POINTS = 600        # длина генерируемого временного ряда
TEST_FRACTION = 0.25  # доля тестовой выборки
VALIDATION_FRACTION = 0.15  # доля валидационной части внутри обучения
EPOCHS = 3000         # число итераций обучения
SEED = 42

# Скорости обучения, среди которых выбирается наилучшая
LEARNING_RATES = [0.001, 0.003, 0.005, 0.01, 0.02,
                  0.03, 0.05, 0.08, 0.1, 0.2, 0.3]

# ---------------------------------------------------------------------------
# Воспроизводимость результатов
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Лабораторная работа №6")
print("Рекуррентные ИНС (сеть Элмана)")
print("Вариант 10: a=0.2, b=0.4, c=0.09, d=0.4")
print("Устройство для вычислений:", DEVICE)
print("Версия PyTorch:", torch.__version__)


# ---------------------------------------------------------------------------
# 1. Генерация данных и подготовка выборок
# ---------------------------------------------------------------------------
def generate_data():
    """Генерирует временной ряд и делит его на обучающую и тестовую части."""
    x = np.linspace(0.0, 20.0 * np.pi, N_POINTS)
    y = A * np.cos(B * x) + C * np.sin(D * x)

    split_idx = int(N_POINTS * (1.0 - TEST_FRACTION))
    x_train_raw = x[:split_idx]
    y_train_raw = y[:split_idx]
    x_test_raw = x[split_idx:]
    y_test_raw = y[split_idx:]

    # Данные подаются в сеть в исходном масштабе, без нормализации.
    print("\nПараметры временного ряда:")
    print(f"  длина ряда:            {N_POINTS}")
    print(f"  обучающая часть:       {len(y_train_raw)} точек")
    print(f"  тестовая часть:        {len(y_test_raw)} точек")
    print(f"  среднее (по train):    {y_train_raw.mean():.6f}")
    print(f"  СКО (по train):        {y_train_raw.std():.6f}")
    print(f"  диапазон y:            [{y.min():.4f}; {y.max():.4f}]")

    return {
        "x": x,
        "y": y,
        "x_train_raw": x_train_raw,
        "y_train_raw": y_train_raw,
        "x_test_raw": x_test_raw,
        "y_test_raw": y_test_raw,
        "y_train": y_train_raw,
        "y_test": y_test_raw,
        "split_idx": split_idx,
    }


def make_windows(series, window):
    """Формирует выборку скользящим окном: X — window прошлых значений, Y — следующее."""
    X, Y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        Y.append(series[i])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32).reshape(-1, 1),
    )


# ---------------------------------------------------------------------------
# 2. Архитектура рекуррентной сети Элмана
# ---------------------------------------------------------------------------
class ElmanCell(nn.Module):
    """Один такт сети Элмана: вход, контекстный слой и сигмоида."""

    def __init__(self, n_hidden):
        super().__init__()
        self.input_layer = nn.Linear(1, n_hidden)               # веса входов
        self.context_layer = nn.Linear(n_hidden, n_hidden, bias=False)  # веса контекста

    def forward(self, x_t, h_prev):
        return torch.sigmoid(self.input_layer(x_t) + self.context_layer(h_prev))


class ElmanRNN(nn.Module):
    """Сеть Элмана: рекуррентный скрытый слой и линейный выход."""

    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.cell = ElmanCell(n_hidden)
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x):
        """x: тензор (batch, n_inputs); окно обрабатывается по одному отсчёту."""
        batch = x.size(0)
        h = torch.zeros(batch, self.n_hidden, device=x.device, dtype=x.dtype)
        for t in range(x.size(1)):
            h = self.cell(x[:, t:t + 1], h)
        return self.output(h)


# ---------------------------------------------------------------------------
# 3. Обучение и оценка
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, learning_rate, epochs=EPOCHS, seed=SEED):
    """Обучает РНС с заданной скоростью обучения и возвращает модель и историю ошибки."""
    torch.manual_seed(seed)
    model = ElmanRNN(N_HIDDEN).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(X_train)
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return model, history


def predict(model, X):
    """Возвращает прогноз модели в виде массива NumPy."""
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy().ravel()


def regression_metrics(y_true, y_pred):
    """Считает основные метрики качества регрессии."""
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    max_error = float(np.max(np.abs(error)))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Максимальное отклонение": max_error,
    }


# ---------------------------------------------------------------------------
# 4. Графики
# ---------------------------------------------------------------------------
def plot_function(x_values, y_true, y_pred, title, path,
                  color="tab:blue", ylabel="y"):
    """График прогнозируемой функции и отклонений (обучающий или тестовый участок)."""
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(x_values, y_true, color="black", linewidth=1.8,
                 label="Эталонное значение")
    axes[0].plot(x_values, y_pred, color=color, linewidth=1.4,
                 linestyle="--", label="Прогноз РНС")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_values, y_pred - y_true, color="tab:red", linewidth=1.0)
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Отклонение")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print("Сохранён график:", path)


def plot_loss_curve(history, path):
    """График изменения ошибки MSE в зависимости от итерации."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(range(1, len(history) + 1), history, color="tab:red")
    ax.set_title("Изменение ошибки в зависимости от итерации")
    ax.set_xlabel("Итерация (эпоха)")
    ax.set_ylabel("MSE")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print("Сохранён график:", path)


def plot_comparison(x_values, y_true, y_lab5, y_lab6, path):
    """Сравнение прогнозов ИНС (ЛР 5) и РНС Элмана (ЛР 6) на тестовом участке."""
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(x_values, y_true, color="black", linewidth=1.8,
                 label="Эталонное значение")
    axes[0].plot(x_values, y_lab5, color="tab:blue", linewidth=1.3,
                 linestyle="--", label="ИНС 6–2–1 (ЛР 5)")
    axes[0].plot(x_values, y_lab6, color="tab:orange", linewidth=1.3,
                 linestyle="-.", label="Сеть Элмана (ЛР 6)")
    axes[0].set_ylabel("y")
    axes[0].set_title("Сравнение прогнозов на тестовом участке")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_values, y_lab5 - y_true, color="tab:blue", linewidth=1.0,
                 label="ИНС 6–2–1 (ЛР 5)")
    axes[1].plot(x_values, y_lab6 - y_true, color="tab:orange", linewidth=1.0,
                 label="Сеть Элмана (ЛР 6)")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Отклонение")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print("Сохранён график:", path)


def plot_metrics_comparison(metrics_lab5, metrics_lab6, path):
    """Столбчатая диаграмма сравнения ошибок двух моделей на тесте."""
    names = ["RMSE", "MAE", "Максимальное отклонение"]
    values_lab5 = [metrics_lab5[n] for n in names]
    values_lab6 = [metrics_lab6[n] for n in names]

    positions = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars5 = ax.bar(positions - width / 2, values_lab5, width,
                   color="tab:blue", label="ИНС 6–2–1 (ЛР 5)")
    bars6 = ax.bar(positions + width / 2, values_lab6, width,
                   color="tab:orange", label="Сеть Элмана (ЛР 6)")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel("Значение ошибки (лог. шкала)")
    ax.set_title("Сравнение ошибок моделей на тестовой выборке")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend()

    for bars in (bars5, bars6):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2e}".replace(".", ","),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print("Сохранён график:", path)


# ---------------------------------------------------------------------------
# 5. Основной сценарий
# ---------------------------------------------------------------------------
def main():
    data = generate_data()

    X_train, Y_train = make_windows(data["y_train"], N_INPUTS)
    X_test, Y_test = make_windows(data["y_test"], N_INPUTS)

    # Целевые позиции x для окон
    x_train_targets = data["x_train_raw"][N_INPUTS:]
    x_test_targets = data["x_test_raw"][N_INPUTS:]

    # Для выбора скорости обучения из обучающей части выделяется
    # валидационная выборка (последние 15 % окон). Тестовая выборка
    # при подборе гиперпараметра не используется.
    n_val = int(len(X_train) * VALIDATION_FRACTION)
    X_fit, Y_fit = X_train[:-n_val], Y_train[:-n_val]
    X_val, Y_val = X_train[-n_val:], Y_train[-n_val:]

    print("\nРазмеры выборок после формирования окон:")
    print(f"  X_train (итоговая): {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_fit:              {X_fit.shape}, Y_fit:  {Y_fit.shape}")
    print(f"  X_val:              {X_val.shape}, Y_val:  {Y_val.shape}")
    print(f"  X_test:             {X_test.shape}, Y_test: {Y_test.shape}")

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=DEVICE)
    Y_fit_t = torch.tensor(Y_fit, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=DEVICE)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=DEVICE)

    # ------------------------------------------------------------------
    # Подбор скорости обучения. Результаты работы приводятся для
    # значения, при котором достигается минимальная ошибка.
    # ------------------------------------------------------------------
    print("\nПодбор скорости обучения:")
    lr_rows = []
    best = None
    for lr in LEARNING_RATES:
        model, _ = train_model(X_fit_t, Y_fit_t, lr)
        fit_pred = predict(model, X_fit_t)
        val_pred = predict(model, X_val_t)
        fit_mse = float(np.mean((fit_pred - Y_fit.ravel()) ** 2))
        val_mse = float(np.mean((val_pred - Y_val.ravel()) ** 2))
        lr_rows.append({
            "Скорость обучения": lr,
            "MSE на обучении": fit_mse,
            "MSE на валидации": val_mse,
        })
        print(f"  lr={lr:<6} | MSE_fit={fit_mse:.8f} | MSE_val={val_mse:.8f}")

        if best is None or val_mse < best["val_mse"]:
            best = {"lr": lr, "val_mse": val_mse}

    best_lr = best["lr"]
    print(f"\nЛучшая скорость обучения: {best_lr}")
    print(f"Минимальная ошибка MSE на валидации: {best['val_mse']:.8f}")

    # После выбора гиперпараметра модель обучается заново на всей
    # обучающей выборке.
    model, history = train_model(X_train_t, Y_train_t, best_lr)
    print("Модель обучена заново на полной обучающей выборке.")

    lr_table = pd.DataFrame(lr_rows)
    lr_table.to_csv(REPORT_DIR / "lab6_lr_selection.csv", index=False)

    # ------------------------------------------------------------------
    # Прогноз (данные в исходном масштабе, денормализация не требуется)
    # ------------------------------------------------------------------
    train_pred = predict(model, X_train_t)
    test_pred = predict(model, X_test_t)

    train_true = Y_train.ravel()
    test_true = Y_test.ravel()

    train_metrics = regression_metrics(train_true, train_pred)
    test_metrics = regression_metrics(test_true, test_pred)

    print("\nМетрики на обучающей выборке:")
    for name, value in train_metrics.items():
        print(f"  {name}: {value:.8f}")
    print("\nМетрики на тестовой выборке:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.8f}")

    # ------------------------------------------------------------------
    # Таблицы результатов
    # ------------------------------------------------------------------
    train_table = pd.DataFrame({
        "x": x_train_targets,
        "Эталонное значение": train_true,
        "Полученное значение": train_pred,
        "Отклонение": train_pred - train_true,
    })
    test_table = pd.DataFrame({
        "x": x_test_targets,
        "Эталонное значение": test_true,
        "Полученное значение": test_pred,
        "Отклонение": test_pred - test_true,
    })

    train_table.to_csv(REPORT_DIR / "lab6_training_results.csv", index=False)
    test_table.to_csv(REPORT_DIR / "lab6_test_results.csv", index=False)

    metrics_table = pd.DataFrame({
        "Показатель": ["MSE", "RMSE", "MAE", "R2", "Максимальное отклонение"],
        "Обучение": [
            train_metrics["MSE"], train_metrics["RMSE"], train_metrics["MAE"],
            train_metrics["R2"], train_metrics["Максимальное отклонение"],
        ],
        "Тест": [
            test_metrics["MSE"], test_metrics["RMSE"], test_metrics["MAE"],
            test_metrics["R2"], test_metrics["Максимальное отклонение"],
        ],
    })
    metrics_table.to_csv(REPORT_DIR / "lab6_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # Сравнение с результатами ЛР 5
    # ------------------------------------------------------------------
    if LAB5_METRICS_PATH.exists():
        lab5_table = pd.read_csv(LAB5_METRICS_PATH)
        lab5_metrics = {
            row["Показатель"]: float(row["Тест"])
            for _, row in lab5_table.iterrows()
        }

        comparison_table = pd.DataFrame({
            "Показатель": ["MSE", "RMSE", "MAE", "R2", "Максимальное отклонение"],
            "ИНС 6–2–1 (ЛР 5)": [
                lab5_metrics["MSE"], lab5_metrics["RMSE"], lab5_metrics["MAE"],
                lab5_metrics["R2"], lab5_metrics["Максимальное отклонение"],
            ],
            "Сеть Элмана (ЛР 6)": [
                test_metrics["MSE"], test_metrics["RMSE"], test_metrics["MAE"],
                test_metrics["R2"], test_metrics["Максимальное отклонение"],
            ],
        })
        comparison_table.to_csv(REPORT_DIR / "lab6_comparison.csv", index=False)
        print("\nСравнение с ЛР 5 (тестовая выборка):")
        print(comparison_table.to_string(index=False))

        # Совмещение прогнозов двух моделей на одном графике
        if LAB5_TEST_PATH.exists():
            lab5_test = pd.read_csv(LAB5_TEST_PATH)
            lab5_pred = np.interp(
                x_test_targets,
                lab5_test["x"].to_numpy(),
                lab5_test["Полученное значение"].to_numpy(),
            )
            plot_comparison(
                x_test_targets, test_true, lab5_pred, test_pred,
                REPORT_DIR / "lab6_comparison_test.png",
            )
        plot_metrics_comparison(
            lab5_metrics, test_metrics,
            REPORT_DIR / "lab6_metrics_comparison.png",
        )
    else:
        print("\nФайл с метриками ЛР 5 не найден, сравнение пропущено.")

    # ------------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------------
    plot_function(
        x_train_targets, train_true, train_pred,
        "Прогнозируемая функция на участке обучения",
        REPORT_DIR / "lab6_function_training.png",
    )
    plot_loss_curve(history, REPORT_DIR / "lab6_loss_curve.png")
    plot_function(
        x_test_targets, test_true, test_pred,
        "Прогноз на тестовом участке",
        REPORT_DIR / "lab6_forecast_test.png",
        color="tab:orange",
    )

    # ------------------------------------------------------------------
    # Краткий итог в консоль
    # ------------------------------------------------------------------
    print("\nПервые 10 строк результатов обучения:")
    print(train_table.head(10).to_string(index=False))
    print("\nПервые 10 строк результатов прогнозирования:")
    print(test_table.head(10).to_string(index=False))
    print("\nРабота программы завершена.")


if __name__ == "__main__":
    main()
