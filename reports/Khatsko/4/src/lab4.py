import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # сохранение графиков без открытия окна
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

# Воспроизводимость результатов: фиксируем генераторы случайных чисел.
# Инициализация весов сети и разбиение выборки при каждом запуске будут одинаковыми.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Загрузка данных и бинарная целевая переменная
df = pd.read_csv("winequality-white.csv", sep=";")
df = df.drop_duplicates()
print("Размер данных (без дубликатов):", df.shape)
df["is_good"] = (df["quality"] >= 7).astype(np.float32)
print("Хороших вин (оценка >= 7):", int(df['is_good'].sum()), "из", len(df))

# 2. Разделение выборки и стандартизация признаков
X = df.drop(columns=["quality", "is_good"])
y = df["is_good"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# StandardScaler обучается только на train, чтобы информация из test
# не "утекала" в модель
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Обучающая выборка:", X_train.shape[0], "| Тестовая выборка:", X_test.shape[0])

# 3. Преобразование данных в тензоры PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train.to_numpy().reshape(-1, 1), dtype=torch.float32)
y_test_t = torch.tensor(y_test.to_numpy().reshape(-1, 1), dtype=torch.float32)

# Обучение идет мини-батчами по 32 объекта: за одну эпоху weights
# обновляются не один, а ceil(3168 / 32) = 99 раз, поэтому сеть успевает сойтись
BATCH_SIZE = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)


# 4. Архитектура нейронной сети (многослойный перцептрон)
class MLP(nn.Module):
    """Вариант 3: два скрытых слоя по 12 нейронов (ReLU), выход - 1 нейрон.

    Sigmoid не выносится в сеть: лосс BCEWithLogitsLoss внутри себя
    применяет сигмоиду к логитам (так вычисления численно устойчивее).
    Сигмоида применяется отдельно на этапе оценки модели.
    """

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 12),  # входной -> первый скрытый слой
            nn.ReLU(),
            nn.Linear(12, 12),          # первый -> второй скрытый слой
            nn.ReLU(),
            nn.Linear(12, 1),           # второй скрытый -> выходной слой (логит)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(X_train_t.shape[1])
print("\nАрхитектура модели:")
print(model)
n_params = sum(p.numel() for p in model.parameters())
print("Количество обучаемых параметров:", n_params)

# 5. Функция потерь и оптимизатор
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 6. Цикл обучения
def train(model, optimizer, n_epochs):
    """Обучает модель, возвращает средние потери на train и test по эпохам."""
    train_losses, test_losses = [], []

    for epoch in range(n_epochs):
        model.train()           # режим обучения
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)               # forward pass (логиты)
            loss = criterion(y_pred, y_batch)     # потери на мини-батче

            optimizer.zero_grad()                 # обнуляем градиенты
            loss.backward()                       # обратное распространение ошибки
            optimizer.step()                      # шаг оптимизации

            epoch_loss += loss.item() * len(X_batch)

        # средние потери за эпоху на обеих выборках (контроль переобучения)
        train_losses.append(epoch_loss / len(train_loader.dataset))
        with torch.no_grad():
            model.eval()
            test_losses.append(criterion(model(X_test_t), y_test_t).item())

        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch + 1:3d}: loss (train) = {train_losses[-1]:.4f}, "
                  f"loss (test) = {test_losses[-1]:.4f}")

    return train_losses, test_losses


def evaluate(model, X, y_true, name):
    """Оценивает модель на тестовых данных и печатает метрики."""
    model.eval()                            # режим оценки
    with torch.no_grad():                   # градиенты не нужны
        probs = torch.sigmoid(model(X))     # логиты -> вероятности
        pred = (probs > 0.5).float()        # порог 0.5 -> класс 0/1

    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1-score": f1_score(y_true, pred),
    }
    print(f"\n{name}")
    print(f"Accuracy = {metrics['Accuracy']:.3f}, Precision = {metrics['Precision']:.3f}, "
          f"Recall = {metrics['Recall']:.3f}, F1-score = {metrics['F1-score']:.3f}")
    return metrics, pred


EPOCHS = 100
print(f"\n--- Обучение базовой модели ({EPOCHS} эпох) ---")
train_losses, test_losses = train(model, optimizer, EPOCHS)
base_metrics, base_pred = evaluate(model, X_test_t, y_test, f"Базовая модель ({EPOCHS} эпох):")

# 7. Эксперимент: обучение с удвоенным количеством эпох.
# Заново фиксируем сид ДО создания модели, чтобы веса инициализировались
# так же, как у базовой: сравнение тогда отличается только числом эпох.
torch.manual_seed(SEED)
model_exp = MLP(X_train_t.shape[1])
optimizer_exp = torch.optim.Adam(model_exp.parameters(), lr=0.001)

EPOCHS_EXP = EPOCHS * 2
print(f"\n--- Эксперимент: обучение с {EPOCHS_EXP} эпохами ---")
train_losses_exp, test_losses_exp = train(model_exp, optimizer_exp, EPOCHS_EXP)
exp_metrics, exp_pred = evaluate(model_exp, X_test_t, y_test, f"Эксперимент ({EPOCHS_EXP} эпох):")

# 8. График функции потерь
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS_EXP + 1), train_losses_exp, label="Loss (train), 200 эпох")
plt.plot(range(1, EPOCHS_EXP + 1), test_losses_exp, label="Loss (test), 200 эпох")
plt.axvline(EPOCHS, color="red", linestyle="--", label=f"Эпоха {EPOCHS} (точка сравнения)")
plt.xlabel("Эпоха")
plt.ylabel("BCEWithLogitsLoss")
plt.title("Кривые обучения MLP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("01_loss_curves.png")
plt.close()

# 9. Матрицы ошибок обеих моделей
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, pred) in zip(axes, [(f"{EPOCHS} эпох", base_pred),
                                   (f"{EPOCHS_EXP} эпох", exp_pred)]):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Обычное", "Хорошее"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(f"MLP, {name}")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Реально")
plt.tight_layout()
plt.savefig("02_confusion_matrices.png")
plt.close()

# 10. Сравнение метрик базовой модели и эксперимента
results = pd.DataFrame(
    [base_metrics, exp_metrics],
    index=[f"MLP ({EPOCHS} эпох)", f"MLP ({EPOCHS_EXP} эпох)"],
).round(3)
print("\nСравнение моделей (класс 'хорошее вино')")
print(results.to_string())

f1_delta = exp_metrics["F1-score"] - base_metrics["F1-score"]
print(f"\nИзменение F1-score при удвоении эпох: {f1_delta:+.3f}")

metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width / 2, [base_metrics[m] for m in metrics_names],
                width, label=f"{EPOCHS} эпох", color="#4C72B0")
bars2 = plt.bar(x + width / 2, [exp_metrics[m] for m in metrics_names],
                width, label=f"{EPOCHS_EXP} эпох", color="#55A868")
for bars in (bars1, bars2):
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", fontsize=9)
plt.xticks(x, metrics_names)
plt.ylim(0, 1.1)
plt.ylabel("Значение метрики")
plt.title("Влияние количества эпох обучения на метрики MLP")
plt.legend()
plt.tight_layout()
plt.savefig("03_metrics_comparison.png")
plt.close()
