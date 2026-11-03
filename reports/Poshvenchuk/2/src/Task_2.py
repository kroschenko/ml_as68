import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Загрузка
df = pd.read_csv("breast_cancer.csv")

# Убираем служебные/пустые столбцы
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == "id"])

# diagnosis -> target
if "diagnosis" in df.columns:
    df = df.rename(columns={"diagnosis": "target"})

# M/B -> 0/1 (malignant = 0, benign = 1)
df["target"] = df["target"].astype(str).str.strip().map({"M": 0, "B": 1})
assert df["target"].notna().all(), "В target остались NaN — проверьте CSV"
df["target"] = df["target"].astype(int)
df = df.dropna()

# 2. Разделение 80/20
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
n_test = int(0.2 * len(df))
test_idx, train_idx = idx[:n_test], idx[n_test:]

# Формирование матриц признаков (X) и целевых переменных (y) для train и test
X = df.drop(columns=["target"])
y = df["target"].to_numpy()
X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y[train_idx], y[test_idx]

# 3. Стандартизация по train
# Вычисление среднего и стандартного отклонения только по обучающей выборке
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Добавление столбца единиц для свободного члена
add_bias = lambda A: np.column_stack([np.ones(len(A)), A.to_numpy(dtype=float)])
Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

# 4. Логистическая регрессия (градиентный спуск, L2-регуляризация)
# Обучение модели: 5000 итераций градиентного спуска для поиска весов
sigmoid = lambda z: 1 / (1 + np.exp(-z))
w = np.zeros(Xb_train.shape[1])
lr, lam = 0.1, 0.01
for _ in range(5000):
    p = sigmoid(Xb_train @ w)
    grad = Xb_train.T @ (p - y_train) / len(y_train)
    reg = lam * w
    reg[0] = 0 # свободный член не регуляризуем
    w -= lr * (grad + reg)

y_pred = (sigmoid(Xb_test @ w) >= 0.5).astype(int)

# 5. Матрица ошибок и метрики (положительный класс — malignant = 0)
y_true_pos = (y_test == 0).astype(int)
y_pred_pos = (y_pred == 0).astype(int)

# Расчет метрик качества классификации (TP, TN, FP, FN)
tp = np.sum((y_pred_pos == 1) & (y_true_pos == 1))
tn = np.sum((y_pred_pos == 0) & (y_true_pos == 0))
fp = np.sum((y_pred_pos == 1) & (y_true_pos == 0))
fn = np.sum((y_pred_pos == 0) & (y_true_pos == 1))

# Вычисление Accuracy, Precision, Recall и F1-score
accuracy = (tp + tn) / len(y_test)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(f"Общая точность: {accuracy:.4f}")
print(f"Точность предсказания рака: {precision:.4f}")
print(f"Полнота (сколько найдено больных раком): {recall:.4f}")

# Визуализация матрицы ошибок
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Доброкачественная", "Злокачественная"])
ax.set_yticks([0, 1], ["Доброкачественная", "Злокачественная"])
ax.set_xlabel("Предсказано моделью")
ax.set_ylabel("Истинный диагноз")
ax.set_title("Матрица ошибок — Рак груди")

# Вывод цифр в ячейках
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
        
plt.tight_layout()
plt.show()