import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv("creditcard.csv") 

# Целевая переменная: Class (0 - норма, 1 - мошенничество)
y = df["Class"].to_numpy()
X = df.drop(columns=["Class"]).to_numpy(dtype=float)

# 2. Разделение 80/20 (с сохранением пропорций классов - stratify)
# Так как классы несбалансированы, разделяем вручную с сохранением пропорций
np.random.seed(42)
idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]

np.random.shuffle(idx_0)
np.random.shuffle(idx_1)

n_test_0 = int(0.2 * len(idx_0))
n_test_1 = int(0.2 * len(idx_1))

test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 3. Стандартизация признаков (важно для логистической регрессии)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1 # защита от деления на 0
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

add_bias = lambda A: np.column_stack([np.ones(len(A)), A])

# 4. Логистическая регрессия (градиентный спуск)
# Для борьбы с дисбалансом используем веса классов
weight_0 = len(y_train) / (2 * np.sum(y_train == 0))
weight_1 = len(y_train) / (2 * np.sum(y_train == 1))
sample_weights = np.where(y_train == 0, weight_0, weight_1)

sigmoid = lambda z: 1 / (1 + np.exp(-z))
w = np.zeros(X_train.shape[1] + 1)
lr, lam = 0.01, 0.001
for _ in range(2000):
    p = sigmoid(add_bias(X_train) @ w)
    grad = add_bias(X_train).T @ ((p - y_train) * sample_weights) / len(y_train)
    reg = lam * w
    reg[0] = 0
    w -= lr * (grad + reg)

y_pred = (sigmoid(add_bias(X_test) @ w) >= 0.5).astype(int)

# 5. Матрица ошибок и метрики
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

accuracy = (tp + tn) / len(y_test)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

# 6. Матрица ошибок (визуализация)
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Норма (0)", "Мошенничество (1)"])
ax.set_yticks([0, 1], ["Норма (0)", "Мошенничество (1)"])
ax.set_xlabel("Предсказано")
ax.set_ylabel("Истинное значение")
ax.set_title("Матрица ошибок")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.show()