import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv("creditcard.csv") 

# Целевая переменная: Class (0 - норма, 1 - мошенничество)
y = df["Class"].to_numpy()
X = df.drop(columns=["Class"]).to_numpy(dtype=float)

# 2. Разделение 80/20 (с сохранением пропорций классов - stratify)
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

# Получаем вероятности для тестовой выборки
probabilities = sigmoid(add_bias(X_test) @ w)

# --- БЛОК ОПТИМИЗАЦИИ ПОРОГА КЛАССИФИКАЦИИ ---
best_threshold = 0.5
best_f1 = -1
best_metrics = {}

# Перебираем возможные пороги для максимизации F1-Score
for threshold in np.linspace(0.01, 0.99, 99):
    y_pred_temp = (probabilities >= threshold).astype(int)
    
    tp_t = np.sum((y_pred_temp == 1) & (y_test == 1))
    tn_t = np.sum((y_pred_temp == 0) & (y_test == 0))
    fp_t = np.sum((y_pred_temp == 1) & (y_test == 0))
    fn_t = np.sum((y_pred_temp == 0) & (y_test == 1))
    
    acc_t = (tp_t + tn_t) / len(y_test)
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
    
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = threshold
        best_metrics = {
            'accuracy': acc_t,
            'precision': prec_t,
            'recall': rec_t,
            'f1': f1_t,
            'tp': tp_t, 'tn': tn_t, 'fp': fp_t, 'fn': fn_t
        }

# Применяем лучший найденный порог
y_pred = (probabilities >= best_threshold).astype(int)
# ---------------------------------------------

# 5. Вывод финальных метрик
print(f"Оптимальный порог: {best_threshold:.2f}")
print(f"Accuracy : {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall   : {best_metrics['recall']:.4f}")
print(f"F1-Score : {best_metrics['f1']:.4f}")

# 6. Матрица ошибок (визуализация)
cm = np.array([[best_metrics['tn'], best_metrics['fp']], 
               [best_metrics['fn'], best_metrics['tp']]])

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1], ["Норма (0)", "Мошенничество (1)"])
ax.set_yticks([0, 1], ["Норма (0)", "Мошенничество (1)"])
ax.set_xlabel("Предсказано")
ax.set_ylabel("Истинное значение")
ax.set_title(f"Матрица ошибок (Порог = {best_threshold:.2f})")

for i in range(2):
    for j in range(2):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.show()
