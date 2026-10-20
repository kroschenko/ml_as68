import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка, целевая переменная (0 = здоров, 1-4 = болезнь)
df = pd.read_csv("heart_disease_uci.csv")
y = (df["num"] > 0).astype(int).to_numpy()
X = df.drop(columns=["num", "id", "dataset"])

# 2. Категориальные признаки: пропуски -> "unknown", затем one-hot
cat_cols = X.select_dtypes(exclude="number").columns.tolist()  # включая fbs, exang
num_cols = X.select_dtypes(include="number").columns.tolist()
X[cat_cols] = X[cat_cols].fillna("unknown").astype(str)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

# 3. Разделение 80/20 (случайное)
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
n_test = int(0.2 * len(X))
test_idx, train_idx = idx[:n_test], idx[n_test:]
X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y[train_idx], y[test_idx]

# 4. Пропуски в числовых: медиана по train; масштабирование по train
med = X_train[num_cols].median()
X_train[num_cols] = X_train[num_cols].fillna(med)
X_test[num_cols] = X_test[num_cols].fillna(med)
mean, std = X_train[num_cols].mean(), X_train[num_cols].std()
X_train[num_cols] = (X_train[num_cols] - mean) / std
X_test[num_cols] = (X_test[num_cols] - mean) / std

add_bias = lambda A: np.column_stack([np.ones(len(A)), A.to_numpy(dtype=float)])
Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

# 5. Логистическая регрессия (градиентный спуск, L2-регуляризация)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
w = np.zeros(Xb_train.shape[1])
lr, lam = 0.1, 0.01
for _ in range(5000):
    p = sigmoid(Xb_train @ w)
    grad = Xb_train.T @ (p - y_train) / len(y_train)
    reg = lam * w
    reg[0] = 0                      # свободный член не регуляризуем
    w -= lr * (grad + reg)

y_pred = (sigmoid(Xb_test @ w) >= 0.5).astype(int)

# 6. Матрица ошибок и метрики
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

accuracy = (tp + tn) / len(y_test)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")

cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Здоров", "Болен"])
ax.set_yticks([0, 1], ["Здоров", "Болен"])
ax.set_xlabel("Предсказано")
ax.set_ylabel("Истинное значение")
ax.set_title("Матрица ошибок")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.show()