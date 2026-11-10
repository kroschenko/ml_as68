import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a, b, c, d = 0.4, 0.4, 0.08, 0.4
L = 6  # Кол-во входов ИНС
H = 2  # Кол-во НЭ в скрытом слое

epochs = 2000
lr = 0.05
test_frac = 0.25
N = 600

#  ГЕНЕРАЦИЯ ВРЕМЕННОГО РЯДА И РАЗДЕЛЕНИЕ
x = np.linspace(0, 20 * np.pi, N)
y = a * np.cos(b * x) + c * np.sin(d * x)

# Разделение строго ДО нормализации и нарезки окон
split_idx = int(N * (1 - test_frac))
y_train_raw = y[:split_idx]
y_test_raw = y[split_idx:]
x_train_raw = x[:split_idx]
x_test_raw = x[split_idx:]

# Вычисление параметров нормализации строго по y_train
y_mean = y_train_raw.mean()
y_std = y_train_raw.std()

y_train_norm = (y_train_raw - y_mean) / y_std
y_test_norm = (y_test_raw - y_mean) / y_std

# Функция формирования датасета скользящим окном L
def make_dataset(y_series, L):
    X, Y = [], []
    for i in range(L, len(y_series)):
        X.append(y_series[i - L:i])
        Y.append(y_series[i])
    return np.array(X), np.array(Y).reshape(-1, 1)

X_train, Y_train_norm = make_dataset(y_train_norm, L)
X_test, Y_test_norm = make_dataset(y_test_norm, L)

# Данные для графиков и таблиц в исходном масштабе
Y_train_true = Y_train_norm * y_std + y_mean
Y_test_true = Y_test_norm * y_std + y_mean

x_train_targets = x_train_raw[L:]
x_test_targets = x_test_raw[L:]

def sigmoid(z):
    z_clip = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z_clip))

def sigmoid_deriv(a_sigmoid):
    return a_sigmoid * (1.0 - a_sigmoid)

# Инициализация весов
rng = np.random.default_rng(42)
W1 = rng.normal(0.0, np.sqrt(1.0 / L), size=(L, H))
b1 = np.zeros((1, H))
W2 = rng.normal(0.0, np.sqrt(1.0 / H), size=(H, 1))
b2 = np.zeros((1, 1))

# Прямой проход
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    return z2, (X, z1, a1, z2)

# Обратный проход
def backward(W1, b1, W2, b2, cache, y_pred, Y, lr):
    X, z1, a1, z2 = cache
    num_samples = X.shape[0]

    err = y_pred - Y
    grad_out = (2.0 / num_samples) * err

    dW2 = a1.T @ grad_out
    db2 = np.sum(grad_out, axis=0, keepdims=True)

    da1 = grad_out @ W2.T
    dz1 = da1 * sigmoid_deriv(a1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    loss = np.mean(err ** 2)
    return W1, b1, W2, b2, loss

# Обучение ИНС
loss_history = []
for epoch in range(1, epochs + 1):
    y_pred, cache = forward(X_train, W1, b1, W2, b2)
    W1, b1, W2, b2, loss = backward(W1, b1, W2, b2, cache, Y_train_norm, lr)
    loss_history.append(loss)
    if epoch % 500 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | MSE={loss:.6f}")

# --- 4. ПРОГНОЗИРОВАНИЕ И ДЕНОРМАЛИЗАЦИЯ ---
def predict(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    return a1 @ W2 + b2

train_pred_norm = predict(X_train, W1, b1, W2, b2)
test_pred_norm = predict(X_test, W1, b1, W2, b2)

# Возвращаем результаты в исходный масштаб
train_pred = train_pred_norm * y_std + y_mean
test_pred = test_pred_norm * y_std + y_mean

# --- 5. ВЫВОД РЕЗУЛЬТАТОВ ---
train_df = pd.DataFrame({
    "Эталонное значение": Y_train_true.flatten(),
    "Полученное значение": train_pred.flatten(),
    "Отклонение": (train_pred - Y_train_true).flatten()
})

test_df = pd.DataFrame({
    "Эталонное значение": Y_test_true.flatten(),
    "Полученное значение": test_pred.flatten(),
    "Отклонение": (test_pred - Y_test_true).flatten()
})

print("\nРезультаты обучения (ЛР5):")
print(train_df.head(10))

print("\nРезультаты прогнозирования (ЛР5):")
print(test_df.head(10))

# График 1: Функция на участке обучения
plt.figure(figsize=(10, 4))
plt.plot(x_train_targets, Y_train_true, label="Эталон (train)", color="black")
plt.plot(x_train_targets, train_pred, label="Модель (train)", color="blue", linestyle="--")
plt.title("График прогнозируемой функции на участке обучения (ЛР5)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# График 2: Ошибка MSE
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color="red")
plt.title("График изменения ошибки в зависимости от итерации (MSE)")
plt.xlabel("Итерация (Эпоха)")
plt.ylabel("MSE")
plt.grid(True)
plt.show()