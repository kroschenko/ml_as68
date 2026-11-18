import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# парметры
A, B, C, D = 0.1, 0.5, 0.09, 0.5
L, H = 8, 3                 
EPOCHS = 2000
TEST_FRAC = 0.25
N = 200
SEED = 42
LR_CANDIDATES = [0.001, 0.01, 0.05, 0.1, 0.5]   

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def make_dataset(y_series, window):
    X, Y = [], []
    for i in range(window, len(y_series)):
        X.append(y_series[i - window:i])
        Y.append(y_series[i])
    return np.array(X), np.array(Y).reshape(-1, 1)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    return z2, (X, a1, z2)

def backward(params, cache, y_pred, Y, lr):
    W1, b1, W2, b2 = params
    X, a1, _ = cache
    n = X.shape[0]
    grad_out = (2.0 / n) * (y_pred - Y)

    dW2 = a1.T @ grad_out
    db2 = grad_out.sum(axis=0, keepdims=True)

    dz1 = (grad_out @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
    loss = np.mean((y_pred - Y) ** 2)
    return (W1, b1, W2, b2), loss

def init_weights(seed):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(1.0 / L), size=(L, H))
    b1 = np.zeros((1, H))
    W2 = rng.normal(0, np.sqrt(1.0 / H), size=(H, 1))
    b2 = np.zeros((1, 1))
    return [W1, b1, W2, b2]

def train(X, Y, lr, epochs, seed=SEED):
    params = init_weights(seed)
    losses = []
    for _ in range(epochs):
        y_pred, cache = forward(X, *params)
        params, loss = backward(params, cache, y_pred, Y, lr)
        losses.append(loss)
    return params, losses

def predict(X, params):
    W1, b1, W2, b2 = params
    return sigmoid(X @ W1 + b1) @ W2 + b2

# нормализация
x = np.linspace(0, 20 * np.pi, N)
y = A * np.cos(B * x) + C * np.sin(D * x)

split = int(N * (1 - TEST_FRAC))
y_tr_raw, y_te_raw = y[:split], y[split:]
x_tr_raw, x_te_raw = x[:split], x[split:]

y_mean, y_std = y_tr_raw.mean(), y_tr_raw.std()
y_tr = (y_tr_raw - y_mean) / y_std
y_te = (y_te_raw - y_mean) / y_std

X_tr, Y_tr_n = make_dataset(y_tr, L)
X_te, Y_te_n = make_dataset(y_te, L)

Y_tr_true = Y_tr_n * y_std + y_mean
Y_te_true = Y_te_n * y_std + y_mean
x_tr_tgt = x_tr_raw[L:]
x_te_tgt = x_te_raw[L:]

# подбор альфа
print("Подбор alpha (lr):")
best_lr, best_mse = None, float('inf')
for lr_try in LR_CANDIDATES:
    p, _ = train(X_tr, Y_tr_n, lr_try, EPOCHS)
    mse = np.mean((predict(X_te, p) - Y_te_n) ** 2)
    print(f"  lr={lr_try:<6} | MSE_test = {mse:.6f}")
    if mse < best_mse:
        best_mse, best_lr = mse, lr_try
print(f"\n>>> Лучшее alpha = {best_lr} (MSE_test = {best_mse:.6f})\n")

# обучение с лучшим альфа
params, loss_history = train(X_tr, Y_tr_n, best_lr, EPOCHS)

train_pred = predict(X_tr, params) * y_std + y_mean
test_pred  = predict(X_te, params) * y_std + y_mean

# таблицы
train_df = pd.DataFrame({
    "Эталонное значение": np.round(Y_tr_true.flatten(), 4),
    "Полученное значение": np.round(train_pred.flatten(), 4),
    "Отклонение": np.round((train_pred - Y_tr_true).flatten(), 4),
})
test_df = pd.DataFrame({
    "Эталонное значение": np.round(Y_te_true.flatten(), 4),
    "Полученное значение": np.round(test_pred.flatten(), 4),
    "Отклонение": np.round((test_pred - Y_te_true).flatten(), 4),
})

print("=" * 60); print("ТАБЛИЦА 1: РЕЗУЛЬТАТЫ ОБУЧЕНИЯ"); print("=" * 60)
print(train_df.to_string())
print("\n" + "=" * 60); print("ТАБЛИЦА 2: РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ"); print("=" * 60)
print(test_df.to_string())

mae_tr = np.mean(np.abs(train_pred - Y_tr_true))
mae_te = np.mean(np.abs(test_pred - Y_te_true))
print(f"\nMAE train = {mae_tr:.4f} | MAE test = {mae_te:.4f}")

# графики
plt.figure(figsize=(10, 4))
plt.plot(x_tr_tgt, Y_tr_true, 'k', label="Эталон (train)")
plt.plot(x_tr_tgt, train_pred, 'b--', label=f"Модель (train), α={best_lr}")
plt.title(f"Прогнозируемая функция на участке обучения (α={best_lr})")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(8, 4))
plt.plot(loss_history, 'r')
plt.title("Изменение ошибки MSE от итерации")
plt.xlabel("Эпоха"); plt.ylabel("MSE"); plt.grid(True); plt.show()