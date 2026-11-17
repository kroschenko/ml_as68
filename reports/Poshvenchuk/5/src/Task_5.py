import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Фиксация сидов
torch.manual_seed(52)
np.random.seed(52)

# 1. Параметры
a, b, c, d = 0.2, 0.6, 0.05, 0.6
N_INPUTS = 10
ITERATIONS = 500
x = np.arange(0, 150, 0.5)
y = a * np.cos(b * x) + c * np.sin(d * x)

# 2. Формирование выборки
def make_dataset(data, window):
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        Y.append(data[i+window])
    return np.array(X), np.array(Y).reshape(-1, 1)

X, Y = make_dataset(y, N_INPUTS)

# Разделение на Train и Test
train_split = int(len(X) * 0.8)
X_train, X_test = X[:train_split], X[train_split:]
Y_train, Y_test = Y[:train_split], Y[train_split:]
plot_idx = train_split + N_INPUTS

# Перевод в тензоры
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

# 3. Архитектура сети (10 входов -> 4 скрытых нейрона -> 1 выход)
class PredictorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(N_INPUTS, 4)
        self.act = nn.Sigmoid()
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.act(x)
        return self.output(x)

model = PredictorNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Цикл обучения
loss_history = []

for _ in range(ITERATIONS):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, Y_train_t)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# 5. Оценка и возврат к исходному масштабу
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).numpy().flatten()
    test_pred = model(X_test_t).numpy().flatten()

Y_train_orig = Y_train.flatten()
train_pred_orig = train_pred.flatten()

Y_test_orig = Y_test.flatten()
test_pred_orig = test_pred.flatten()

# 6. Визуализация
plt.figure(figsize=(12, 6))
x_train_plot = x[N_INPUTS:plot_idx]
plt.plot(x_train_plot, Y_train_orig, color='lime', label='Эталон')
plt.plot(x_train_plot, train_pred_orig, color='crimson', linestyle='--', label='Прогноз', linewidth=1.0)
plt.title("График прогнозируемой функции на участке обучения")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(loss_history, color='crimson')
plt.title("График изменения ошибки в зависимости от итерации")
plt.xlabel("Итерация")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# 7. Вывод таблиц
def print_table(orig, pred, title):
    df = pd.DataFrame({
        "Эталонное значение": orig,
        "Полученное значение": pred,
        "Отклонение": np.abs(orig - pred)
    })
    print(f"\n{title}:")
    print(df.head(30).round(4).to_string(index=False))

print_table(Y_train_orig, train_pred_orig, "Результаты обучения")
print_table(Y_test_orig, test_pred_orig, "Результаты прогнозирования")