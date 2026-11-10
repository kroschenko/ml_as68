import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)

# данные
a = 0.2
b = 0.2
c = 0.06
d = 0.2
x = np.arange(0, 150, 0.5)  
y = a * np.cos(b * x) + c * np.sin(d * x)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

WINDOW_SIZE = 8

def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])
    return np.array(X), np.array(Y).reshape(-1, 1)

X, Y = create_dataset(y_scaled, WINDOW_SIZE)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]
split_idx = train_size + WINDOW_SIZE

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

# архитектура
class RegressionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(WINDOW_SIZE, 3)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(3, 1) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

model = RegressionMLP()
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

# обучение
epochs = 1000
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, Y_train_t)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# оценка
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).numpy().flatten()
    test_pred = model(X_test_t).numpy().flatten()

Y_train_orig = scaler.inverse_transform(Y_train).flatten()
train_pred_orig = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()

Y_test_orig = scaler.inverse_transform(Y_test).flatten()
test_pred_orig = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

# прогнозируемая функция
plt.figure(figsize=(10, 5))

x_train_plot = x[WINDOW_SIZE:split_idx]

plt.plot(x_train_plot, Y_train_orig, color='black', label='Эталон (train)')
plt.plot(x_train_plot, train_pred_orig, color='blue', linestyle='--', label='Модель (train)')

plt.title("График прогнозируемой функции на участке обучения")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='red')
plt.title("График изменения ошибки (MSE) от итерации")
plt.xlabel("Итерация")
plt.ylabel("Ошибка (MSE)")
plt.grid(True)
plt.show()


# таблицы
def print_table(orig, pred, title):
    df = pd.DataFrame({
        "Эталонное значение": orig,
        "Полученное значение": pred,
        "Отклонение": np.abs(orig - pred)
    })
    print(f"\n--- {title} (первые 10 строк) ---")
    print(df.head(10).round(4).to_string(index=False))

print_table(Y_train_orig, train_pred_orig, "Результаты обучения")
print_table(Y_test_orig, test_pred_orig, "Результаты прогнозирования")