import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

a, b, c, d = 0.4, 0.2, 0.07, 0.2
n_inputs = 8
n_hidden = 3

x = np.linspace(0, 50, 500)
y = a * np.cos(b * x) + c * np.sin(d * x)

X_data, y_data = [], []
for i in range(len(y) - n_inputs):
    X_data.append(y[i : i + n_inputs])
    y_data.append(y[i + n_inputs])

X_data = torch.tensor(np.array(X_data), dtype=torch.float32)
y_data = torch.tensor(np.array(y_data), dtype=torch.float32).unsqueeze(1)

split_idx = int(len(X_data) * 0.8)
X_train, y_train = X_data[:split_idx], y_data[:split_idx]
X_test, y_test = X_data[split_idx:], y_data[split_idx:]

class TimeSeriesNN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(TimeSeriesNN, self).__init__()
        self.hidden = nn.Linear(in_features, hidden_features)
        self.act = nn.Sigmoid()
        self.output = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.act(x)
        x = self.output(x)
        return x

model = TimeSeriesNN(n_inputs, n_hidden)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1500
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

model.eval()
with torch.no_grad():
    train_preds = model(X_train).numpy()
    test_preds = model(X_test).numpy()

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(x[:split_idx+n_inputs], y[:split_idx+n_inputs], label="Участок обучения", color='blue')
plt.plot(x[split_idx+n_inputs:], y[split_idx+n_inputs:], label="Участок прогнозирования", color='orange')
plt.title("Прогнозируемая функция")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_history, color='red')
plt.title("Динамика ошибки обучения")
plt.xlabel("Итерация")
plt.ylabel("MSE")
plt.grid(True)
plt.tight_layout()
plt.show()

df_train = pd.DataFrame({
    'Эталонное значение': y_train.numpy().squeeze(),
    'Полученное значение': train_preds.squeeze(),
    'Отклонение': np.abs(y_train.numpy().squeeze() - train_preds.squeeze())
})

df_test = pd.DataFrame({
    'Эталонное значение': y_test.numpy().squeeze(),
    'Полученное значение': test_preds.squeeze(),
    'Отклонение': np.abs(y_test.numpy().squeeze() - test_preds.squeeze())
})

print("--- РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ---")
print(df_train.to_string(index=False))

print("\n--- РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ ---")
print(df_test.to_string(index=False))