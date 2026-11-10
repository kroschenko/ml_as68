import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# фиксируем seed, чтобы сравнение моделей было честным
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# загружаем датасет digits
digits = load_digits()
X, y = digits.data, digits.target

# стандартизация признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# делим на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# переводим в тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

n_features = X_train.shape[1]
n_classes = len(np.unique(y))


# модель с одним скрытым слоем на 32 нейрона
class MLPOneHidden(nn.Module):
    def __init__(self, n_features, n_classes, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# модель с двумя скрытыми слоями по 32 нейрона
class MLPTwoHidden(nn.Module):
    def __init__(self, n_features, n_classes, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print("epoch", epoch + 1, "loss", round(loss.item(), 4))

    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1).numpy()
    y_true = y_test.numpy()
    acc = accuracy_score(y_true, preds)
    return acc


EPOCHS = 100
LR = 0.001

print("model A: 1 hidden layer, 32 neurons")
model_a = MLPOneHidden(n_features, n_classes, hidden=32)
train_model(model_a, X_train, y_train, epochs=EPOCHS, lr=LR)
acc_a = evaluate_model(model_a, X_test, y_test)

print()
print("model B: 2 hidden layers, 32 neurons each")
model_b = MLPTwoHidden(n_features, n_classes, hidden=32)
train_model(model_b, X_train, y_train, epochs=EPOCHS, lr=LR)
acc_b = evaluate_model(model_b, X_test, y_test)

print()
print("accuracy A:", round(acc_a, 4))
print("accuracy B:", round(acc_b, 4))
print("diff B - A:", round(acc_b - acc_a, 4))