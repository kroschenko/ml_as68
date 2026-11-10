import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, accuracy_score

class WineQualityNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(11, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
df = pd.read_csv("winequality-white (2).csv", sep=";")

df.columns = df.columns.str.strip()

df = df.dropna(subset=["quality"])

X = df.drop(columns=["quality"])
y = (df["quality"] >= 7).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = WineQualityNN()
criterion = nn.BCEWithLogitsLoss()         
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

for epoch in range(epochs):
    model.train()

    for x_batch, y_batch in loader:
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():
    y_pred = model(X_test)
    predicted = (y_pred > 0).int()

y_true = y_test.numpy().flatten()
y_pred_classes = predicted.numpy().flatten()

accuracy = accuracy_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes)

print("Accuracy:", accuracy)
print("F1-score:", f1)