# Лабораторная работа №4. Вариант 1: классификация ирисов (MLP, PyTorch)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HERE = Path(__file__).resolve().parent   # папка со скриптом: графики сохраняем сюда


def find_csv(name="iris.csv"):
    """Ищем iris.csv в папке скрипта, затем в родительских папках."""
    for d in [HERE, *HERE.parents[:4]]:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} не найден рядом со скриптом или в родительских папках ({HERE})")


SEED = 42
EPOCHS = 500
LR = 0.001


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# 1. Импорт и подготовка данных
df = pd.read_csv(find_csv())
X = df.drop(columns="variety").values
y = LabelEncoder().fit_transform(df["variety"])       # Setosa/Versicolor/Virginica -> 0/1/2
class_names = sorted(df["variety"].unique())

# То же разбиение, что и в ЛР№3: 70/30 со стратификацией
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)

# Стандартизация: параметры считаем только по train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Тензоры PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)    # для CrossEntropyLoss метки - long
y_test_t = torch.tensor(y_test, dtype=torch.long)
print("train:", X_train_t.shape, "test:", X_test_t.shape, "\n")


# 2. Архитектуры
class MLP(nn.Module):
    """Основная: 4 -> 10 (ReLU) -> 3 (Softmax)."""
    def __init__(self, n_in=4, n_hidden=10, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        # Возвращаем логиты: Softmax встроен в nn.CrossEntropyLoss,
        # вероятности получаем отдельно через torch.softmax при предсказании
        return self.out(self.relu(self.fc1(x)))


class MLP2(nn.Module):
    """Эксперимент: 4 -> 10 (ReLU) -> 5 (ReLU) -> 3 (Softmax)."""
    def __init__(self, n_in=4, n_h1=10, n_h2=5, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n_h2, n_out)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


# 3-4. Обучение
def train_model(model_cls, seed=SEED, epochs=EPOCHS, verbose=False):
    set_seed(seed)
    model = model_cls()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        y_pred = model(X_train_t)                # forward pass
        loss = criterion(y_pred, y_train_t)      # потери
        optimizer.zero_grad()                    # обнуление градиентов
        loss.backward()                          # обратное распространение
        optimizer.step()                         # шаг оптимизации

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_t), y_test_t).item()
        train_losses.append(loss.item())
        test_losses.append(test_loss)

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}: train loss = {loss.item():.4f}, test loss = {test_loss:.4f}")
    return model, train_losses, test_losses


# 5. Оценка
def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).numpy()
    return pred, probs.numpy()


def report(name, model):
    pred, _ = evaluate(model)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print("=" * 60)
    print(f"{name}: accuracy = {acc:.4f}, macro F1 = {f1:.4f}")
    print("Матрица ошибок:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=class_names, digits=3))
    n_params = sum(p.numel() for p in model.parameters())
    print("Число параметров:", n_params, "\n")
    return acc, f1


print("Обучение модели 1: 4-10-3")
model1, tr1, te1 = train_model(MLP, verbose=True)
acc1, f1_1 = report("MLP (1 скрытый слой, 10 нейронов)", model1)

print("Обучение модели 2: 4-10-5-3")
model2, tr2, te2 = train_model(MLP2, verbose=True)
acc2, f1_2 = report("MLP (2 скрытых слоя: 10 и 5 нейронов)", model2)

# Тест из 45 объектов мал, поэтому повторяем обучение с разными seed
print("Устойчивость: 10 запусков с разной инициализацией весов")
seeds = range(10)
accs1, accs2 = [], []
for s in seeds:
    m1, _, _ = train_model(MLP, seed=s)
    m2, _, _ = train_model(MLP2, seed=s)
    accs1.append(accuracy_score(y_test, evaluate(m1)[0]))
    accs2.append(accuracy_score(y_test, evaluate(m2)[0]))
print(f"  4-10-3   : accuracy = {np.mean(accs1):.4f} ± {np.std(accs1):.4f}")
print(f"  4-10-5-3 : accuracy = {np.mean(accs2):.4f} ± {np.std(accs2):.4f}\n")

# Графики
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (tr, te, title) in zip(axes, [(tr1, te1, "4-10-3"), (tr2, te2, "4-10-5-3")]):
    ax.plot(tr, label="train")
    ax.plot(te, label="test", ls="--")
    ax.set_title(f"Функция потерь: {title}")
    ax.set_xlabel("Эпоха"); ax.grid(alpha=.3); ax.legend()
axes[0].set_ylabel("CrossEntropy loss")
plt.tight_layout(); plt.savefig(HERE / "mlp_loss.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
names = ["1 скрытый\n(10)", "2 скрытых\n(10, 5)"]
bars = plt.bar(names, [acc1, acc2], color=["#4c72b0", "#55a868"])
plt.ylim(0.8, 1.02); plt.ylabel("Accuracy (test)"); plt.title("Сравнение архитектур MLP")
for b, v in zip(bars, [acc1, acc2]):
    plt.text(b.get_x() + b.get_width() / 2, v + .005, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig(HERE / "mlp_compare.png", dpi=150); plt.close()