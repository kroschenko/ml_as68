import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# 1. Загрузка и очистка
df = pd.read_csv("breast_cancer.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == "id"])

if "diagnosis" in df.columns:
    df = df.rename(columns={"diagnosis": "target"})
df["target"] = df["target"].astype(str).str.strip().map({"M": 0, "B": 1})
df = df.dropna()

X = df.drop(columns=["target"])
y = df["target"]

# 2. Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=52, stratify=y
)

# 3. Стандартизация и обучение
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=52)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 4. Вывод метрик
print(f"Общая точность: {accuracy_score(y_test, y_pred):.4f}")
print(f"Точность предсказания рака: {precision_score(y_test, y_pred, pos_label=0):.4f}")
print(f"Полнота (сколько найдено больных раком): {recall_score(y_test, y_pred, pos_label=0):.4f}")

# 5. Матрица ошибок
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, 
    display_labels=["Доброкачественная", "Злокачественная"]
)
plt.title("Матрица ошибок — Рак груди")
plt.show()