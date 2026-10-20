import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("pima-indians-diabetes.csv")

df.columns = df.columns.str.strip()

print("Первые строки данных:")
print(df.head())

print("\nРазмер датасета:", df.shape)

zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in zero_as_missing:
    df[column] = df[column].replace(0, float("nan"))

print("\nПропуски до обработки:")
print(df.isnull().sum())

df = df.dropna(subset=["Outcome"])

X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("\nМодель успешно обучена.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

print("\nМетрики качества модели:")
print("Accuracy =", round(accuracy, 3))
print("Precision =", round(precision, 3))
print("Recall =", round(recall, 3))

print("\nПодробный отчёт:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Матрица ошибок:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Нет диабета (0)", "Диабет (1)"],
    yticklabels=["Нет диабета (0)", "Диабет (1)"]
)

plt.title("Матрица ошибок — диагностика диабета")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.tight_layout()
plt.show()
