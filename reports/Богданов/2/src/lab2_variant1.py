import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# ============ РЕГРЕССИЯ: California Housing ============

# загрузка данных
df = pd.read_csv("california_housing.csv")

# (технически необходимая обработка: пропуски и текстовый признак)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

X = df.drop(columns="median_house_value")
y = df["median_house_value"]

# разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# обучение линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# предсказания для тестовой выборки
y_pred = model.predict(X_test)

# метрики MSE и R²
print("MSE =", mean_squared_error(y_test, y_pred))
print("R2  =", r2_score(y_test, y_pred))

# диаграмма рассеяния median_income / median_house_value с линией регрессии
line = LinearRegression().fit(X_train[["median_income"]], y_train)
xs = pd.DataFrame({"median_income": np.linspace(X["median_income"].min(),
                                                X["median_income"].max(), 100)})
plt.figure(figsize=(9, 6))
plt.scatter(X["median_income"], y, s=6, alpha=0.3)
plt.plot(xs["median_income"], line.predict(xs), color="red")
plt.xlabel("median_income")
plt.ylabel("median_house_value")
plt.title("California Housing: median_income vs median_house_value")
plt.show()

# ============ КЛАССИФИКАЦИЯ: Titanic ============

# загрузка и предварительная обработка
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
df["Age"] = df["Age"].fillna(df["Age"].median())               # заполнение пропусков
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})            # категории -> числа
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

X = df.drop(columns="Survived")
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# обучение логистической регрессии
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy, Precision, Recall
print("Accuracy  =", accuracy_score(y_test, y_pred))
print("Precision =", precision_score(y_test, y_pred))
print("Recall    =", recall_score(y_test, y_pred))

# матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=["Не выжил", "Выжил"]).plot(cmap="Blues")
plt.title("Titanic: матрица ошибок")
plt.show()
