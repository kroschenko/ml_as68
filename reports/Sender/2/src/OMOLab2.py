import matplotlib
matplotlib.use("Agg")  # Сохраняем графики в файлы (если нужно видеть окна, закомментируй эту строку)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------- ЧАСТЬ 1. РЕГРЕССИЯ: Car Price Prediction ----------
def run_regression_task():
    print("=" * 60)
    print("ЧАСТЬ 1. РЕГРЕССИЯ: Car Price Prediction")
    print("=" * 60)

    df = pd.read_csv("CarPrice_Assignment.csv")

    # 5–6 числовых признаков
    features = ["horsepower", "enginesize", "citympg", "highwaympg", "curbweight"]
    target = "price"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"R²  = {r2_score(y_test, y_pred):.4f}")
    print(f"MAE = {mean_absolute_error(y_test, y_pred):.2f}\n")

    # Scatter price vs horsepower + линия парной регрессии
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test["horsepower"], y=y_test, color="blue", alpha=0.7,
                    label="Тестовая выборка")
    
    single = LinearRegression()
    single.fit(X_train[["horsepower"]], y_train)
    x_line = np.linspace(X["horsepower"].min(), X["horsepower"].max(), 100)
    y_line = single.predict(x_line.reshape(-1, 1))
    
    plt.plot(x_line, y_line, color="red", linewidth=2.5,
             label="Линия парной регрессии")
    
    plt.title("Цена автомобиля vs мощность (horsepower)")
    plt.xlabel("horsepower")
    plt.ylabel("price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("regression_scatter.png", dpi=150)
    print("График регрессии сохранен как 'regression_scatter.png'\n")


# ---------- ЧАСТЬ 2. КЛАССИФИКАЦИЯ: Adult Census Income ----------
def run_classification_task():
    print("=" * 60)
    print("ЧАСТЬ 2. КЛАССИФИКАЦИЯ: Adult Census Income")
    print("=" * 60)

    df = pd.read_csv("adult.csv")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace("?", "Unknown")

    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    X = df.drop(columns=["income"])
    y = df["income"]

    # one-hot для категориальных
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Масштабирование (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log = LogisticRegression(max_iter=2000, random_state=42)
    log.fit(X_train, y_train)
    y_pred = log.predict(X_test)

    print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision = {precision_score(y_test, y_pred):.4f} (класс >50K)")
    print(f"Recall    = {recall_score(y_test, y_pred):.4f} (класс >50K)\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"])
    
    plt.title("Матрица ошибок (Adult Census Income)")
    plt.ylabel("Реальные классы")
    plt.xlabel("Предсказанные классы")
    plt.tight_layout()
    plt.savefig("classification_confusion_matrix.png", dpi=150)
    print("Матрица ошибок сохранена как 'classification_confusion_matrix.png'\n")


if __name__ == "__main__":
    run_regression_task()
    print()
    run_classification_task()