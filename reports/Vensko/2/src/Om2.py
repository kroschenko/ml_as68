from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# CSV-файлы должны находиться в одной папке с этим Python-файлом.
BASE_DIR = Path(__file__).resolve().parent
INSURANCE_FILE = BASE_DIR / "medical_cost_personal_dataset.csv"
HEART_FILE = BASE_DIR / "heart_disease_uci.csv"


def check_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {file_path.name}. "
            "Поместите CSV-файл в одну папку с variant2_ml.py."
        )


def regression_task():
    print("\nРЕГРЕССИЯ: ПРОГНОЗИРОВАНИЕ МЕДИЦИНСКИХ РАСХОДОВ ")

    check_file(INSURANCE_FILE)
    data = pd.read_csv(INSURANCE_FILE)
    print("Размер датасета:", data.shape)

    categorical_features = ["sex", "smoker", "region"]
    numerical_features = ["age", "bmi", "children"]

    X = data[categorical_features + numerical_features]
    y = data["charges"]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_features),
        ("numerical", numerical_pipeline, numerical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nМетрики качества линейной регрессии:")
    print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
    print("R2:", round(r2_score(y_test, predictions), 4))

    bmi_model = LinearRegression()
    bmi_model.fit(data[["bmi"]], y)

    bmi_values = np.linspace(data["bmi"].min(), data["bmi"].max(), 200)
    charges_line = bmi_model.predict(pd.DataFrame({"bmi": bmi_values}))

    plt.figure(figsize=(9, 6))
    plt.scatter(data["bmi"], y, alpha=0.6, label="Фактические значения")
    plt.plot(bmi_values, charges_line, linewidth=2, label="Линия регрессии")
    plt.title("Зависимость медицинских расходов от BMI")
    plt.xlabel("BMI (индекс массы тела)")
    plt.ylabel("Charges (медицинские расходы)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "regression_bmi.png", dpi=300)
    plt.show()
    plt.close()


def classification_task():
    print("\nКЛАССИФИКАЦИЯ: ДИАГНОСТИКА ЗАБОЛЕВАНИЙ СЕРДЦА")

    check_file(HEART_FILE)
    data = pd.read_csv(HEART_FILE)
    print("Размер датасета:", data.shape)
    data["target"] = (data["num"] > 0).astype(int)

    categorical_features = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]
    numerical_features = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    feature_columns = categorical_features + numerical_features

    X = data[feature_columns]
    y = data["target"]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_features),
        ("numerical", numerical_pipeline, numerical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classification", LogisticRegression(max_iter=2000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nМетрики качества логистической регрессии:")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1-score:", round(f1_score(y_test, predictions, zero_division=0), 4))

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Нет болезни", "Есть болезнь"],
    )
    display.plot(values_format="d")
    plt.title("Матрица ошибок логистической регрессии")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=300)
    plt.show()
    plt.close()


def main():
    regression_task()
    classification_task()
    print("\nРабота завершена.")

if __name__ == "__main__":
    main()
