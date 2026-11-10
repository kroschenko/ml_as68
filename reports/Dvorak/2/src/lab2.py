
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

print("Библиотеки успешно загружены!")


# Задание 1: РЕГРЕССИЯ — FISH MARKET

# 1. Загрузка данных
df_fish = pd.read_csv("Fish.csv")

print("\nПервые 5 строк датасета Fish Market:")
print(df_fish.head())

print("\nРазмер датасета:", df_fish.shape)

# 2. Выбор признаков и целевой переменной
features = ["Length1", "Length2", "Length3", "Height", "Width"]
target = "Weight"

X = df_fish[features]
y = df_fish[target]

# 3. Разделение на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=74
)

# 4. Обучение модели линейной регрессии
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# 5. Предсказания на тестовой выборке
y_pred = reg_model.predict(X_test)

# 6. Расчёт метрик R² и RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nРезультаты модели регрессии")
print(f"R² (Coefficient of Determination): {r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

# 7. Визуализация Length3 и Weight с линией регрессии
plt.figure(figsize=(10, 6))

sns.regplot(
    data=df_fish,
    x="Length3",
    y="Weight",
    scatter_kws={"alpha": 0.7, "color": "blue"},
    line_kws={"color": "red", "linewidth": 2}
)

plt.title(
    "Зависимость веса рыбы от длины Length3",
    fontsize=14
)
plt.xlabel("Length3 (см)", fontsize=12)
plt.ylabel("Weight (г)", fontsize=12)
plt.tight_layout()
plt.savefig("fish_regression.png", dpi=300)
plt.show()



# Задание 2: КЛАССИФИКАЦИЯ — BANK MARKETING


# 1. Загрузка данных
df_bank = pd.read_csv("bank.csv", sep=";")

print("\nПервые 5 строк датасета Bank Marketing:")
print(df_bank.head())

print("\nРазмер датасета:", df_bank.shape)

# 2. Подготовка данных
df_bank.columns = df_bank.columns.str.strip()

# Удаляем лишние пробелы в текстовых значениях
for column in df_bank.select_dtypes(include=["object"]).columns:
    df_bank[column] = df_bank[column].str.strip()

# Целевая переменная: yes = 1, no = 0
df_bank["y"] = df_bank["y"].str.lower().map({
    "yes": 1,
    "no": 0
})

# Преобразуем категориальные признаки в числовые
df_bank_encoded = pd.get_dummies(
    df_bank,
    columns=df_bank.select_dtypes(
        include=["object", "category"]
    ).columns.tolist(),
    drop_first=True,
    dtype=int
)

# 3. Выбор признаков и целевой переменной
X_class = df_bank_encoded.drop("y", axis=1)
y_class = df_bank_encoded["y"]

# 4. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# 5. Стандартизация признаков
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Обучение модели логистической регрессии
log_model = LogisticRegression(
    max_iter=10000,
    random_state=42
)
log_model.fit(X_train_scaled, y_train)

# 7. Предсказания на тестовой выборке
y_pred_c = log_model.predict(X_test_scaled)

# 8. Расчёт метрик
acc = accuracy_score(y_test, y_pred_c)

precision = precision_score(
    y_test, y_pred_c, zero_division=0
)

recall = recall_score(
    y_test, y_pred_c, zero_division=0
)

print("\nРезультаты модели классификации")
print(f"Accuracy: {acc:.4f}")
print(f"Precision для класса 'yes': {precision:.4f}")
print(f"Recall для класса 'yes': {recall:.4f}")

# 9. Построение матрицы ошибок
cm = confusion_matrix(
    y_test,
    y_pred_c,
    labels=[0, 1]
)

plt.figure(figsize=(6, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["no", "yes"],
    yticklabels=["no", "yes"]
)

plt.title("Матрица ошибок (Confusion Matrix)", fontsize=14)
plt.xlabel("Предсказано моделью", fontsize=12)
plt.ylabel("Реальные данные", fontsize=12)
plt.tight_layout()
plt.savefig("bank_confusion_matrix.png", dpi=300)
plt.show()

print("\nРабота завершена!")
