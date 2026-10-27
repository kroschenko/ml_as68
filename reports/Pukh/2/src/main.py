import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# =====================================================
# Часть 1. Регрессия: прогнозирование цены автомобиля
# =====================================================

# Загрузка данных
cars = pd.read_csv("CarPrice_Assignment.csv")

print("Car Price: размер данных", cars.shape)
print(cars.head())
print("\nПропуски:", cars.isnull().sum().sum())

# Выбор 6 числовых признаков
features = [
    "horsepower",
    "enginesize",
    "curbweight",
    "carwidth",
    "citympg",
    "highwaympg"
]

X = cars[features]
y = cars["price"]

print("\nКорреляция признаков с ценой:")
print(cars[features + ["price"]].corr()["price"].round(3))

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели линейной регрессии
reg = LinearRegression()
reg.fit(X_train, y_train)

# Предсказание и оценка качества
y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nКоэффициенты модели:")
for name, coef in zip(features, reg.coef_):
    print(f"  {name}: {coef:.2f}")
print(f"  intercept: {reg.intercept_:.2f}")

print(f"\nR2:  {r2:.3f}")
print(f"MAE: {mae:.2f}")

# Визуализация: price от horsepower с линией регрессии
# (линия строится по модели с одним признаком horsepower)
hp_model = LinearRegression()
hp_model.fit(cars[["horsepower"]], y)

hp_range = pd.DataFrame({
    "horsepower": range(cars["horsepower"].min(),
                        cars["horsepower"].max() + 1)
})

plt.figure(figsize=(10, 6))
plt.scatter(cars["horsepower"], y, alpha=0.6, label="Автомобили")
plt.plot(hp_range["horsepower"], hp_model.predict(hp_range),
         color="red", linewidth=2, label="Линия регрессии")
plt.xlabel("Мощность двигателя (horsepower)")
plt.ylabel("Цена автомобиля (price)")
plt.title("Зависимость цены автомобиля от мощности")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("01_price_horsepower.png")
plt.close()

print(f"\nЛиния регрессии: price = {hp_model.coef_[0]:.2f} * horsepower "
      f"+ ({hp_model.intercept_:.2f})")


# =====================================================
# Часть 2. Классификация: прогнозирование дохода >50K
# =====================================================

# Загрузка данных, "?" считаем пропуском
adult = pd.read_csv("adult.csv", na_values=["?"])

print("\n\nAdult: размер данных", adult.shape)
print(adult.head())

print("\nПропуски до обработки:")
print(adult.isnull().sum()[adult.isnull().sum() > 0])

# Заполнение пропусков самым частым значением (модой)
for col in ["workclass", "occupation", "native.country"]:
    adult[col] = adult[col].fillna(adult[col].mode()[0])

print("Пропусков после обработки:", adult.isnull().sum().sum())

# Целевая переменная: 1 — доход >50K, 0 — <=50K
adult["income"] = (adult["income"] == ">50K").astype(int)
print("\nРаспределение классов:")
print(adult["income"].value_counts())

# Преобразование категориальных признаков (One-Hot Encoding)
X = adult.drop(columns=["income"])
X = pd.get_dummies(X, drop_first=True, dtype=int)
y = adult["income"]

print("Признаков после кодирования:", X.shape[1])

# Разделение на выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Стандартизация (нужна для сходимости логистической регрессии)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели логистической регрессии
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Предсказание и метрики
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\nAccuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Матрица ошибок (логистическая регрессия)")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.tight_layout()
plt.savefig("02_confusion_matrix.png")
plt.close()

print("\nГрафики сохранены.")
