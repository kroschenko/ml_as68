import pandas as pd
import matplotlib.pyplot as plt

# Задание 1. Загрузка данных
df = pd.read_csv("auto-mpg.csv", na_values=["?"])

# Поддержка разных названий столбцов
df = df.rename(columns={
    "model_year": "model year",
    "name": "car name"
})

print("Источник данных: auto-mpg.csv")

print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о данных:")
df.info()

print("\nКоличество пропусков:")
print(df.isnull().sum())

print("\nСтатистика:")
print(df.describe(include="all"))

print("\nМедиана:")
print(df.median(numeric_only=True))

print("\nСтандартное отклонение:")
print(df.std(numeric_only=True))


# Задание 2. Обработка horsepower
df["horsepower"] = pd.to_numeric(
    df["horsepower"], errors="coerce"
)

mean_hp = df["horsepower"].mean()

df["horsepower"] = df["horsepower"].fillna(mean_hp)

print("\nСреднее horsepower:", round(mean_hp, 2))
print("Пропусков после обработки:",
      df["horsepower"].isnull().sum())


# Задание 3. Зависимость mpg от weight
plt.figure(figsize=(10, 6))

plt.scatter(
    df["weight"],
    df["mpg"],
    alpha=0.6
)

plt.xlabel("Вес автомобиля (weight)")
plt.ylabel("Мили на галлон (mpg)")
plt.title("Зависимость mpg от веса автомобиля")
plt.grid(True)
plt.tight_layout()

plt.savefig("01_weight_mpg.png")
plt.close()

correlation = df["mpg"].corr(df["weight"])

print("\nКорреляция mpg и weight:",
      round(correlation, 3))


# Задание 4. Кодирование origin
print("\nСтраны производства:")
print(df["origin"].value_counts())

origin_encoded = pd.get_dummies(
    df["origin"],
    prefix="origin",
    dtype=int
)

df = pd.concat([df, origin_encoded], axis=1)

print("\nЗакодированные значения origin:")
print(origin_encoded.head())


# Задание 5. Возраст автомобиля
# Годы в данных представлены числами 70-82
df["age"] = 1983 - (1900 + df["model year"])

print("\nВозраст автомобилей:")
print(df[["model year", "age"]].head())


# Задание 6. Распределение цилиндров
counts = df["cylinders"].value_counts().sort_index()

plt.figure(figsize=(10, 6))

plt.bar(
    counts.index.astype(str),
    counts.values
)

plt.xlabel("Количество цилиндров")
plt.ylabel("Количество автомобилей")
plt.title("Распределение автомобилей по числу цилиндров")
plt.tight_layout()

plt.savefig("02_cylinders.png")
plt.close()

print("\nРаспределение цилиндров:")
print(counts)


# Нормализация числовых признаков
numeric_cols = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "age"
]

normalized = df[numeric_cols].copy()

for col in numeric_cols:
    min_val = normalized[col].min()
    max_val = normalized[col].max()

    normalized[col] = (
        (normalized[col] - min_val) / (max_val - min_val)
    )

print("\nНормализованные данные:")
print(normalized.head())


# Сохранение результатов
df.to_csv(
    "auto_mpg_processed.csv",
    index=False
)

normalized.to_csv(
    "auto_mpg_normalized.csv",
    index=False
)

print("\nФайлы CSV и графики сохранены.")