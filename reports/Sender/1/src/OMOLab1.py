# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Создаем папку для графиков, если ее нет
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("ЗАДАНИЕ 1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

# Загружаем данные
df = pd.read_csv("auto-mpg.csv")

print("\nПервые 5 строк:")
print(df.head())

print("\nТипы столбцов:")
print(df.dtypes)

print("\nИнформация о столбцах (.info()):")
df.info()

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2. ОБРАБОТКА ПРОПУСКОВ В HORSEPOWER")
print("=" * 60)

# Заменяем '?' на NaN и приводим к числовому типу
df["horsepower"] = df["horsepower"].astype(str).str.strip().replace("?", np.nan)
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

print("Пропусков до:", df["horsepower"].isna().sum())
hp_mean = df["horsepower"].mean()
print("Среднее horsepower:", round(hp_mean, 2))

df["horsepower"] = df["horsepower"].fillna(hp_mean)
print("Пропусков после:", df["horsepower"].isna().sum())

print("\n" + "=" * 60)
print("ЗАДАНИЕ 3. ДИАГРАММА РАССЕЯНИЯ (mpg от weight)")
print("=" * 60)

corr = df["mpg"].corr(df["weight"])
print("Корреляция mpg и weight:", round(corr, 3))

plt.figure(figsize=(8, 6))
plt.scatter(df["weight"], df["mpg"])
plt.title("Зависимость расхода топлива (mpg) от веса автомобиля (weight)")
plt.xlabel("weight")
plt.ylabel("mpg")
plt.tight_layout()
plt.savefig("plots/scatter_mpg_weight.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("ЗАДАНИЕ 4. ПРЕОБРАЗОВАНИЕ ORIGIN (One-Hot Encoding)")
print("=" * 60)

df = pd.get_dummies(df, columns=["origin"], prefix="origin")
for c in [c for c in df.columns if c.startswith("origin_")]:
    df[c] = df[c].astype(int)

print("Столбцы origin:", [c for c in df.columns if c.startswith("origin_")])

print("\n" + "=" * 60)
print("ЗАДАНИЕ 5. НОВЫЙ ПРИЗНАК AGE")
print("=" * 60)

# model year в файле двузначный (70..82) -> приводим к полному году (+1900)
df["age"] = 1983 - (df["model year"] + 1900)
print("age: min =", df["age"].min(), ", max =", df["age"].max())

print("\n" + "=" * 60)
print("ЗАДАНИЕ 6. РАСПРЕДЕЛЕНИЕ CYLINDERS (Bar chart)")
print("=" * 60)

counts = df["cylinders"].value_counts().sort_index()
print("\nРаспределение cylinders:")
print(counts)

plt.figure(figsize=(7, 5))
plt.bar(counts.index.astype(str), counts.values)
plt.title("Распределение количества цилиндров (cylinders)")
plt.xlabel("Количество цилиндров")
plt.ylabel("Количество автомобилей")
plt.tight_layout()
plt.savefig("plots/bar_cylinders.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("ГОТОВО. Графики сохранены в папку plots/")
print("=" * 60)