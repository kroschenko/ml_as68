

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ============================================================
# ЗАДАНИЕ 1. Загрузка данных и проверка пропусков
# ============================================================
print("=" * 60)
print("ЗАДАНИЕ 1. Загрузка данных и проверка пропусков")
print("=" * 60)

# Загружаем встроенный датасет Iris через seaborn (файл iris.csv не нужен)
df = sns.load_dataset('iris')

# Переименовываем колонки под имена, используемые в коде
df = df.rename(columns={
    'species': 'variety',
    'sepal_length': 'sepal.length',
    'sepal_width': 'sepal.width',
    'petal_length': 'petal.length',
    'petal_width': 'petal.width'
})

print("\nПервые 5 строк датасета:")
print(df.head())

print("\nОбщая информация о датасете:")
print(df.info())


# ============================================================
# Описательная статистика
# ============================================================
print("\n" + "=" * 60)
print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА (mean, median, std)")
print("=" * 60)
print(df.describe())

print("\nМедианы по признакам:")
print(df.median(numeric_only=True))

print("\nСтандартные отклонения по признакам:")
print(df.std(numeric_only=True))


# ============================================================
# Обработка пропусков
# ============================================================
print("\n" + "=" * 60)
print("ОБРАБОТКА ПРОПУСКОВ")
print("=" * 60)

if df.isnull().sum().sum() > 0:
    print("Найдены пропуски. Заполняем средним значением...")
    df = df.fillna(df.mean(numeric_only=True))
    print("Пропуски обработаны.")
else:
    print("Пропусков нет. Обработка не требуется.")

print("\nКоличество пропущенных значений по столбцам:")
print(df.isnull().sum())

print(f"\nОбщее количество пропусков: {df.isnull().sum().sum()}")


# ============================================================
# ЗАДАНИЕ 2. Количество образцов каждого вида ириса
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 2. Количество образцов каждого вида ириса")
print("=" * 60)

species_counts = df["variety"].value_counts()
print("\nКоличество образцов по видам:")
print(species_counts)

plt.figure(figsize=(8, 5))
colors = ['#4C72B0', '#DD8452', '#55A868']
bars = plt.bar(species_counts.index, species_counts.values, color=colors)
plt.title("Количество образцов каждого вида ириса", fontsize=14)
plt.xlabel("Вид ириса")
plt.ylabel("Количество образцов")
for bar, val in zip(bars, species_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             str(val), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig("task2_species_count.png", dpi=120)
plt.show()


# ============================================================
# ЗАДАНИЕ 3. Парные диаграммы рассеяния (pair plot)
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 3. Парные диаграммы рассеяния (pair plot)")
print("=" * 60)

pair_plot = sns.pairplot(df, hue="variety", palette="husl",
                         diag_kind="kde", height=2.2)
pair_plot.fig.suptitle("Парные диаграммы рассеяния признаков Iris",
                       y=1.02, fontsize=14)
plt.savefig("task3_pairplot.png", dpi=120, bbox_inches='tight')
plt.show()

print("\nВывод: признаки petal.length и petal.width хорошо разделяют")
print("вид Setosa от остальных. Versicolor и Virginica частично")
print("перекрываются, но образуют различимые кластеры.")


# ============================================================
# ЗАДАНИЕ 4. Средние значения признаков по видам
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 4. Средние значения признаков по видам")
print("=" * 60)

numeric_cols = ["sepal.length", "sepal.width",
                "petal.length", "petal.width"]
means = df.groupby("variety")[numeric_cols].mean()
print("\nСредние значения признаков для каждого вида:")
print(means.round(3))


# ============================================================
# ЗАДАНИЕ 5. Box plot для Petal Length по видам
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 5. Box plot для Petal Length по видам")
print("=" * 60)

plt.figure(figsize=(9, 6))
sns.boxplot(x="variety", y="petal.length", data=df, palette="Set2")
plt.title("Распределение длины лепестка (Petal Length) по видам ириса",
          fontsize=14)
plt.xlabel("Вид ириса")
plt.ylabel("Длина лепестка (см)")
plt.tight_layout()
plt.savefig("task5_boxplot_petal_length.png", dpi=120)
plt.show()

print("\nВывод: у Setosa длина лепестка заметно меньше и имеет")
print("наименьший разброс. У Virginica — наибольшая медиана и")
print("наибольший разброс значений.")


# ============================================================
# ЗАДАНИЕ 6. Стандартизация данных
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 6. Стандартизация данных")
print("=" * 60)

# --- Способ 1: ручная стандартизация ---
df_standardized = df.copy()
for col in numeric_cols:
    df_standardized[col] = (df[col] - df[col].mean()) / df[col].std()

print("\nСтатистика стандартизированных признаков:")
print(df_standardized[numeric_cols].describe().round(3))

print("\nСредние значения после стандартизации:")
print(df_standardized[numeric_cols].mean().round(6))

print("\nСтандартные отклонения после стандартизации:")
print(df_standardized[numeric_cols].std().round(6))

# --- Способ 2: через sklearn ---
scaler = StandardScaler()
scaled_sklearn = scaler.fit_transform(df[numeric_cols])
print("\nПроверка через sklearn StandardScaler (первые 3 строки):")
print(pd.DataFrame(scaled_sklearn[:3],
                   columns=numeric_cols).round(3))

# --- Визуализация до/после ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, col in enumerate(numeric_cols):
    axes[0, i].hist(df[col], bins=15,
                    color='#4C72B0', edgecolor='black')
    axes[0, i].set_title(f"Исходные: {col}", fontsize=10)

    axes[1, i].hist(df_standardized[col], bins=15,
                    color='#55A868', edgecolor='black')
    axes[1, i].set_title(f"Стандартизированные: {col}", fontsize=10)

plt.suptitle("Сравнение распределений до и после стандартизации",
             fontsize=14)
plt.tight_layout()
plt.savefig("task6_standardization.png", dpi=120)
plt.show()


# ============================================================
# ДОПОЛНИТЕЛЬНО. One-Hot Encoding признака variety
# ============================================================
print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНО. One-Hot Encoding признака variety")
print("=" * 60)

df_encoded = pd.get_dummies(df, columns=["variety"], prefix="variety")
print("\nРезультат One-Hot Encoding (первые 5 строк):")
print(df_encoded.head())

print("\n" + "=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
print("=" * 60)