import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
file = pd.read_csv("iris.csv")


# 1. Проверка пропущенных значений
print("Пропущенные значения:")
print(file.isnull().sum())


# 2. Количество образцов каждого вида ириса
print("\nКоличество образцов каждого вида:")
print(file["variety"].value_counts())


# 3. Pair plot для всех признаков
sns.pairplot(file, hue="variety")
plt.show()


# 4. Средние значения признаков для каждого вида
print("\nСредние значения признаков:")
print(file.groupby("variety").mean(numeric_only=True))


# 5. Box plot для Petal Length
sns.boxplot(
    x="variety",
    y="petal.length",
    data=file
)

plt.xlabel("Вид ириса")
plt.ylabel("Petal Length (cm)")
plt.title("Распределение длины лепестка по видам ириса")
plt.show()


# 6. Стандартизация данных
features = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width"
]

df_standardized = file.copy()

df_standardized[features] = (
    file[features] - file[features].mean()
) / file[features].std()

print("\nСтандартизированные данные:")
print(df_standardized)