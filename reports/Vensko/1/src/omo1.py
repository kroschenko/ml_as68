import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "BostonHousing.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1500)

print("Первые 5 строк:")
print(df.head().to_string(index=False))

print("\n1. Основные статистические характеристики:")
print(df.describe().round(3).to_string())

print("\nКоличество пропусков:")
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include="number").columns
correlation_matrix = df[numeric_columns].corr()

print("\n2. Матрица корреляции:")
print(correlation_matrix.round(3).to_string())

plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True
)
plt.title("Матрица корреляции признаков Boston Housing")
plt.tight_layout()
plt.show()

correlations = correlation_matrix["MEDV"].drop(["MEDV", "CAT. MEDV"])
strongest_feature = correlations.abs().idxmax()
strongest_correlation = correlations[strongest_feature]

print("\n3. Корреляция признаков с MEDV:")
print(correlations.sort_values(key=abs, ascending=False).round(4))

print(f"\nПризнак с самой сильной корреляцией: {strongest_feature}")
print(f"Коэффициент корреляции: {strongest_correlation:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(
    df[strongest_feature],
    df["MEDV"],
    color="steelblue",
    alpha=0.5,
    edgecolors="black"
)
plt.title(f"Зависимость MEDV от {strongest_feature}")
plt.xlabel(strongest_feature)
plt.ylabel("MEDV — медианная стоимость жилья")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

df_normalized = df.copy()

for column in numeric_columns:
    minimum = df[column].min()
    maximum = df[column].max()

    if maximum != minimum:
        df_normalized[column] = (df[column] - minimum) / (maximum - minimum)
    else:
        df_normalized[column] = 0.0

print("\n5. Первые 5 строк после нормализации:")
print(df_normalized.head().round(3).to_string(index=False))

print("\nМинимальные и максимальные значения после нормализации:")
print(
    df_normalized[numeric_columns]
    .agg(["min", "max"])
    .round(3)
    .to_string()
)

plt.figure(figsize=(8, 5))
plt.hist(
    df["CRIM"],
    bins=30,
    color="cornflowerblue",
    edgecolor="black"
)
plt.title("Распределение уровня преступности")
plt.xlabel("CRIM — уровень преступности")
plt.ylabel("Количество районов")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()