import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("BostonHousing.csv")

print(df.describe())

print(df.dtypes)

print(df.isnull())


corr = df.corr()
corr_with_medv = df.corr(numeric_only=True)['MEDV'].sort_values(ascending=False)
print(corr_with_medv)


plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Матрица корреляции")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['MEDV'], df['LSTAT'], s=20, c='steelblue', alpha=0.6, edgecolors='black', linewidths=0.3)
plt.xlabel('MEDV')
plt.ylabel('LSTAT')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['CRIM'], bins= 30, color='steelblue', edgecolor='black')
plt.xlabel('Уровень преступности')
plt.ylabel('Частота')
plt.title('Распределение уровня преступности')
plt.show()

numerics_cols = df.select_dtypes(include='number').columns
df[numerics_cols] = (df[numerics_cols] - df[numerics_cols].min()) / (df[numerics_cols].max() - df[numerics_cols].min())

print("После нормализаций")
print(df.head())
print(df[numerics_cols].describe().loc[['min', 'max']])