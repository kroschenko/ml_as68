import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-white.csv', sep=';')

# числовые столбцы (исключаем quality, это целевая переменная)
cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols.remove('quality')

# считаем выбросы по каждому признаку через IQR
outliers = {}
for col in cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n = ((df[col] < low) | (df[col] > high)).sum()
    outliers[col] = n

outliers = pd.Series(outliers).sort_values(ascending=False)
print("Количество выбросов по признакам:")
print(outliers)

# признак с максимумом
print("\nБольше всего выбросов в:", outliers.idxmax(), "—", outliers.max())

# box plot для всех признаков
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cols):
    axes[i].boxplot(df[col])
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xticks([])

# убираем пустой подграфик
for j in range(len(cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()