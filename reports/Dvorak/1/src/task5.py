import pandas as pd
import matplotlib.pyplot as plt
# Загрузка данных
df = pd.read_csv(r'C:\питон\project\ml_as68\reports\Dvorak\1\src\winequality-white.csv', sep=';')
# Настройки отображения
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Общий "ящик с усами" по всем числовым признакам
df.boxplot(grid=False, rot=45, figsize=(14, 6))
plt.title('Распределение числовых признаков (box plot)')
plt.suptitle("")  # Убираем автоматический заголовок
plt.xlabel('Признаки')
plt.ylabel('Значение')
plt.show()
# Программный подсчёт выбросов по правилу IQR
# Выброс = значение < Q1 - 1.5*IQR или > Q3 + 1.5*IQR
def count_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()
numeric_cols = df.drop('quality', axis=1).columns
outliers = df[numeric_cols].apply(count_outliers).sort_values(ascending=False)
print("\nКоличество выбросов по каждому признаку:")
print(outliers)
print("\nПризнак с наибольшим количеством выбросов:")
print(outliers.idxmax(), "-", outliers.max(), "выбросов")
# Отдельный "ящик с усами" для признака-лидера
top_feature = outliers.idxmax()
df.boxplot(column=top_feature, grid=False, figsize=(6, 6))
plt.title(f'Распределение {top_feature}')
plt.suptitle("")  # Убираем автоматический заголовок
plt.xlabel(top_feature)
plt.ylabel('Значение')
plt.show()