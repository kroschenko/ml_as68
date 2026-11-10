import pandas as pd

# Загружаем датасет. Разделитель — точка с запятой
df = pd.read_csv(r'C:\питон\project\ml_as68\reports\Dvorak\1\src\winequality-white.csv', sep=';')

pd.set_option('display.max_rows', None)        # Показывать все строки
pd.set_option('display.max_columns', None)     # Показывать все столбцы
pd.set_option('display.width', None)           # Без ограничения по ширине
pd.set_option('display.max_colwidth', None)    # Полная ширина столбцов

print(df)

print("\nИнформация о типах столбцов:")
print(df.dtypes)
