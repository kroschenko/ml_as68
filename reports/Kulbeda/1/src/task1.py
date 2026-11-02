import pandas as pd

df = pd.read_csv('winequality-white.csv', sep=';')

print("Первые 5 строк:")
print(df.head())

print("\nТипы столбцов:")
print(df.dtypes)