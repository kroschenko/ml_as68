import pandas as pd

df = pd.read_csv('winequality-white.csv', sep=';')

# числовые признаки без целевой переменной
cols = df.select_dtypes(include=['number']).columns.tolist()
cols.remove('quality')

# копируем датасет, чтобы исходный не портить
df_std = df.copy()

# стандартизация: z = (x - mean) / std
for col in cols:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()

# проверка: среднее ~0, std ~1
print("Проверка после стандартизации:")
print(df_std[cols].agg(['mean', 'std']).round(4))

print("\nПервые 5 строк:")
print(df_std[cols].head())