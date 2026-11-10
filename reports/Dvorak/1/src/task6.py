import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r'C:\питон\project\ml_as68\reports\Dvorak\1\src\winequality-white.csv', sep=';')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
numeric_columns = df.select_dtypes(include='number').columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print("\nСтандартизированные данные:")
print(df)
print("\nСтатистика стандартизированных признаков:")
print(df[numeric_columns].describe())