import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\питон\project\ml_as68\reports\Dvorak\1\src\winequality-white.csv', sep=';')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Считаем коэффициент корреляции Пирсона между fixed acidity и pH
correlation = df['fixed acidity'].corr(df['pH'])
print("\nКорреляция между fixed acidity и pH:")
print(correlation)
# Диаграмма рассеяния
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['fixed acidity'], df['pH'], color='blue', alpha=0.5)
ax.set_xlabel('Fixed acidity')
ax.set_ylabel('pH')
ax.set_title('Зависимость pH от fixed acidity', fontsize=14)
plt.tight_layout()
plt.show()