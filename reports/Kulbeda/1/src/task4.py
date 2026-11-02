import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-white.csv', sep=';')

# корреляция между двумя признаками
corr = df['fixed acidity'].corr(df['pH'])
print("Корреляция fixed acidity и pH:", round(corr, 3))

# диаграмма рассеяния
plt.figure(figsize=(8, 6))
plt.scatter(df['fixed acidity'], df['pH'])
plt.title('Зависимость pH от fixed acidity')
plt.xlabel('fixed acidity')
plt.ylabel('pH')
plt.show()