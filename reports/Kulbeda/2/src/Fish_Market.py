import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# загружаем данные
df = pd.read_csv('Fish.csv')

# признаки и цель
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Weight']

# делим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# обучаем линейную регрессию
model = LinearRegression()
model.fit(X_train, y_train)

# предсказания и метрики
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("R2 =", r2)
print("RMSE =", rmse)

# график Length3 vs Weight с линией регрессии
plt.figure(figsize=(7,5))
plt.scatter(df['Length3'], df['Weight'])

# отдельная модель только по Length3, чтобы линия была ровной
m2 = LinearRegression()
m2.fit(df[['Length3']], df['Weight'])
x_line = np.linspace(df['Length3'].min(), df['Length3'].max(), 100).reshape(-1,1)
y_line = m2.predict(x_line)
plt.plot(x_line, y_line, color='red')

plt.xlabel('Length3')
plt.ylabel('Weight')
plt.title('Weight vs Length3')
plt.grid(True)
plt.show()