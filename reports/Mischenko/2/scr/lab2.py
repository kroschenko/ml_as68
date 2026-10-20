import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Загрузка файла
df = pd.read_csv('student-por.csv', sep=';')

X = df[['studytime', 'failures', 'G1', 'G2']]
y = df['G3']

# 2. Обучение модели
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Метрики
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R2: {r2_score(y_test, y_pred):.3f}")

results = pd.DataFrame({
    'Реальная оценка (G3)': y_test,
    'Предсказанная моделью': y_pred,
    'Ошибка (разница)': y_test - y_pred
})

print(results.head(10)) 

# 4. Визуализация 
plt.figure(figsize=(8, 5))
plt.scatter(df['G2'], df['G3'], alpha=0.5, label='Данные студентов')

simple_model = LinearRegression().fit(df[['G2']], df['G3'])
plt.plot(df['G2'], simple_model.predict(df[['G2']]), color='red', linewidth=2, label='Линия регрессии')

plt.title('Зависимость итоговой оценки (G3) от оценки за 2-й период (G2)')
plt.xlabel('Оценка за предыдущий период (G2)')
plt.ylabel('Итоговая оценка (G3)')
plt.legend()
plt.show()