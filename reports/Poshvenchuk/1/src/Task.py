import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Загрузка данных и проверка пропусков
df = pd.read_csv('heart.csv')
print(df.head())
print("\nПропуски в данных:", df.isnull().sum().sum())

# 2. Статистический анализ
print("\nСредние значения:")
print(df.mean(numeric_only=True).round(2))
print("\nМедианы:")
print(df.median(numeric_only=True).round(2))
print("\nСтд. отклонения:")
print(df.std(numeric_only=True).round(2))

# 3. Обработка пропусков
if df.isnull().sum().sum() > 0:
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    print("\nПропуски заполнены медианой")
else:
    print("Пропусков нет — обработка не требуется")

# 4. One-Hot Encoding для sex
df['sex'] = df['sex'].map({0: 'female', 1: 'male'})
df = pd.get_dummies(df, columns=['sex'], prefix='sex', dtype=int)

# 5. Средний холестерин по группам
mean_chol = df.groupby('target')['chol'].mean().round(2)
mean_chol.index = ['Здоровые', 'Больные']
print("\nСредний холестерин:\n", mean_chol.to_string())

# 6. Нормализация
cols = ['age', 'trestbps', 'chol', 'thalach']
df[cols] = MinMaxScaler().fit_transform(df[cols])

# --- ГРАФИКИ ---
# Столбчатая
counts = df['target'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(['Здоровые', 'Больные'], counts.values, color=['#4CAF50', '#E53935'])
plt.title('Распределение пациентов')
for i, v in enumerate(counts.values):
    plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
plt.show()

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df[df['target'] == 0]['age'], df[df['target'] == 0]['thalach'],
            c='#4CAF50', label='Здоровые', alpha=0.7, edgecolors='k')
plt.scatter(df[df['target'] == 1]['age'], df[df['target'] == 1]['thalach'],
            c='#E53935', label='Больные', alpha=0.7, edgecolors='k')
plt.xlabel('Возраст'); plt.ylabel('Максимальный пульс')
plt.title('Зависимость пульса от возраста'); plt.legend(); plt.grid(alpha=0.3)
plt.show()

# Гистограмма
plt.figure(figsize=(8, 5))
plt.hist(df['chol'], bins=30, color='steelblue', edgecolor='black')
plt.xlabel('Холестерин'); plt.ylabel('Частота')
plt.title('Распределение холестерина')
plt.show()