import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Общее задание, загрузка и аналитика

# Получаем путь к папке, где лежит сам файл lab1.py
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "pima-indians-diabetes.csv")

# Загрузка данных
df = pd.read_csv(csv_path, comment="#", header=None)
df.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

print("Общая информация о данных (.info()):")
df.info()

print("\nСтатистические характеристики (describe):")
print(df.describe())

print("\nПроверка явных пропусков:")
print(df.isnull().sum())

# 2. Обработка нулей в Glucose, BloodPressure, SkinThickness
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness"]
for col in cols_to_fix:
    median_val = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median_val)

print("\nСтатистика после замены нулей медианой:")
print(df[cols_to_fix].describe())

# Визуализация и обработка

# 3. Гистограммы для BMI и Age
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df["BMI"].hist(bins=20, color="skyblue", edgecolor="black")
plt.title("Распределение BMI")
plt.xlabel("BMI")
plt.ylabel("Количество")

plt.subplot(1, 2, 2)
df["Age"].hist(bins=20, color="orange", edgecolor="black")
plt.title("Распределение Age")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.tight_layout()
plt.show()

# 4. Матрица корреляции для Glucose, BMI, Age, Outcome
selected_cols = ["Glucose", "BMI", "Age", "Outcome"]
corr_matrix = df[selected_cols].corr()

print("\nМатрица корреляции:")
print(corr_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Матрица корреляции (Glucose, BMI, Age, Outcome)")
plt.show()

# 5. Круговая диаграмма для Outcome
outcome_counts = df["Outcome"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    outcome_counts, 
    labels=["Здоровы (0)", "Диабет (1)"], 
    autopct="%1.1f%%", 
    colors=["#66b3ff", "#ff9999"], 
    startangle=90
)
plt.title("Распределение признака Outcome")
plt.show()

# 6. Стандартизация всех признаков, кроме Outcome
feature_cols = [col for col in df.columns if col != "Outcome"]
for col in feature_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

print("\nПервые 5 строк после стандартизации:")
print(df.head())