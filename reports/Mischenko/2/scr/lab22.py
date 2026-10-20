import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

print("Загрузка датасета Mushroom...")
df = pd.read_csv(r'D:\5 семестр\ОМО\2 лаба\mushrooms.csv')
print(f"Данные загружены. Исходный размер: {df.shape}")

df = df[df['stalk-root'] != '?']
print(f"Размер после удаления пропусков: {df.shape}")

# 2. Подготовка данных и One-Hot Encoding
y = (df['class'] == 'p').astype(int)
X = df.drop('class', axis=1)
X_encoded = pd.get_dummies(X)
print(f"Количество признаков после One-Hot Encoding: {X_encoded.shape[1]}")
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 3. Обучение логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Расчет метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n--- Результаты модели ---")
print(f"Общая точность: {accuracy:.4f}")
print(f"Точность предсказания ядовитости: {precision:.4f}")
print(f"Полнота - сколько ядовитых грибов нашли: {recall:.4f}")

# 5. Построение матрицы ошибок
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Съедобный (e)', 'Ядовитый (p)'])
disp.plot(cmap=plt.cm.Blues) 

plt.title('Матрица ошибок (Confusion Matrix)')
plt.grid(False) 
plt.show()