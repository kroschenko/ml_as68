import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Загрузка данных (проверь имя файла!)
df = pd.read_csv("mushrooms.csv")
df = df.replace("?", pd.NA).dropna()

# Признаки и цель
y = df["class"]
X = pd.get_dummies(df.drop(columns="class"))

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Набор данных Mushroom")
print("Количество объектов:", X.shape[0])
print("Количество признаков:", X.shape[1])
print("Обучающая выборка:", X_train.shape[0])
print("Тестовая выборка:", X_test.shape[0])

# Модели
models = {
    "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}

# Словарь для сохранения Recall каждой модели
results = {}

# Обучение и вывод результатов
print("\n===================== РЕЗУЛЬТАТЫ МОДЕЛЕЙ =====================")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision_p = precision_score(y_test, y_pred, pos_label="p", zero_division=0)
    recall_p    = recall_score(y_test, y_pred, pos_label="p", zero_division=0)
    accuracy    = accuracy_score(y_test, y_pred)

    print(f"\nМодель: {name}")
    print(f"  Точность для 'p' (Precision): {precision_p:.4f}")
    print(f"  Полнота для 'p' (Recall):     {recall_p:.4f}")
    print(f"  Общая точность (Accuracy):    {accuracy:.4f}")
    
    results[name] = recall_p

# Итоговый вывод (как в твоем изначальном файле)
best_model = max(results, key=results.get)
worst_model = min(results, key=results.get)

print("\n========== ВЫВОД ==========")
print(f"Лучшая модель: {best_model}")
print(f"Полнота (p) = {results[best_model]:.4f}, "
      f"худшая — {worst_model} "
      f"({results[worst_model]:.4f}).")