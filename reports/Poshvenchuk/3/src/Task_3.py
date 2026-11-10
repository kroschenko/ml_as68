import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score

# 1. Загрузка данных
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv('pima-indians-diabetes.csv', header=None, names=columns, comment='#')

X = df.drop(columns=['Outcome'])
y = df['Outcome']

print(f"Всего записей: {X.shape[0]}, из них признаков: {X.shape[1]}")

# 2. Разделение выборки (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)

# 3. Создание моделей
models = {
    "k-NN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree": make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(max_depth=4, random_state=123)
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", random_state=123)
    )
}

results = {}

# 4. Обучение и вывод метрик
print("\nОценка классификаторов:")
print("Модель          | Accuracy | Recall (Диабет)")
print("-" * 47)
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    
    results[name] = (acc, rec)
    print(f"{name:<15} | {acc:.4f}   | {rec:.4f}")

# 5. Исследование параметра k для k-NN
print("\nИсследование качества k-NN при разном количестве соседей:")
for k in [1, 3, 5, 7, 9, 11, 13, 15]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    mark = " (базовый из задания)" if k == 5 else ""
    print(f"k={k:<2} -> Accuracy = {acc:.4f}{mark}")

# 6. Финальное сравнение
print("\nТочности моделей:")
for name, (acc, rec) in sorted(results.items(), key=lambda item: item[1][0], reverse=True):
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=lambda k: (results[k][0], results[k][1]))

best_acc = results[best_model][0]
best_rec = results[best_model][1]

print(f"\nЛучшая базовая модель: {best_model} (Accuracy = {best_acc:.4f}, Recall = {best_rec:.4f})")