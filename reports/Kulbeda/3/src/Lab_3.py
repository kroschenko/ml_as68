from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# загружаем датасет digits
digits = load_digits()
X = digits.data
y = digits.target

print("Размер данных:", X.shape)
print("Классы:", sorted(int(val) for val in set(y)))

# делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Обучающая:", X_train.shape[0], "Тестовая:", X_test.shape[0])

# подбираем k для k-NN
print("\nПодбор k для k-NN:")
best_k = 1
best_acc = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("k =", k, "accuracy =", round(acc, 4))
    if acc > best_acc:
        best_acc = acc
        best_k = k
print("Лучшее k =", best_k, "accuracy =", round(best_acc, 4))

# обучаем три модели
models = {
    "k-NN": KNeighborsClassifier(n_neighbors=best_k),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print("\nМодель:", name)
    print("accuracy =", round(acc, 4))
    print(classification_report(y_test, pred, digits=4, zero_division=0))

# сравниваем модели и выводим победителя
print("\nСравнение моделей:")
for name, acc in results.items():
    print(name, "->", round(acc, 4))

best_model = max(results, key=results.get)
print("\nЛучшая модель:", best_model, "с точностью", round(results[best_model], 4))