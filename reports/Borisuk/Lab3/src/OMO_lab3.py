
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("ДАТАСЕТ IRIS")
print("Количество объектов:", len(X))
print("Количество признаков:", len(X.columns))
print("Количество классов:", len(iris.target_names))
print("Классы:", iris.target_names)

print("\nПервые 5 строк:")
print(X.head())

print("\nРаспределение классов:")
print(pd.Series(y).value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nРазмер обучающей выборки:", len(X_train))
print("Размер тестовой выборки:", len(X_test))

k_values = range(1, 16)
knn_scores = []

print("\nПОДБОР ОПТИМАЛЬНОГО K")

for k in k_values:
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    )

    knn_scores.append(scores.mean())
    print("K =", k, "| Средняя точность:", round(scores.mean(), 4))

best_k = list(k_values)[knn_scores.index(max(knn_scores))]

print("\nОптимальное значение K:", best_k)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)

tree = DecisionTreeClassifier(random_state=42)

svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", random_state=42)
)

models = {
    "k-NN": knn,
    "Decision Tree": tree,
    "SVM": svm
}

results = {}

print("\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(name, "| Accuracy:", round(accuracy, 4))

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")

for name, accuracy in results.items():
    print(name, "-", round(accuracy * 100, 2), "%")

best_model = max(results, key=results.get)

print("\nЛУЧШАЯ МОДЕЛЬ:", best_model)
print("ТОЧНОСТЬ:", round(results[best_model] * 100, 2), "%")
