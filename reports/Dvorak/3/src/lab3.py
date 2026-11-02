from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def main():
    digits = load_digits()

    X = digits.data
    y = digits.target

    print("Набор данных Digits")
    print("Количество объектов:", X.shape[0])
    print("Количество признаков:", X.shape[1])
    print("Классы:", digits.target_names)

    # Заменили 42 на 123
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    print("\nОбучающая выборка:", X_train.shape[0])
    print("Тестовая выборка:", X_test.shape[0])

    models = {
        "k-NN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=123),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf")
        )
    }

    results = {}

    for name, model in models.items():
        print("\nМодель:", name)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy

        print("Общая точность:", round(accuracy, 4))
        print(classification_report(
            y_test, predictions, labels=list(range(10)),
            zero_division=0
        ))

    print("\nИсследование параметра k для k-NN")

    for k in [1, 3, 5, 7, 9, 11]:
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k)
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"k = {k}: Accuracy = {accuracy:.4f}")

    print("\nСравнение моделей")

    for name, accuracy in sorted(
        results.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{name}: Accuracy = {accuracy:.4f}")

    best_model = max(results, key=results.get)
    print("Лучшая модель:", best_model)
    print("Её точность:", round(results[best_model], 4))


if __name__ == "__main__":
    main()