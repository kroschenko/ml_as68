# Лабораторная работа №3. Вариант 1: Iris
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Загрузка и знакомство с данными
df = pd.read_csv("iris.csv")
print("Размер:", df.shape)
print(df.head(), "\n")
print(df.describe().round(2), "\n")
print("Классы:\n", df["variety"].value_counts(), "\n")
print("Пропуски:", df.isna().sum().sum(), "\n")

X = df.drop(columns="variety")
y = df["variety"]

# 2. Разделение на train/test (70/30, стратификация по классам)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print("train:", X_train.shape, "test:", X_test.shape, "\n")

# 3. Подбор k для k-NN (кросс-валидация на train + точность на test)
ks = range(1, 31)
cv_acc, test_acc = [], []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv_acc.append(cross_val_score(m, X_train, y_train, cv=5).mean())
    m.fit(X_train, y_train)
    test_acc.append(accuracy_score(y_test, m.predict(X_test)))

best_k = ks[int(np.argmax(cv_acc))]   # первый k с максимумом CV-точности
print(f"Оптимальное k (по 5-fold CV на train): {best_k}, CV accuracy = {max(cv_acc):.4f}\n")

plt.figure(figsize=(7, 4))
plt.plot(ks, cv_acc, "o-", label="CV (train, 5-fold)")
plt.plot(ks, test_acc, "s--", label="test")
plt.axvline(best_k, color="gray", ls=":", label=f"k = {best_k}")
plt.xlabel("k (число соседей)"); plt.ylabel("Accuracy")
plt.title("k-NN: зависимость точности от k"); plt.grid(alpha=.3); plt.legend()
plt.tight_layout(); plt.savefig("knn_k.png", dpi=150); plt.close()

# Модели
models = {
    f"k-NN (k={best_k})": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
}

# 4. Оценка на тесте
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, pred)
    print("=" * 60)
    print(name, f"-> accuracy = {results[name]:.4f}")
    print("Матрица ошибок:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=3))

# Для Decision Tree дополнительно исследуем глубину
print("Decision Tree: влияние max_depth")
for d in [1, 2, 3, 4, 5, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    cv = cross_val_score(t, X_train, y_train, cv=5).mean()
    t.fit(X_train, y_train)
    print(f"  max_depth={d}: CV={cv:.4f}, test={accuracy_score(y_test, t.predict(X_test)):.4f}")

# Для SVM — сравнение ядер
print("\nSVM: влияние ядра")
for kern in ["linear", "rbf", "poly"]:
    s = make_pipeline(StandardScaler(), SVC(kernel=kern))
    cv = cross_val_score(s, X_train, y_train, cv=5).mean()
    s.fit(X_train, y_train)
    print(f"  {kern}: CV={cv:.4f}, test={accuracy_score(y_test, s.predict(X_test)):.4f}")

# Устойчивость вывода: 5-fold CV на всей выборке (тест из 45 объектов слишком мал)
print("\n5-fold CV на всей выборке (150 объектов):")
for name, model in models.items():
    sc = cross_val_score(model, X, y, cv=5)
    print(f"  {name}: {sc.mean():.4f} ± {sc.std():.4f}")

# Итоговый график
plt.figure(figsize=(6, 4))
bars = plt.bar(results.keys(), results.values(), color=["#4c72b0", "#55a868", "#c44e52"])
plt.ylim(0.8, 1.02); plt.ylabel("Accuracy (test)"); plt.title("Сравнение моделей")
for b, v in zip(bars, results.values()):
    plt.text(b.get_x() + b.get_width()/2, v + .005, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig("compare.png", dpi=150); plt.close()
