import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

df = pd.read_csv("winequality-white.csv", sep=";")

df.columns = df.columns.str.strip()

df = df.dropna(subset=["quality"])

X = df.drop(columns=["quality"])
y = (df["quality"] >= 7).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify= y
)

knn_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

tree_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

svm_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf"))
])

knn_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Получение предсказаний
knn_pred = knn_model.predict(X_test)
tree_pred = tree_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

# Расчёт F1-score
knn_f1 = f1_score(y_test, knn_pred)
tree_f1 = f1_score(y_test, tree_pred)
svm_f1 = f1_score(y_test, svm_pred)

# Вывод результатов
print("Результаты сравнения моделей:")
print("k-NN F1-score:", round(knn_f1, 3))
print("Decision Tree F1-score:", round(tree_f1, 3))
print("SVM F1-score:", round(svm_f1, 3))
