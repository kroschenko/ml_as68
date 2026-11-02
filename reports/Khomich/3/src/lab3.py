print("Лабораторная работа №3")
print("Вариант 10 — Adult Census Income")

# %% 1. Загрузка данных
import numpy as np
import pandas as pd
from IPython.display import display

df = pd.read_csv('adult.csv')

display(df.head())
print("Размер датасета:", df.shape)
print("Столбцы:", df.columns.tolist())
print("Пропуски ('?') по столбцам:")
print((df == '?').sum()[lambda s: s > 0])

# %% 2. Обработка пропусков и категориальных признаков, разделение выборок
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Пропуски в датасете обозначены знаком '?'
df = df.replace('?', np.nan)

# Признаки и целевая переменная: 1 — доход > 50K, 0 — доход <= 50K
X = df.drop(columns=['income'])
y = df['income'].apply(lambda v: 1 if '>50K' in str(v) else 0)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

# Разделение: 75% — обучение, 25% — тестирование
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Предобработка: числовые — медиана + стандартизация,
# категориальные — самое частое значение + One-Hot Encoding
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols),
])

# fit только на обучающей выборке, чтобы не было утечки данных
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print("Обучающая выборка:", X_train_prep.shape)
print("Тестовая выборка:", X_test_prep.shape)
print("Доля класса >50K в train:", round(y_train.mean(), 3))

# %% 3. Обучение моделей k-NN, Decision Tree, SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf')
}

for name, model in models.items():
    model.fit(X_train_prep, y_train)
    print(f"{name}: обучение завершено")

# %% Метрики базовых моделей (precision для класса >50K — главная)
from sklearn.metrics import classification_report, accuracy_score, precision_score

for name, model in models.items():
    y_pred = model.predict(X_test_prep)

    print(f"\n{name}")
    print(classification_report(
        y_test, y_pred,
        target_names=['<=50K', '>50K'],
        zero_division=0
    ))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision (>50K):", round(precision_score(y_test, y_pred, zero_division=0), 3))

# %% 4. Исследование k-NN при разном k
from sklearn.model_selection import cross_val_score

print("Исследование k-NN")
knn_results = []
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)

    # кросс-валидация на обучающей выборке (по precision для >50K)
    cv_prec = cross_val_score(
        knn, X_train_prep, y_train, cv=5, scoring='precision'
    ).mean()

    # precision на тестовой выборке
    knn.fit(X_train_prep, y_train)
    test_prec = precision_score(y_test, knn.predict(X_test_prep), zero_division=0)

    knn_results.append({'k': k, 'CV precision': cv_prec, 'Test precision': test_prec})
    print(f"k = {k:2}: CV precision = {cv_prec:.3f}, test precision = {test_prec:.3f}")

knn_df = pd.DataFrame(knn_results)

# Лучшее k выбираем по кросс-валидации (не по тесту!)
best_k = int(knn_df.loc[knn_df['CV precision'].idxmax(), 'k'])
print("\nЛучшее k по кросс-валидации:", best_k)

# %% 5. Итоговое сравнение моделей по precision для класса ">50K"
best_knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_prep, y_train)

final_models = {
    f"k-NN (k={best_k})": best_knn,
    "Decision Tree (max_depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train_prep, y_train),
    "SVM (rbf)": models["SVM"],
}

rows = []
for name, model in final_models.items():
    y_pred = model.predict(X_test_prep)
    rows.append({
        'Модель': name,
        'Precision (>50K)': precision_score(y_test, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred),
    })
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K'], zero_division=0))

summary = pd.DataFrame(rows).sort_values('Precision (>50K)', ascending=False)
print("\nСВОДНАЯ ТАБЛИЦА")
display(summary.round(4))

best_model = summary.iloc[0]['Модель']
print(f"Лучше всего по precision для класса >50K: {best_model}")