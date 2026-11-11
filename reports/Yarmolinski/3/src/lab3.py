
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import precision_score


# ==========================================
# 1. ЗАГРУЗКА ДАТАСЕТА
# ==========================================

print("=" * 55)
print("ЛАБОРАТОРНАЯ РАБОТА")
print("Вариант 10 — Adult Census Income")
print("=" * 55)

url = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/adult/adult.data"
)

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

print("\nЗагрузка датасета...")

try:
    df = pd.read_csv(
        url,
        names=columns,
        na_values="?",
        skipinitialspace=True
    )
    print("Датасет успешно загружен!")

except Exception as e:
    print("Ошибка загрузки датасета:", e)
    raise SystemExit

df.to_csv("adult_orange.csv", index=False) 

# ==========================================
# 2. ОБРАБОТКА ДАННЫХ
# ==========================================

# Удаляем лишние пробелы
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df.to_csv("adult_orange.csv", index=False)

# Удаляем строки без целевой переменной
df = df.dropna(subset=["income"])

# Целевая переменная:
# <=50K — доход не превышает 50 тысяч
# >50K — доход превышает 50 тысяч

X = df.drop(columns=["income"])
y = df["income"]

print("\nРазмер датасета:", df.shape)

print("\nРаспределение классов:")
print(y.value_counts())


# Определяем типы признаков

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ==========================================
# 3. РАЗДЕЛЕНИЕ НА ВЫБОРКИ
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nОбучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)


# ==========================================
# 4. ПРЕДОБРАБОТКА ПРИЗНАКОВ
# ==========================================

# Числовые признаки
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Категориальные признаки
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Объединяем обработчики
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==========================================
# 5. СОЗДАНИЕ МОДЕЛЕЙ
# ==========================================

knn = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)

tree = DecisionTreeClassifier(
    random_state=42
)

svm = LinearSVC(
    random_state=42,
    max_iter=5000
)

models = {
    "k-NN": knn,
    "Decision Tree": tree,
    "SVM": svm
}


# ==========================================
# 6. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ
# ==========================================

results = []

print("\n" + "=" * 55)
print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
print("=" * 55)

for name, model in models.items():

    print("\nОбучение модели:", name)

    # Для каждой модели создаём отдельный Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Обучение
    pipeline.fit(X_train, y_train)

    # Предсказание
    y_pred = pipeline.predict(X_test)

    # Precision для класса >50K
    precision = precision_score(
        y_test,
        y_pred,
        pos_label=">50K",
        zero_division=0
    )

    results.append({
        "Модель": name,
        "Precision (>50K)": precision
    })

    print("Precision:", round(precision, 4))


# ==========================================
# 7. СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ==========================================

df_result = pd.DataFrame(results)

df_result = df_result.sort_values(
    by="Precision (>50K)",
    ascending=False
)

print("\n" + "=" * 55)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 55)

print(df_result.to_string(index=False))


# ==========================================
# 8. ОПРЕДЕЛЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ==========================================

best_model = df_result.iloc[0]["Модель"]
best_precision = df_result.iloc[0]["Precision (>50K)"]

print("\n" + "=" * 55)
print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
print("=" * 55)

print("Лучшая модель:", best_model)
print("Precision:", round(best_precision, 4))

print("\nРабота программы завершена!")
