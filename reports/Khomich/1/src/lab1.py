import pandas as pd
import matplotlib.pyplot as plt


# общие задания
# 1. Загрузить предложенный набор данных (по вариантам) в DataFrame библиотеки Pandas.
df = pd.read_csv("german_credit.csv")

# Удаляем повторную строку заголовков
df = df[df["default"] != "default"].copy()

# Приводим числовые столбцы к числовому типу
df["default"] = pd.to_numeric(df["default"])
df["age"] = pd.to_numeric(df["age"])
df["credit_amount"] = pd.to_numeric(df["credit_amount"])
df["duration_in_month"] = pd.to_numeric(df["duration_in_month"])

print("Общая информация о данных:")
df.info()

# 2. Провести исследовательский анализ: изучить типы данных,
# количество пропусков, основные статистические показатели (среднее,
# медиана, стандартное отклонение).
# 2.1. Типы данных
print("\nТипы данных:")
print(df.dtypes)

# 2.2. Количество пропусков
print("\nКоличество пропущенных значений:")
print(df.isnull().sum())

# 2.3. Основные статистические показатели (среднее,
# медиана, стандартное отклонение)
print("\nСреднее значение:")
print(df.mean(numeric_only=True))

print("\nМедиана:")
print(df.median(numeric_only=True))

print("\nСтандартное отклонение:")
print(df.std(numeric_only=True))


# 3. Обработать пропущенные значения (например, заполнить
# средним значением или удалить строки/столбцы).
# Проверка наличия пропусков
if df.isnull().sum().sum() == 0:
    print("\nПропущенных значений нет.")
else:
    df = df.fillna(df.mean(numeric_only=True))


# 4. Преобразовать категориальные признаки в числовые с помощью
# метода One-Hot Encoding.
# в заданиях по вариантам


# 5. Выполнить нормализацию или стандартизацию числовых признаков.
# в заданиях по вариантам


# 6. Построить несколько графиков для визуализации данных(гистограммы,
# диаграммы рассеяния) и сделать выводы о зависимостях междупризнаками.
# Гистограммы
df["age"].hist()
plt.title("Распределение возраста")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.show()

df["credit_amount"].hist()
plt.title("Распределение суммы кредита")
plt.xlabel("Сумма кредита")
plt.ylabel("Количество")
plt.show()

# Scatterplot
plt.scatter(df["age"], df["credit_amount"])
plt.title("Зависимость суммы кредита от возраста")
plt.xlabel("Возраст")
plt.ylabel("Сумма кредита")
plt.show()



# Вариант 10. German Credit Data

# 1. Загрузите данные и выведите первые 5 строк,
# а также общую информацию о столбцах (.info()).

print("\nПервые 5 строк:")
print(df.head())

print("\nОбщая информация о столбцах:")
df.info()


# 2. Проанализируйте распределение цели кредита (Purpose).
# Визуализируйте 5 самых популярных целей.

purpose_counts = df["purpose"].value_counts().head(5)

print("\n5 самых популярных целей кредита:")
print(purpose_counts)

purpose_counts.plot(kind="bar")
plt.title("5 самых популярных целей кредита")
plt.xlabel("Цель кредита")
plt.ylabel("Количество")
plt.xticks(rotation=45)
plt.show()


# 3. Преобразуйте категориальные признаки Sex и Housing в числовой формат.
df["Sex"] = df["personal_status_sex"].str.extract(r"(male|female)")

# One-Hot Encoding для Sex и Housing
df = pd.get_dummies(df, columns=["Sex", "housing"], dtype=int)

print("\nДанные после преобразования Sex и Housing:")
print(df[["Sex_female", "Sex_male", "housing_for free",
          "housing_own", "housing_rent"]].head())


# 4. Постройте "ящик с усами" для Credit amount,
# чтобы сравнить суммы кредитов у "хороших" и "плохих" заемщиков.

good = df[df["default"] == 0]["credit_amount"]
bad = df[df["default"] == 1]["credit_amount"]

plt.boxplot([good, bad], tick_labels=["Хорошие", "Плохие"])
plt.title("Credit amount для хороших и плохих заемщиков")
plt.ylabel("Сумма кредита")
plt.show()

# 5. Создайте сводную таблицу, показывающую средний возраст (Age)
# и среднюю длительность кредита (Duration) для каждой категории кредитной
# истории (Credit history).

summary = df.groupby("credit_history").agg(
    mean_age=("age", "mean"),
    mean_duration=("duration_in_month", "mean")
)

print("\nСредний возраст и срок кредита по Credit history:")
print(summary)


# 6. Нормализуйте числовые столбцы Age, Credit amount, Duration.

columns_to_normalize = [
    "age",
    "credit_amount",
    "duration_in_month"
]

for column in columns_to_normalize:
    df[column] = (
        (df[column] - df[column].min())
        / (df[column].max() - df[column].min())
    )

print("\nДанные после нормализации:")
print(df[["age", "credit_amount", "duration_in_month"]].head())