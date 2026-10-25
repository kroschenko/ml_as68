import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("german_credit.csv")

# 1. Информация о данных
print("Размер данных:", df.shape)
print("\nПервые строки:")
print(df.head())

print("\nИнформация о данных:")
df.info()

# 2. Топ-5 целей кредита
purpose = df["purpose"].value_counts().head(5)

names = {
    "domestic appliances": "Бытовая техника",
    "car (new)": "Новый автомобиль",
    "radio/television": "Радио/телевизор",
    "car (used)": "Подержанный автомобиль",
    "business": "Бизнес"
}

purpose.index = purpose.index.map(names)

print("\nТоп-5 целей кредита:")
print(purpose)

purpose.plot(kind="bar", title="Топ-5 целей кредита")
plt.xlabel("Цель кредита")
plt.ylabel("Количество")
plt.xticks(rotation=30)
plt.tight_layout()

# 3. Преобразование Sex и Housing в числа
df["Sex"] = df["personal_status_sex"].str.extract("(male|female)")[0].map({
    "male": 0,
    "female": 1
})

df["Housing"] = df["housing"].map({
    "own": 0,
    "rent": 1,
    "for free": 2
})

print("\nЗакодированные Sex и Housing:")
print(df[["Sex", "Housing"]].head())

# 4. Ящик с усами
good = df[df["default"] == 0]["credit_amount"]
bad = df[df["default"] == 1]["credit_amount"]

plt.figure()
plt.boxplot(
    [good, bad],
    tick_labels=["Хорошие", "Плохие"]
)

plt.title("Суммы кредитов")
plt.xlabel("Тип заемщика")
plt.ylabel("Сумма кредита")

# 5. Сводная таблица
table = df.pivot_table(
    index="credit_history",
    values=["age", "duration_in_month"],
    aggfunc="mean"
)

print("\nСредний возраст и длительность кредита:")
print(table.round(2))

# 6. Нормализация
for col in ["age", "credit_amount", "duration_in_month"]:
    df[col] = (
        df[col] - df[col].min()
    ) / (
        df[col].max() - df[col].min()
    )

print("\nНормализованные данные:")
print(df[["age", "credit_amount", "duration_in_month"]].head())

# Показать графики
plt.show()