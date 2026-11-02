import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Загрузка данных и удаление столбца с наибольшим количеством пропусков
df = pd.read_csv("BostonHousing.csv")

print("--- ЗАДАНИЕ 1 ---")
max_missing_col = df.isnull().sum().idxmax()
print(f"Столбец с наибольшим количеством пропусков: {max_missing_col}")
df = df.drop(columns=[max_missing_col])

# 2. Удаление строк, где отсутствует значение цены (Price)
print("\n--- ЗАДАНИЕ 2 ---")
print(f"Размер датасета ДО удаления строк без цены: {df.shape}")
df = df.dropna(subset=["Price"])
print(f"Размер датасета ПОСЛЕ удаления строк без цены: {df.shape}")

# 3. Построение гистограммы распределения цен на недвижимость
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=40, color="steelblue", kde=True, edgecolor="black")
plt.title("Распределение цен на недвижимость")
plt.xlabel("Цена (Price)")
plt.ylabel("Частота")
plt.tight_layout()
plt.show()

# 4. Расчет средней цены за дом для 5 самых популярных пригородов (Suburb)
print("\n--- ЗАДАНИЕ 4 ---")
top_5_suburbs = df["Suburb"].value_counts().head(5).index
avg_price_top_5 = (
    df[df["Suburb"].isin(top_5_suburbs)].groupby("Suburb")["Price"].mean()
)
print("Средняя цена в ТОП-5 пригородах:")
print(avg_price_top_5)

# 5. Создание нового признака PropertyAge на основе года постройки (YearBuilt)
current_year = datetime.datetime.now().year
df["PropertyAge"] = current_year - df["YearBuilt"]

# 6. Преобразование признака Type в числовой формат с помощью One-Hot Encoding
df = pd.get_dummies(df, columns=["Type"], prefix="Type", dtype=int)

print("\n--- ЗАДАНИЕ 5 и 6 (ПРОВЕРКА ИТОГОВЫХ СТОЛБЦОВ) ---")
print(df[["PropertyAge", "Type_h", "Type_t", "Type_u"]].head())
