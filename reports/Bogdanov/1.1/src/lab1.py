import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print()
df.info()
print()
print("Пропуски:")
print(df.isna().sum())
print()
print(df[["Age", "Fare", "SibSp", "Parch"]].agg(["mean", "median", "std"]).round(3))

# 2. Выжившие и погибшие
counts = df["Survived"].value_counts().sort_index()
print()
print("Погибло:", counts[0], "Выжило:", counts[1])
plt.figure(figsize=(5, 4))
bars = plt.bar(["Погибли (0)", "Выжили (1)"], counts.values, color=["#c0392b", "#27ae60"])
plt.bar_label(bars)
plt.title("Количество выживших и погибших пассажиров")
plt.ylabel("Число пассажиров")
plt.tight_layout()
plt.savefig("fig_survived.png", dpi=150)
plt.close()

# 3. Пропуски в Age -> медиана
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)
print()
print("Медиана Age:", age_median, "| пропусков в Age после:", df["Age"].isna().sum())
# (доп.) 2 пропуска в Embarked заполняем модой
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 4. One-Hot Encoding
df_enc = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
print()
print(df_enc[[c for c in df_enc.columns if c.startswith(("Sex_", "Embarked_"))]].head())

# 5. Гистограмма возрастов
plt.figure(figsize=(6, 4))
plt.hist(df["Age"], bins=20, color="#2980b9", edgecolor="white")
plt.axvline(age_median, color="red", linestyle="--", label=f"медиана = {age_median:.0f}")
plt.title("Распределение возрастов пассажиров")
plt.xlabel("Возраст, лет")
plt.ylabel("Число пассажиров")
plt.legend()
plt.tight_layout()
plt.savefig("fig_age.png", dpi=150)
plt.close()

# 6. FamilySize
df_enc["FamilySize"] = df_enc["SibSp"] + df_enc["Parch"]
print()
print(df_enc[["SibSp", "Parch", "FamilySize"]].head())
print("Распределение FamilySize:")
print(df_enc["FamilySize"].value_counts().sort_index())
print()
print("Доля выживших по FamilySize:")
print(df_enc.groupby("FamilySize")["Survived"].mean().round(3))

# Общее задание: стандартизация + диаграмма рассеяния
for col in ["Age", "Fare", "FamilySize"]:
    df_enc[col + "_std"] = (df_enc[col] - df_enc[col].mean()) / df_enc[col].std()
print()
print(df_enc[["Age_std", "Fare_std", "FamilySize_std"]].agg(["mean", "std"]).round(3))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for s, c, l in [(0, "#c0392b", "Погибли"), (1, "#27ae60", "Выжили")]:
    d = df_enc[df_enc["Survived"] == s]
    ax[0].scatter(d["Age"], d["Fare"], s=12, alpha=.6, color=c, label=l)
ax[0].set_xlabel("Возраст"); ax[0].set_ylabel("Стоимость билета (Fare)")
ax[0].set_title("Возраст и стоимость билета"); ax[0].legend()
fs = df_enc.groupby("FamilySize")["Survived"].mean()
ax[1].bar(fs.index, fs.values, color="#8e44ad")
ax[1].set_xlabel("FamilySize"); ax[1].set_ylabel("Доля выживших")
ax[1].set_title("Выживаемость по размеру семьи")
plt.tight_layout()
plt.savefig("fig_scatter.png", dpi=150)
plt.close()

print()
print("Выживаемость по полу:")
print(df.groupby("Sex")["Survived"].mean().round(3))
print("Выживаемость по классу:")
print(df.groupby("Pclass")["Survived"].mean().round(3))
