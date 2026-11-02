import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Задание 1
df = pd.read_csv("adult.csv")
print(df.head(10))

#Задание 2
df = df.replace("?", np.nan)

workclass_mode = df["workclass"].mode()[0]
occupation_mode = df["occupation"].mode()[0]
country_mode = df["native.country"].mode()[0]

print("Мода для столбца workclass:", workclass_mode)
print("Мода для столбца occupation:", occupation_mode)
print("Мода для столбца native.country:", country_mode)

df = df.fillna({
    "workclass": workclass_mode,
    "occupation": occupation_mode,
    "native.country": country_mode
})

print("DataFrame после заполнения пропущенных значений:")
print(df.head(10))

#Задание 3
grouped_by_sex = df.groupby("sex")
groups_size = grouped_by_sex.size()
print(groups_size)

plt.figure(figsize=(8, 5))
plt.bar(groups_size.index, groups_size.values, color=['skyblue', 'pink'])

plt.xlabel("Пол")
plt.ylabel("Количество людей")
plt.title("Количество мужчин и женщин в наборе данных")
plt.grid(axis='y', alpha=0.5)

#Задание 4
df["race_Black"] = (df["race"] == "Black").astype(int)
df["race_White"] = (df["race"] == "White").astype(int)
df["race_Asian"] = (df["race"] == "Asian-Pac-Islander").astype(int)
df["race_Amer"] = (df["race"] == "Amer-Indian-Eskimo").astype(int)

print(df[["race_Black", "race_White", "race_Asian", "race_Amer"]].head(10))

#Заданием 5
plt.figure(figsize=(14, 6))

plt.hist([df[df['income'] == '<=50K']['age'],
          df[df['income'] == '>50K']['age']],
         bins=16,
         label=['<=50K', '>50K'],
         color=['skyblue', 'pink'])

plt.xlabel("Возраст (Age)")
plt.ylabel("Количество людей")
plt.title("Распределение возраста по группам дохода")
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

#Задание 6
df = df.drop(columns=["race_Black", "race_White", "race_Asian", "race_Amer"])
df['is_usa'] = (df['native.country'].str.strip() == 'United-States').astype(int)
print(df.head(10))