import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("auto-mpg.csv")

df.columns = df.columns.str.strip()

df = df.replace("?", float("nan"))

columns = ["mpg", "cylinders", "horsepower", "weight"]
df = df[columns].copy()

for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Пропуски до обработки:")
print(df.isnull().sum())

df = df.dropna(subset=["mpg"])

X = df[["cylinders", "horsepower", "weight"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)

model.fit(X_train, y_train)

print("\nМодель успешно обучена.")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nРезультаты модели:")
print("MSE =", round(mse, 3))
print("R2 =", round(r2, 3))

results = pd.DataFrame({
    "Реальный mpg": y_test,
    "Предсказанный mpg": y_pred
})

print("\nСравнение реальных и предсказанных значений:")
print(results.head(10).round(2))

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="horsepower",
    y="mpg",
    ci=None,
    scatter_kws={"alpha": 0.6},
    line_kws={"color": "red", "linewidth": 2}
)

plt.title("Зависимость расхода топлива от мощности двигателя")
plt.xlabel("Horsepower — мощность двигателя")
plt.ylabel("MPG — мили на галлон")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
