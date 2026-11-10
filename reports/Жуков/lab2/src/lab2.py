import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("medical_cost_personal_dataset.csv")
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)

X = df.drop(columns=["charges"])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 : {r2_score(y_test, y_pred):.3f}")
print(pd.Series(model.coef_, index=X.columns).sort_values())

k, b = np.polyfit(df["bmi"], df["charges"], 1)
x_line = np.linspace(df["bmi"].min(), df["bmi"].max(), 100)

plt.figure(figsize=(8, 6))
plt.scatter(df["bmi"], df["charges"], c=df["smoker"], cmap="coolwarm", alpha=0.6)
plt.plot(x_line, k * x_line + b, color="black", linewidth=2, label="Линия регрессии")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("Зависимость charges от BMI (красные: курильщики)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()