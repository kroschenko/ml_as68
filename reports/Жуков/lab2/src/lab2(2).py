import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("heart_disease_uci.csv")
y = (df["num"] > 0).astype(int)
X = df.drop(columns=["num", "id", "dataset"])

cat_cols = X.select_dtypes(exclude="number").columns.tolist() 
X[cat_cols] = X[cat_cols].fillna("unknown").astype(str)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

med = X_train.median(numeric_only=True)
X_train = X_train.fillna(med)
X_test = X_test.fillna(med)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, C=100.0)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred):.3f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Здоров", "Болен"])
disp.plot(cmap="Blues")
plt.title("Матрица ошибок")
plt.show()