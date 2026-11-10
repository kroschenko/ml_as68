import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv("breast_cancer.csv")

if 'id' in df.columns:
    df = df.drop(columns=['id'])
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])

y = df['diagnosis'].map({'M':1, 'B':0})

X = df.drop(columns=['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
svm = SVC(kernel='rbf', C=1.0, random_state=42)

models = {
    "k-NN" : knn,
    "Desicion Tree" : tree,
    "SVM" : svm
}

result = []
fig, axes = plt.subplots(1,3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaler, y_train)
    y_pred = model.predict(X_test_scaler)

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    result.append({
        "Модель" : name,
        "Precision" : prec,
        "Recall" : rec,
        "F1-score" : f1
    })

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Доброкач. (0)", "Злокач. (1)"],
                yticklabels=["Доброкач. (0)", "Злокач. (1)"])
    axes[i].set_title(f"{name}\nRecall: {rec:.3f}")
    axes[i].set_xlabel("Предсказано")
    axes[i].set_ylabel("Истинно")

plt.tight_layout()
plt.show()

df_result = pd.DataFrame(result).sort_values(by="Recall", ascending=False)
print("Сравнение моделей")
print(df_result.to_string(index=False))

best_model = df_result.iloc[0]["Модель"]
print(f"\nНаиболее надежная модель для минимизации ложнноотрицательных прогнозов: {best_model}")
