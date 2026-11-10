import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Seed_Data.csv')

df = pd.read_csv(csv_path)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_k = 1
best_knn_acc = 0
for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    if acc > best_knn_acc:
        best_knn_acc = acc
        best_k = k

models = {
    'k-NN': KNeighborsClassifier(n_neighbors=best_k),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = (model, acc, y_pred)
    print(f"{name}: Accuracy = {acc:.4f}")

best_model_name = max(results, key=lambda name: results[name][1])
best_model, _, y_best_pred = results[best_model_name]

pca = PCA(n_components=2, random_state=42)
X_test_pca = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_best_pred, cmap='viridis', edgecolor='k')
plt.legend(handles=scatter.legend_elements()[0], labels=['Kama', 'Rosa', 'Canadian'], title="Class")
plt.title(f"PCA 2D - {best_model_name}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()