import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


titanic = sns.load_dataset('titanic')


df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()


df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])


df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)


X = df.drop('survived', axis=1)
y = df['survived']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model_clf = LogisticRegression(max_iter=1000)
model_clf.fit(X_train, y_train)


y_pred_clf = model_clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred_clf)
precision = precision_score(y_test, y_pred_clf)
recall = recall_score(y_test, y_pred_clf)

print("Метрики классификации (Titanic):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")


cm = confusion_matrix(y_test, y_pred_clf)
print("\nМатрица ошибок (Confusion Matrix):")
print(cm)


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Не выжил (0)', 'Выжил (1)'],
            yticklabels=['Не выжил (0)', 'Выжил (1)'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок — Логистическая регрессия (Titanic)')
plt.tight_layout()
plt.savefig('titanic_confusion_matrix.png', dpi=150)
plt.show()


tn, fp, fn, tp = cm.ravel()
print(f"\nАнализ матрицы ошибок:")
print(f"True Negatives (TN): {tn} — правильно предсказано 'не выжил'")
print(f"False Positives (FP): {fp} — ошибочно предсказано 'выжил'")
print(f"False Negatives (FN): {fn} — ошибочно предсказано 'не выжил'")
print(f"True Positives (TP): {tp} — правильно предсказано 'выжил'")