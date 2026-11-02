import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# читаем файл, разделитель ;
data = pd.read_csv('bank.csv', sep=';')

# y -> 1/0
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# категориальные признаки кодируем one-hot
X = pd.get_dummies(data.drop('y', axis=1), drop_first=True)
y = data['y']

# train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# логистическая регрессия
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)

# предсказания
pred = log.predict(X_test)

# метрики
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)

print("Accuracy =", acc)
print("Precision =", prec)
print("Recall =", rec)

# матрица ошибок
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no', 'yes'])
disp.plot(cmap='Blues')
plt.title('Confusion matrix')
plt.show()