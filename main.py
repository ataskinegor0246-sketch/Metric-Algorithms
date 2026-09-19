import numpy as np
import pandas as pd
from models import KNN_classifier

path = "data/proced_titanic.csv"

data = pd.read_csv(path)

Y = data["Survived"].values
X = data.drop("Survived", axis=1).values.astype(float)

train_part = int(Y.shape[0] * 0.8)

#разделили выборку на треничровочную и тестовую
X_train, X_test = X[:train_part], X[train_part:]
Y_train, Y_test = Y[:train_part], Y[train_part:]

#масштабируем признаки
X_std = X_train.std()
X_mean = X_train.mean()

X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

#классификатор KNN с 10 соседями
model = KNN_classifier(10)

#запомнили выборку для KNN
print("Запомнили выборку для дальнейшей классификации\n")
model.fit(X_train_scaled, Y_train)

#Делаем предсказания
print("Делаем предсказания...")
predictions = model.predict(X_test_scaled)
print("Модель сделала предсказания!!!")

#метрики
confusion_matrix = model.confusion_matrix(predictions, Y_test)
accuracy = model.accuracy()
precision = model.precision()
recall = model.recall()
f1_core = 2*(precision*recall)/(precision+recall)

print("\nМетрки модели:")
print(f"Accuracy = {accuracy}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F1_core = {f1_core}")

print(f"\nConfusion Matrix:\n {confusion_matrix}")



