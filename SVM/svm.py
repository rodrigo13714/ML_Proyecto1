import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


df = pd.read_excel("features_with_labels.xlsx")

#Separar características y etiquetas
X = df.drop(columns=["ID", "Label"])
y = df["Label"].values

#Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

class MiKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_point(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        return max(set(k_labels), key=k_labels.count)

    def predict(self, X_test):
        return np.array([self._predict_point(x) for x in X_test])

# 6. Entrenar el modelo
knn = CustomKNN(k=5)
knn.fit(X_train, y_train)

# 7. Predecir y evaluar
y_pred = knn.predict(X_test)

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("Exactitud:", accuracy_score(y_test, y_pred))
print("F1-score ponderado:", f1_score(y_test, y_pred, average='weighted'))
