
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Cargar dataset
df = pd.read_excel("features_with_labels.xlsx")

# Separar características y etiquetas
X = df.drop(columns=["ID", "Label"])
y = df["Label"]

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn_model.fit(X_train, y_train)

# Predicción y métricas
y_pred = knn_model.predict(X_test)

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))
print("F1-score (ponderado):", f1_score(y_test, y_pred, average='weighted'))
