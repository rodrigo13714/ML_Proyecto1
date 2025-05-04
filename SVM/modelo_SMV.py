import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

df = pd.read_excel("features_with_labels.xlsx")

print("Conteo de clases original:")
print(df["Label"].value_counts())

X = df.drop(columns=["ID", "Label"])
y = df["Label"]

#ESCALADO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#OVERSAMPLING
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nConteo de clases después del SMOTE (solo en entrenamiento):")
print(pd.Series(y_train_smote).value_counts())

# RBF kernel
def rbf_kernel(X1, X2, gamma):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            diff = X1[i] - X2[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return K

# MODELO
class SimpleSVM:
    def __init__(self, C=1.0, gamma=0.015, max_iter=100):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter

    def fit(self, X, y):
        # Convertir etiquetas a -1 y 1
        y = np.where(y == y[0], 1, -1)
        self.X = X
        self.y = y
        n = X.shape[0]
        alpha = np.zeros(n)
        b = 0
        K = rbf_kernel(X, X, self.gamma)

        for _ in range(self.max_iter):
            for i in range(n):
                Ei = (alpha * y) @ K[:, i] + b - y[i]
                if (y[i]*Ei < -0.001 and alpha[i] < self.C) or (y[i]*Ei > 0.001 and alpha[i] > 0):
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)
                    Ej = (alpha * y) @ K[:, j] + b - y[j]

                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    alpha[i] += y[i]*y[j]*(alpha_j_old - alpha[j])

                    b1 = b - Ei - y[i]*(alpha[i]-alpha_i_old)*K[i,i] - y[j]*(alpha[j]-alpha_j_old)*K[i,j]
                    b2 = b - Ej - y[i]*(alpha[i]-alpha_i_old)*K[i,j] - y[j]*(alpha[j]-alpha_j_old)*K[j,j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2)/2

        self.alpha = alpha
        self.b = b
        self.support_vectors = X[alpha > 1e-5]
        self.support_labels = y[alpha > 1e-5]
        self.support_alpha = alpha[alpha > 1e-5]

    def predict(self, X):
        K = rbf_kernel(X, self.support_vectors, self.gamma)
        return np.sign(K @ (self.support_alpha * self.support_labels) + self.b)
    
def decode_labels(predictions, original_label):
    unique = np.unique(original_label)
    return np.where(predictions == 1, unique[0], unique[1])

#ENTRENAMIENTO
svm_custom = SimpleSVM(C=1.0, gamma=0.015, max_iter=100)
svm_custom.fit(X_train_smote, y_train_smote.values)
y_pred_raw = svm_custom.predict(X_test)
y_pred = decode_labels(y_pred_raw, y_train_smote.values)

# METRICAS
def confusion_matrix_manual(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, actual in enumerate(labels):
        for j, pred in enumerate(labels):
            cm[i, j] = np.sum((y_true == actual) & (y_pred == pred))
    return cm, labels

def accuracy_manual(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def f1_score_weighted_manual(y_true, y_pred):
    labels = np.unique(y_true)
    f1_total = 0
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        weight = np.sum(y_true == label) / len(y_true)
        f1_total += f1 * weight
    return f1_total

cm, labels = confusion_matrix_manual(y_test.values, y_pred)
print("Matriz de Confusión:")
print(cm)
print("Etiquetas:", labels)
print("Exactitud (Accuracy):", accuracy_manual(y_test.values, y_pred))
print("F1-score (ponderado):", f1_score_weighted_manual(y_test.values, y_pred))

