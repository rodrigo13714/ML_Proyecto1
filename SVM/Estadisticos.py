import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df1 = pd.read_csv("features_40.csv") 
df2 = pd.read_csv("Labels.csv")      
df = df1.copy()
df['label'] = df2.iloc[:, 0]

print(df.head())
print(f"Shape: {df.shape}")
print("\nDistribución de clases:")
print(df['label'].value_counts())

#DSITRIBUCIÓN
sns.countplot(x='label', data=df)
plt.title('Distribución de clases')
plt.show()

print("\nResumen estadístico:")
print(df.describe())

#HISTOGRAMA
df.drop('label', axis=1).hist(figsize=(20, 15), bins=30)
plt.suptitle("Histogramas de las características")
plt.show()

#CORRELACIÓN
plt.figure(figsize=(15, 12))
sns.heatmap(df.drop('label', axis=1).corr(), cmap='coolwarm', center=0)
plt.title("Matriz de correlación")
plt.show()

#PCA en 2D
X = df.drop('label', axis=1)
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = y

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='coolwarm')
plt.title('PCA: visualización en 2D por clase')
plt.show()
