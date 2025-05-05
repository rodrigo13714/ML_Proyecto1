# Proyecto de Clasificación

Este repositorio contiene la implementación y experimentación de distintos algoritmos de aprendizaje supervisado para clasificar señales de audio, utilizando un conjunto de características extraídas previamente.

##  Descripción General

Se implementaron modelos personalizados de KNN y SVM.
>  **Modelo elegido:** El modelo final utilizado para generar las predicciones del archivo `predicciones_resultado.csv` fue **SVM con kernel RBF**, al haber alcanzado el mejor desempeño en métricas como accuracy y F1-score.

---

## Estructura de Carpetas y Archivos

### ** SVM/**

- `modelo_SMV.py`: Implementación del algoritmo SVM personalizado con kernel RBF. Incluye la optimización de los hiperparámetros `C` y `gamma`, y genera el archivo de predicciones para Kaggle.
- `Estadisticos.py`: Realiza un análisis exploratorio del dataset con visualizaciones estadísticas y reducción de dimensionalidad mediante PCA.
- `predicciones_resultado.csv`: Archivo de salida generado por el modelo SVM, con las predicciones del test listos para ser subidos a Kaggle.

### ** KNN/**

- `Modelo_KNN.ipynb`: Implementación del modelo KNN personalizado con funciones básicas de distancia Euclidiana, cálculo de métricas y selección de `k` óptimo.

##  Datos Utilizados

- `features_40.csv`: Conjunto de entrenamiento con 40 características por muestra.
- `Labels.csv`: Etiquetas de clasificación (0 o 1) correspondientes al conjunto de entrenamiento.
- `x_test_40.csv`: Contiene las caracteristicas extraídas del archivo test
- `x_test_402.csv`: Variante de x_test_40.csv con un cambio en los headers

---

## Requisitos

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
