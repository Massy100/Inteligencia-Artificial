# Conjunto de datos disponible en sklearn.datasets.fetch_openml. 
# Este conjunto de datos contiente informacion clinica de pacientes, 
# como nivel de glucosa, presion arterial, indice de masa corporal, 
# entre otros, y la etiquta indica si la persona tiene diabetes o no.
# Fuente: https://www.researchgate.net/publication/359447724_Pima_Indians_diabetes_mellitus_classification_based_on_machine_learning_ML_algorithms

# Cargar los paquetes requeridos

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_openml

# Cargar el dataset desde OpenML
diabetes = fetch_openml(name='diabetes', version=1, as_frame=True)
df = diabetes.frame
print(df)

# Inspeccion de datos
print(df.info())

# Separar caracteristicas (X) y etiqueta (y)
# Obtener X eliminando la columna de etiqueta
X = df.drop(columns=['class'])
# Obtener Y convirtiendo la etiqueta en un valor numerico

y = df['class'].apply(lambda x: 1 if x == 'tested_positive' else 0)
print(y)

# Dividir el dataset en entrenamiento (80%) y pruebas (20%)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=13,stratify=y)

# Estandarizar las caracteristicas (importante para el perceptron)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

# ENTRENAMIENTO DE PERCEPTRON
# Crear y entrenar el perceptron
perceptron = Perceptron(max_iter=4000, eta0=0.01, random_state=13)
perceptron.fit(X_train, y_train)

# Realizar 
y_pred = perceptron.predict(X_test)

eficiencia = accuracy_score(y_test, y_pred)
print(f'Precision del perceptron: {eficiencia:.2f}')
print('Reporte de clasificacion:')
print(classification_report(y_test, y_pred))

# obtener los pesos (matriz de coeficientes)
pesos = perceptron.coef_

# obtener el sesgo (termino de sesgo o bias)
sesgo = perceptron.intercept_

# mostrar los resultados
print(f'Pesos del perceptron: {pesos}')
print(f'Vector de sesgos: {sesgo}')

print(X.info())

# GRAFICA DE LA FORNTERA DE DECISION
import matplotlib.pyplot as plt

# seleccionar solo dos caracteristicas para graficar (glucosa e IMC)
X_subset = X_train[:,[1,5]]
y_subset = y_train

perceptron.fit(X_subset, y_subset)

# obtener pesos y bias
w = perceptron.coef_
b = perceptron.intercept_
print(w[0][0])

# crear una linea de decision
x1_min, x1_max = X_subset[:,0].min(), X_subset[:,0].max()
x2_min, x2_max = X_subset[:,1].min(), X_subset[:,1].max()
x1_values = np.linspace(x1_min, x1_max, 100)
x2_values = (-w[0][0] / w[0][1]) * x1_values - (b / w[0][1])

# graficar los puntos de datos
plt.figure(figsize=(8,6))
plt.scatter(X_subset[y_subset==0][:,0],
            X_subset[y_subset==0][:,1],
            color = 'blue',
            label = 'No diabetico'
            )

plt.scatter(X_subset[y_subset==1][:,0],
            X_subset[y_subset==1][:,1],
            color = 'red',
            label = 'diabetico'
            )
plt.plot(x1_values, x2_values, 'k-', color='green', label='Frontera de decision')
plt.show()

# APLICACION DE MLP
from sklearn.neural_network import MLPClassifier

# crear un perceptron multicapa con 1 cqapa oculta de 5 neuronas
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', max_iter=1500, random_state=15)

# entrenar el modelo
mlp.fit(X_train, y_train)

# verificar la cantidad de 
print(f'Numero de capas en la red: {len(mlp.coefs_)}. (incluyendo las capas de ocultas y de salida)')
print(F'Neuronas en la capa oculta: {mlp.hidden_layer_sizes[0]}')

# obtener pesos y bias
w = mlp.coefs_
b = mlp.intercepts_
print(f'Pesos de la capa oculta: {w}')
print(f'Interceptos de la capa oculta: {b}')
