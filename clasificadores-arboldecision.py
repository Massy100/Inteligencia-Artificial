# Importar bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importar el separador de muestras para entrenamiento y pruebas
from sklearn.model_selection import train_test_split
# Importar el clasificador: Arbol de decisión
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Importar las metricas para medir la eficiencia del modelo
from sklearn import metrics

# Cargar el dataset
data = sns.load_dataset('iris')

# Seleccionar datos de prueba y entrenamiento
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=26)
print(train)

x_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']

fn = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cn = ['setosa', 'versicolor', 'virginica']

# Conjunto de pruebas
x_test = train[fn]
y_test = train['species']

# Arbol de decisión
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=112)
# Ajustar el modelo a los datos de entrenamiento
mod_dt.fit(x_train, y_train)
# Probar el modelo
prediccion = mod_dt.predict(x_test)
print(prediccion)

# Importancia de los predictores
print(mod_dt.feature_importances_)

# Visualizar las reglas de clasificación
plt.figure(figsize=(10, 8))
plot_tree(mod_dt, feature_names=fn, class_names=cn, filled=True);
plt.show()

# Eficiencia del modelo
eficiencia = metrics.accuracy_score(y_test, prediccion)
print("Eficiencia del modelo: ", eficiencia)

margen_error = pd.DataFrame(x_test)
margen_error['species'] = y_test
margen_error['prediccion'] = prediccion
print(margen_error)

# Ejercicio: clasificar pinguinos (analisis descriptivo colocar nota al final y en otro el arbol de decision)







 