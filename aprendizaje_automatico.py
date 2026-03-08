# Clasificador Naive Bayes
# Pasos para clasificar
# 1. Calcular la probabilidad a priori P(C)
# 2. Calcular la probabilidad condicional P(F_i|C)
# 3. Calcular la probabilidad a posteriori P(C|F_1, F_2, ..., F_n) 
# 4. Seleccionar la clase con la mayor probabilidad a posteriori

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

data = sns.load_dataset('iris')
train, test = train_test_split(data, test_size=0.4, random_state=23, stratify=data['species'])

fn = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cn = ['setosa', 'versicolor', 'virginica']

x_train = train[fn]
y_train = train['species']

# conjunto de pruebas siguiendo el modelo Y = mX + b
x_test = test[fn]
y_test = test['species']

# Clasificador Naive Bayes
mod_gnb = GaussianNB()
mod_gnb.fit(x_train, y_train)
prediccion = mod_gnb.predict(x_test)

eficiencia = metrics.accuracy_score(prediccion, y_test)
print(f"Eficiencia del modelo Naive Bayes: {eficiencia}")

test['y_pred'] = prediccion
print(test)

print(metrics.classification_report(y_test, prediccion))
cm = metrics.confusion_matrix(y_test, prediccion)
fig = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mod_gnb.classes_, yticklabels=mod_gnb.classes_)
plt.xlabel('Prediccion del modelo')
plt.ylabel('Valor real (especie)')
plt.title('Matriz de Confusion del Clasificador Naive Bayes "Iris"')
plt.show()



