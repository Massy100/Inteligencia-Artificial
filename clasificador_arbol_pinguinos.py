# Parte 1: Analisis descriptivo del dataset de pingüinos

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')

# Visualizar las primeras filas 
print("PRIMERAS FILAS DEL DATASET:")
print(penguins.head())

# Información general del dataset
print("INFORMACION DEL DATASET:")
print(penguins.info())

# Estadisticas descriptivas de las variables numEricas
print("ESTADISTICAS DESCRIPTIVAS:")
print(penguins.describe())

# Verificar valores nulos
print("VALORES NULOS POR COLUMNA:")
print(penguins.isnull().sum())

# Eliminar filas con valores nulos para el analisis
penguins_clean = penguins.dropna()

# Distribucion de especies
print("DISTRIBUCION DE ESPECIES:")
print(penguins_clean['species'].value_counts())

# Visualizaciones descriptivas

# 1. Distribución de especies
plt.figure(figsize=(8, 6))
sns.countplot(data=penguins_clean, x='species')
plt.title('Distribución de Especies de Pingüinos')
plt.show()

# 2. Relación entre largo y profundidad del pico por especie
plt.figure(figsize=(10, 6))
sns.scatterplot(data=penguins_clean, x='bill_length_mm', y='bill_depth_mm', 
                hue='species', style='sex', s=100)
plt.title('Relación entre Largo y Profundidad del Pico por Especie')
plt.xlabel('Largo del pico (mm)')
plt.ylabel('Profundidad del pico (mm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3. Distribucion de masa corporal por especie
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=penguins_clean, x='species', y='body_mass_g')
plt.title('Masa Corporal por Especie')
plt.xlabel('Especie')
plt.ylabel('Masa corporal (g)')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.violinplot(data=penguins_clean, x='species', y='body_mass_g')
plt.title('Distribución de Masa Corporal por Especie')
plt.xlabel('Especie')
plt.ylabel('Masa corporal (g)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4. Matriz de correlacion
plt.figure(figsize=(10, 8))
# Seleccionar solo columnas numericas
numeric_cols = penguins_clean.select_dtypes(include=[np.number]).columns
correlation = penguins_clean[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matriz de Correlación - Variables Numéricas')
plt.show()

# 5. Pairplot para visualizar relaciones entre variables
sns.pairplot(penguins_clean, hue='species', corner=True)
plt.suptitle('Relaciones entre Variables por Especie', y=1.02)
plt.show()

print("ANALISIS DESCRIPTIVO COMPLETADO")



# Parte 2: Clasificacion con Arbol de Decision

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Codificar variables categoricas (sexo e isla)
le_sex = LabelEncoder()
le_island = LabelEncoder()

penguins_model = penguins_clean.copy()
penguins_model['sex_encoded'] = le_sex.fit_transform(penguins_model['sex'])
penguins_model['island_encoded'] = le_island.fit_transform(penguins_model['island'])

# Mostrar la codificacion
print("CODIFICACION DE VARIABLES CATEGORICAS:")
print(f"Sexo: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"Isla: {dict(zip(le_island.classes_, le_island.transform(le_island.classes_)))}")
print()

# Seleccionar caracteristicas (predictores) y variable objetivo
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
            'body_mass_g', 'sex_encoded', 'island_encoded']

X = penguins_model[features]
y = penguins_model['species']

# Dividir en conjunto de entrenamiento y prueba (60% entrenamiento, 40% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=15
)

print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} muestras")
print(f"Tamaño del conjunto de prueba: {len(X_test)} muestras")
print()

# Nombres de las caracteristicas para visualizacion
feature_names = features
# Nombres de las clases
class_names = ['Adelie', 'Chinstrap', 'Gentoo']

# Crear y entrenar el arbol de decision
mod_dt = DecisionTreeClassifier(max_depth=4, random_state=152)
mod_dt.fit(X_train, y_train)

# Realizar predicciones
y_pred = mod_dt.predict(X_test)

# Importancia de las caracteristicas
print("IMPORTANCIA DE LAS CARACTERISTICAS:")
for feature, importance in zip(feature_names, mod_dt.feature_importances_):
    print(f"{feature}: {importance:.4f}")
print()

# Visualizar la importancia de las caracteristicas
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({
    'Caracteristica': feature_names,
    'Importancia': mod_dt.feature_importances_
}).sort_values('Importancia', ascending=False)

sns.barplot(data=importance_df, x='Importancia', y='Caracteristica')
plt.title('Importancia de las Caracteristicas en la Clasificación')
plt.xlabel('Importancia')
plt.tight_layout()
plt.show()

# Visualizar el arbol de decision
plt.figure(figsize=(20, 12))
plot_tree(mod_dt, feature_names=feature_names, class_names=class_names, 
          filled=True, rounded=True, fontsize=10)
plt.title('Arbol de Decision para Clasificacion de Pingüinos')
plt.show()

# Evaluar la eficiencia del modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

# print("MÉTRICAS DE EVALUACION DEL MODELO:")
# print(f"Exactitud (Accuracy): {accuracy:.4f}")
# print(f"Precision (Precision): {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")
# print()

# Matriz de confusion
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusion')
plt.xlabel('Prediccion')
plt.ylabel('Valor Real')
plt.show()

# Reporte de clasificación detallado
print("REPORTE DE CLASIFICACION:")
print(metrics.classification_report(y_test, y_pred, target_names=class_names))
print()

# Comparar predicciones con valores reales
resultados = X_test.copy()
resultados['species_real'] = y_test
resultados['species_pred'] = y_pred
resultados['correcto'] = resultados['species_real'] == resultados['species_pred']

print("MUESTRA DE RESULTADOS (10 PRIMERAS FILAS):")
print(resultados[['species_real', 'species_pred', 'correcto']].head(10))
print()
print(f"Predicciones correctas: {resultados['correcto'].sum()} de {len(resultados)}")
print(f"Porcentaje de acierto: {(resultados['correcto'].sum()/len(resultados))*100:.2f}%")