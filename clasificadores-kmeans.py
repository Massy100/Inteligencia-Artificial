# CLASIFICADOR K-MEANS (CLUSTERING - NO SUPERVISIONADO)
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.calibration import LabelEncoder 
sns.set()
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d

ruta = 'Mall_Customers-2.csv'
df = pd.read_csv(ruta, index_col=0)
print(df)
print(df.info())

df.rename({'Gender': 'Genero', 'Age': 'Edad', 
           'Annual Income (k$)': 'Ingreso', 
           'Spending Score (1-100)': 'Gasto'}, axis=1, inplace=True)

print(df.head())

# Analisis exploratorio
print(df.describe())
print(df.describe().T)

print(df.Genero.value_counts())

# Segmentacion tradicional
print(df.Ingreso.hist());
plt.title('Distrubucion de Ingreso Anual')
plt.xlabel('Ingreso en Miles de USD')
plt.show()

df['Segmento'] = np.where(df.Ingreso >= 90, 'Ingreso Alto',
                        np.where(df.Ingreso < 50, 'Ingreso Bajo', 'Ingreso Moderado'))

print(df.Segmento.value_counts())

print(df.groupby('Segmento')['Ingreso'].describe().T)

print(df.plot.scatter(x='Ingreso', y='Gasto'))
plt.show()

scaler = StandardScaler()
col_escalar = ['Edad', 'Ingreso', 'Gasto']
datos_escalados = df.copy()
datos_escalados[col_escalar] = scaler.fit_transform(df[col_escalar])
print(datos_escalados)

print(datos_escalados.plot.scatter(x='Ingreso', y='Gasto'))
plt.show()

# Uso de K-means
from sklearn.cluster import KMeans

modelo = KMeans(n_clusters=5, random_state=16)
modelo.fit(datos_escalados[col_escalar])

datos_escalados['Segmento K'] = modelo.predict(datos_escalados[col_escalar])
print(datos_escalados)

print(datos_escalados['Segmento K'].value_counts())

marcador = ['x', '*', '.', '|', '_']
for segmento in range(5):
    temporal = datos_escalados[datos_escalados['Segmento K']==segmento]
    plt.scatter(temporal.Ingreso, temporal.Gasto, marker=marcador[segmento], #no funciona arregla los marcadores
                label = 'Segmento k'+str(segmento))

datos_escalados[col_escalar].head()
modelo.fit(datos_escalados[col_escalar])
plt.show()
    
# Utilizar mas de dos atributos
datos_escalados[col_escalar].head()
print(datos_escalados.head())
print(datos_escalados['Segmento'].value_counts())

print(datos_escalados.head())

codificador = LabelEncoder()
datos_escalados['Segmento'] = codificador.fit_transform(datos_escalados['Segmento'])
print(datos_escalados.head())

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
ax.scatter3D(datos_escalados['Edad'], datos_escalados['Ingreso'], datos_escalados['Gasto'], c=datos_escalados['Segmento'], cmap='tab10')
ax.set_title('Segmentacion con Clientes')
ax.set_xlabel('Edad')
ax.set_ylabel('Ingreso')
ax.set_zlabel('Gasto')
plt.show()

# Utilizar mas de dos atributos utilizando el segmento obtenido con K-means
datos_escalados[col_escalar].head()
print(datos_escalados.head())
print(datos_escalados['Segmento K'].value_counts())

print(datos_escalados.head())

codificador = LabelEncoder()
datos_escalados['Segmento K'] = codificador.fit_transform(datos_escalados['Segmento K'])
print(datos_escalados.head())

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
ax.scatter3D(datos_escalados['Edad'], datos_escalados['Ingreso'], datos_escalados['Gasto'], c=datos_escalados['Segmento K'], cmap='tab10')
ax.set_title('Segmentacion K con Clientes')
ax.set_xlabel('Edad')
ax.set_ylabel('Ingreso')
ax.set_zlabel('Gasto')
plt.show()






