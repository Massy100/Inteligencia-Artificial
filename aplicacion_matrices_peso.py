# aplicacion de matrices de peso
# matrices de peso y funcion de activacion
import numpy as np
import math
import seaborn as sns 

w1 = np.array([
[-4.98075162144598,-1.99223954059216],
[-0.511478035578839,-0.940845321472705],
[-0.719162995475864,-2.31751409566824],
[1.46884195059585,3.95532891158279],
[1.47363302377538,3.36734954205866],
])

w2 = np.array([
[-0.133809988595947,-39.1575306577403,-0.680474828407853],
[0.409209278678977,453.786492177918,0.119695333647765],
[-1.24906630268864,-126.278983380989,1.81246226380567]
])

w3 = np.array([
[1.19576193524164,0.614544515309062,-1.24136520770107],
[0.896516115866176,-2.10819619744759,1.85449390974523],
[-0.0103272710372201,-0.917582743291455,0.901659541411071],
[-1.82543801822397,1.09624710495826,1.11852018732296]
])

def f_act(X):
    activada = np.array([1/(1 + np.exp(-x)) for x in X], dtype=np.float64)
    return activada

# cargar datos
iris = sns.load_dataset('iris')
print(iris.head())

iris.info()
xcols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
especie = iris['species'].unique()
print(especie)

X = iris[xcols].copy()
X.insert(0, 'bias', 1)
print(X.head())

## Prediccion
prediccion = []
for index, fila in X.iterrows():
    capa1 = f_act(fila.dot(w1))
    capa1 = np.insert(capa1, 0, 1)
    capa2 = f_act(capa1.dot(w2))
    capa2 = np.insert(capa2, 0, 1)
    salida = f_act(capa2.dot(w3))
    prediccion.append(especie[np.argmax(salida)])
    
iris['Prediccion'] = prediccion
print(iris.head())

print('Erroneas')
erroneas = iris[iris['species'] != iris['Prediccion']]
print(erroneas)

eficiencia = (1 - len(erroneas)/len(iris)) * 100
print(f'Eficiencia del modelo: {eficiencia}%')
    
    
    
