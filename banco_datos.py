import seaborn as sns
print(sns.get_dataset_names()) #obtener nombres de datasets disponibles en seaborn

data = sns.load_dataset('penguins') #cargar dataset 

print(data) #mostrar las primeras filas del dataset

print(data.info()) #informacion del dataset

print(data.describe()) #estadisticas descriptivas del dataset

print(data['flipper_length_mm'].describe()) #descripcion estadistica de una columna especifica

print(data.head(7)) #mostrar las primeras 7 filas del dataset

print(data.tail(7)) #mostrar las ultimas 7 filas del dataset

print(data[100:110]) #mostrar filas desde la 100 hasta la 109

print(data['species']) 

print(data['species'].unique()) #valores unicos en la columna 'species'

print(data['species']=='Gentoo') #filtrar filas donde la especie es 'Gentoo'   

print(data[data['species']=='Gentoo']) #mostrar filas donde la especie es 'Gentoo'

print(data['species'].value_counts()) #contar ocurrencias de cada especie

adelie = (data[(data['species']=='Adelie') & (data['island']=='Biscoe')]) #filtrar filas donde la especie es 'Adelie' y la isla es 'Biscoe'

print(adelie)

print(adelie.sort_values(by='body_mass_g', ascending=True, inplace=True))

print(adelie) #ordenar el dataframe 'adelie' por 'body_mass_g' en orden descendente 

tabla_reducida = data[['species','sex','body_mass_g']] #crear un nuevo dataframe con columnas especificas
print(tabla_reducida)#mostrar la columna 'species'

#con solo cambiar el nombre de la variable ya podemos ver el valor unico en cada registro de datos
print(data['sex'].value_counts()) #contar ocurrencias de cada valor en la columna 'sex'

print(data['island'].value_counts()) #contar ocurrencias de cada valor en la columna 'island'

sexos = data.groupby(by =['species','sex'])

print(sexos.value_counts()) #contar ocurrencias de cada combinacion de 'species'

print(sexos.describe()) #estadisticas descriptivas agrupadas por 'species'

print(data['body_mass_g'].plot()) #calcular la media de la columna 'body_mass_g'
#realizar un grafico de la columna 'body_mass_g'
#para visualizar la distribucion de los datos debemos usar histograma
import matplotlib.pyplot as plt
data['body_mass_g'].hist()
#plt.show()

data['body_mass_g'].plot(kind='box') #realizar un grafico de caja para 'body_mass_g'
#plt.show()

especie = data['species'].value_counts() #contar ocurrencias de cada especie

print(especie) #realizar un grafico de barras para las especies

plt.bar(especie.index, especie) 
#plt.show()#realizar un grafico de barras horizontales para las especies

data['body_mass_g'].plot(kind='box')
plt.show() #realizar un grafico de caja para 'body_mass_g'