#Se requiere la instalacion de los siguientes paquetes
#install.packages(c('tidyverse','caret','neuralnet'))
#Cargar paquetes 
library(tidyverse)
library(caret)
library(neuralnet)
#cargar conjunto de datos
datos = iris
#separacion de los catos en conjunto de entrenamiento y pruebas
muestra = createDataPartition(datos$Species,p=0.8, list = F)
train = datos[muestra,]
test = datos[-muestra,]
#analisis exploratorio
head(train,5)
tail(train,5)
train[17:25,]
sepal_length = train$Sepal.Length
hist(sepal_length)
hist(train$Petal.Length)

#Entrenamiento de red neuronal
red.neuronal = neuralnet(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = train, hidden = c(2,3))
red.neuronal$act.fct
plot(red.neuronal)

#Aplicar la red neuronal al conjunto de pruebas

prediccion = predict(red.neuronal, test, type='class')

#Decodificar maximo = Especie
specie.decod = apply(prediccion, 1, which.max)
specie.pred = data_frame(specie.decod)

specie.pred = mutate(specie.pred, especie = recode(specie.pred$specie.decod, "1" = "Setosa", "2" = "Versicolor", "3" = "Virginica"))

test$Species.pred = specie.pred$
  

  

  


