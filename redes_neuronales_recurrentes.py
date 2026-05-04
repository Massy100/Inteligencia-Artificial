import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.utils import to_categorical

texto_base = 'Las redes neuronales recurrentes son utiles para procesar secuencias'

# Crear un diccionario de caracteres a enteros
caracteres = sorted(list(set(texto_base)))
char_to_int = {c: i for i, c in enumerate(caracteres)}
int_to_char = {i: c for c, i in char_to_int.items()}
print(char_to_int)
print(int_to_char)

seq_len = 3
x = []
y =[]

# Preparar los datos de entrenamiento
for i in range(len(texto_base) - seq_len):
    seq_in = texto_base[i:i + seq_len]
    seq_out = texto_base[i + seq_len]
    x.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])
    
# Convertir a numpy arrays
x = np.array(x)
y = to_categorical(y, num_classes=len(caracteres))

print(x.shape)

x = np.reshape(x, (x.shape[0], x.shape[1], 1))

#crear modelo

model = Sequential()
model.add(SimpleRNN(32, input_shape =[x.shape[1], 1]))
model.add(Dense(len(caracteres), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# entrenar el modelo
model.fit(x, y, epochs=200, verbose=0)

# Secuencia de prueba
secuencia = 'ecu'
entrada = np.array([[char_to_int[char] for char in secuencia]])
entrada = np.reshape(entrada, (1, seq_len, 1))

prediccion = model.predict(entrada, verbose=1)
indice = np.argmax(prediccion)
print(f'Siguiente letra despues de "{secuencia}": {int_to_char[indice]}')


print(texto_base[3:6])

