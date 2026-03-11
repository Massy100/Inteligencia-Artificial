# Importar pauqetes
from matplotlib import pyplot as plt
import numpy as np
import imageio
from skimage import color

# Lectura de la imagen
ruta = 'gray-wolf.jpg'
imgIn = imageio.v2.imread(ruta)

# Caracteristicas de la imagen = matriz w * h * 3
print('Dimensiones de la imagen: ', imgIn.shape)

# Imprimir el contenido de las capas de color
print('Pixel en [0,0,0]', imgIn[0,0,0])
print('Pixel en [0,0,1]', imgIn[0,0,1])
print('Pixel en [0,0,2]', imgIn[0,0,2])
print('Pixel en [0,0,:]', imgIn[0,0,:])

# Visualizacion de la imagen
plt.imshow(imgIn)
plt.show()

plt.imshow(imgIn[:,:,0])
plt.show()

# Tratamiento de la imagen
imgGray = color.rgb2gray(imgIn)
print('Dimensiones de la imagen en escala de grises: ', imgGray.shape)
print('Tipo de datos: ', imgGray.dtype)
plt.show()

plt.imshow(imgGray, cmap='gray')
plt.show()

imgSeccion = imgGray[110:440, 215:525]
plt.imshow(imgSeccion, cmap='gray')
plt.show()

imgModificada = imgGray.copy()
imgModificada[imgModificada < 0.2] = 0
plt.imshow(imgModificada, cmap='gray')
plt.show()

plt.hist(imgGray.flatten(), bins=100);
plt.show()
plt.hist(imgModificada.flatten(), bins=100);
plt.show()

imageio.imwrite('lobo-grises-modificada.jpg', (imgModificada * 255).astype(np.uint8))


