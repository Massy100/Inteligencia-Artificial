# INTRODUCCION A OPENCV
# Importar pauqetes
from matplotlib import pyplot as plt
import numpy as np
import imageio
from skimage import color
# Importar OpenCV
import cv2
#import common
import pylab

# Modificar las dimensiones de visualizacion
pylab.rcParams['figure.figsize'] = (6.4, 4.0)
ruta = 'wall-e.jpg'
input_image = cv2.imread(ruta)
plt.imshow(input_image)
plt.show()

plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.show()

# Inversion vertical
flipped_image = cv2.flip(input_image, 0)
plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
plt.show()

# Inversion horizontal
flipped_image = cv2.flip(input_image, 1)
plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
plt.show()




