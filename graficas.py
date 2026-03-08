import numpy as np
import matplotlib.pyplot as plt
#coordenadas
x= np.arange(1,5, 0.4)

#graficacion de x al cuadrado
plt.plot(x,x**2,'g--')

plt.show()

#graficacion de x al cubo
y1 = x
y2 = x**2
y3 = x**3

plt.plot(x,y1,'g--',x,y2,'bs',x,y3,'r^')

plt.show()
#plt.savefig('grafica1.png')