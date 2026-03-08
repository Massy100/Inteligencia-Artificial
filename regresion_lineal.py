import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt 

data = sns.load_dataset('iris')
print(data.head())

data['species'].unique()
print(data['species'].unique())

data['species'].value_counts()
print(data['species'].value_counts())

x = data['petal_length']
y = data['sepal_width']

# agregar columna constante = 1 para calcular el intercepto
x = sm.add_constant(x)
print(x.head())

resultado = sm.OLS(y, x).fit()
print(resultado.summary())

# r^2 indice de determinacion
# r = indice de correlacion
#pendiente y = mx + b y + 0.4158 x - 0.3661
#petal_width = 0.4158 * petal_length - 0.3661
#r2 = 0.927 es el indice de determinacion es la primera tabla
#r = 0.9628 es el indice de correlacion

resultado.params
print(resultado.params)

resultado.rsquared
print(resultado.rsquared)

np.sqrt(resultado.rsquared)
print(np.sqrt(resultado.rsquared))

# GRAFICO DE REGRESION LINEAL
sns.set_theme(color_codes=True)
ax = sns.regplot(data = data, x = 'petal_width', y = 'petal_length') #regration plot
ax.set_xlabel('Largo del petalo')
ax.set_ylabel('Ancho del petalo')
ax.set_title('Regresion lineal entre largo y ancho del petalo')
ax2 = sns.jointplot(data = data, x = 'petal_width', y = 'petal_length',
                    kind = 'reg', truncate = False,
                    color = 'm', height=7)
sns.pairplot(data)
ax.plot()
plt.show()

