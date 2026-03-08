import matplotlib.pyplot as plt
import camelot
import pandas as pd

archivo = 'Analisis Anual 2018 ETAS.pdf'

tabla = camelot.read_pdf(archivo, pages='3')
print(f'Tabla extraida: {tabla}')
print(tabla[0].df)

area = list(map(lambda area: area.strip(), tabla[0].df[0][2].split('\n')))
area.extend(list(map(lambda area: area.strip(), tabla[0].df[0][2].split('\n'))))
print(area)

casos2017 = [int(caso.strip()) for caso in tabla[0].df[1][2].split('\n')]
casos2017.extend([int(caso.strip()) for caso in tabla[0].df[1][2].split('\n')])
print(casos2017)

tasas2017 = [float(tasa.strip()) for tasa in tabla[0].df[2][2].split('\n')]
tasas2017.extend([float(tasa.strip()) for tasa in tabla[0].df[2][2].split('\n')])
print(tasas2017)

casos2018 = [int(caso.strip()) for caso in tabla[0].df[1][2].split('\n')]
casos2018.extend([int(caso.strip()) for caso in tabla[0].df[1][2].split('\n')])
print(casos2018)

tasas2018 = [float(tasa.strip()) for tasa in tabla[0].df[2][2].split('\n')]
tasas2018.extend([float(tasa.strip()) for tasa in tabla[0].df[2][2].split('\n')])
print(tasas2018)

etas = pd.DataFrame({
    'Area de salud': area,
    'Casos 2017': casos2017,
    'Tasa 2017': tasas2017,
    'Casos 2018': casos2018,
    'Tasa 2018': tasas2018
})
print(etas)

etas.describe()


plt.figure(figsize=(10, 6))
etas[['Tasa 2017', 'Tasa 2018']].boxplot()
plt.title('Comparación de Tasas 2017 vs 2018')
plt.ylabel('Tasa')
plt.grid(True, alpha=0.3)
plt.show()

etas.plot(x='Area de salud', y=['Tasa 2017', 'Tasa 2018'], kind='bar', figsize=(12, 7))
plt.show()






