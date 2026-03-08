#requerido
#bautifulSoup
#pip install bs4
from bs4 import BeautifulSoup
#libreria para hacer peticiones HTTP
import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd 
# https://es.wikipedia.org/wiki/Organizaci%C3%B3n_territorial_de_Guatemala
url = 'https://es.wikipedia.org/wiki/Organizaci%C3%B3n_territorial_de_Guatemala'

response = requests.get(url)
print(response.status_code) #200 = OKs
html = response.text
soup = BeautifulSoup(html, 'html.parser')
header = {'User-Agent':'CarlsdeLeon/1.0 (contacto:carlosdeleonijo@gmail.com)'}
response = requests.get(url, headers=header)
print(response.status_code) #200 = OKs
html = response.text

soup = BeautifulSoup(html, 'html.parser')
tabla = soup.find('table', {'class':'wikitable'})
tabla = pd.read_html(StringIO(str(tabla)))
departamento = tabla[1]
departamento.columns = ['Departamento', 'Cabecera', 'Superficie', 'Poblacion']
print(departamento)
region = soup.find_all('table')[2]
region = pd.read_html(StringIO(str(region)))[0]
print(region)
segeplan = soup.find_all('table')[3]
segeplan = pd.read_html(StringIO(str(segeplan)))[0]
print(segeplan)
subregion = soup.find_all('table')[4]
subregion = pd.read_html(StringIO(str(subregion)))[0]
print(subregion)

departamento['Superficie'] = list(int(x.replace('\xa0','')) for x in departamento['Superficie'])
print(departamento['Superficie'])