import numpy as np
import pandas as pd
import seaborn as sns

# cargar datos desde modulo
vuelos = sns.load_dataset("flights")
vuelos

# cargar datos desde archivo xlsx
ruta_archivo = "datos.xlsx"
tabla = pd.read_excel(ruta_archivo)
tabla

# cargar datos desde archivo csv
planilla = pd.read_csv('planilla.csv')
planilla.set_index('codigo', inplace=True)
planilla

planilla.dtypes
planilla['fecha'] = pd.to_datetime(planilla['fecha_contratacion'])
planilla.dtypes

import yfinance as yf
ticker_apple = 'AAPL'
start_date = '2026-01-01'
end_date = '2026-01-19'
datos_apple = yf.download(ticker_apple, start=start_date, end=end_date)
datos_apple

