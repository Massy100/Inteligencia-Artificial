import pandas as pd

# Enlace
# https://drive.google.com/file/d/1yGjTQBG7gGWeJ90m6ExV1OKx4i1aZz1D/view?usp=sharing
idFile = '1yGjTQBG7gGWeJ90m6ExV1OKx4i1aZz1D'
ruta = f'https://drive.google.com/uc?export=download&id={idFile}'
archivoGoogle = pd.read_csv(ruta)
print(archivoGoogle)

rutaGit = 'https://raw.githubusercontent.com/abemen/datasets/refs/heads/main/antropometricas.csv'
antropo = pd.read_csv(rutaGit)
print(antropo)

otros = pd.read_clipboard(sep=',')
print(otros)
otros.to_excel('otros.xlsx', index=False)

