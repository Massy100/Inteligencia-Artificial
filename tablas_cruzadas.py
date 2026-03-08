import pandas as pd

ruta_archivo = "datos.xlsx"
tabla = pd.read_excel(ruta_archivo, engine='openpyxl')

# 1. Tabla de contingencia entre Edad y Sexo
contingencia = pd.crosstab(tabla['Edad'], tabla['Sexo'])
print("Tabla de contingencia Edad vs Sexo:")
print(contingencia)
print()

# 2. Calcular intervalos de edad
n = len(tabla)
intervalos = pd.cut(tabla['Edad'], bins=5)
print("Intervalos de edad:")
print(intervalos.value_counts().sort_index())
print()

# 3. Calcular frecuencia por intervalos
frecuencias = pd.crosstab(index=intervalos, columns='frecuencia')
print("Frecuencia por intervalos:")
print(frecuencias)
print()

# 4. Calcular estadísticas adicionales
frecuencias['fa'] = frecuencias['frecuencia'].cumsum()  # frecuencia acumulada
frecuencias['fr'] = frecuencias['frecuencia'] / n  # frecuencia relativa
frecuencias['fra'] = frecuencias['fr'].cumsum()  # frecuencia relativa acumulada
frecuencias['%'] = frecuencias['fr'] * 100  # frecuencia porcentual
frecuencias['%a'] = frecuencias['fra'] * 100  # frecuencia porcentual acumulada

print("Tabla de frecuencias completa:")
print(frecuencias)
print()

# 5. Calcular marca de clase (Xm) y media ponderada
# Para acceder a los puntos medios de los intervalos
frecuencias['Xm'] = [interval.mid for interval in frecuencias.index]

# Calcular f * Xm
frecuencias['f*Xm'] = frecuencias['frecuencia'] * frecuencias['Xm']

# Calcular la media (suma de f*Xm / n)
media = frecuencias['f*Xm'].sum() / n
print(f"Media (promedio ponderado): {media}")
print()

# Mostrar tabla final con todas las columnas
print("Tabla final completa:")
print(frecuencias)