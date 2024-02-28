#Clase 2 Introduccion al manejo de datos en Python con pandas

mi_variable = "Hola Mundo"
print(mi_variable)
#Lista de numeros
lista_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(lista_x)

#Diccionario

Diccionario_notas = {"Joel": 80, "Tim": 95}
Diccionario_notas["Kate"] = 100
print(Diccionario_notas)

# Creemos vectores con 5 elementos repetidos cada uno
vector_entero = [10] * 5
vector_flotante = [3.14] * 5 # Con decimales
vector_complejo = [(1 + 2j)] * 5

# Crear un diccionario que contenga estos vectores
diccionario = {
    "entero": vector_entero,
    "flotante": vector_flotante,
    "complejo": vector_complejo
}

print(diccionario)

cadena_simple = 'Hola, mundo!'
print(cadena_simple)

cadena_doble = ["Python es poderoso!", "Me gusta aprender"]

# Crear un DataFrame con los datos de rendimiento en juegos
import pandas as pd
datos = {
    'Nombre': ['Juan', 'Kathe', 'Bronson', 'Poli'],
    'Juego 1 (puntos)': [150, 180, 130, 200],
    'Juego 2 (puntos)': [120, 90, 110, 150],
    'Juego 3 (puntos)': [200, 160, 180, 190]
}

df = pd.DataFrame(datos)

# Mostrar el DataFrame
print(df)

# Importar datos
imp_sri = pd.read_excel ("data/IMP_SRI.xlsx")
print(df)