import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statsmodels.discrete.discrete_model import Logit
# Clave espacio_lavado	Población objetivo sexo == "Mujer"
#Cargos los datos

df = pd.read_csv("sample_endi_model_10p.txt", sep=";")

# Eliminar filas con valores nulos en la columna "dcronica"
df = df[~df["dcronica"].isna()]
df = df[~df["espacio_lavado"].isna()]
df = df[~df["sexo"].isna()]
len(df)


# Filtrar los datos para incluir solo niños y niñas de la sierra que tengan valores válidos en la columna 'quintil'
df.groupby("sexo").size()

# Filtrar datos para la población objetivo (niños en espacio lavado)
poblacion_objetivo = df[(df["sexo"] == "Mujer")]


# Calcular cuántos niños se encuentran en la población objetivo
cantidad_niñas = len(poblacion_objetivo)

# Calcular el conteo de la variable clave (quintil) respecto a esos niños
conteo_espacio_lavado = poblacion_objetivo['espacio_lavado'].value_counts()

print("Cantidad de niños en la población objetivo (región Sierra):", cantidad_niñas)
print("Conteo de la variable 'espacio_lavado' respecto a las niñas:0 Sin espacios de lavado 1 Con espacio de lavado")
print(conteo_espacio_lavado)

######################################################################################################
#Ejercicio 2

#Definimos variables categóricas
variables_categoricas = [ 'sexo', 'espacio_lavado']

#Generamos una copia de la base original
transformador = StandardScaler()
datos_escalados = df.copy()

# Variables ficticias de las categóricas
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)

print(datos_dummies)

#La variable d cronica se limpio anteriormente

# Definimos las variables dependientes e independientes
X = datos_dummies[['sexo_Mujer', 'espacio_lavado_1.0']]
y = datos_dummies["dcronica"]
weights = datos_dummies['fexp_nino']

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

#Estimación del modelo
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

#Las mujeres tiene una probabilidad menor a tener disminución crónica /
#que los hombres, aunque el estimador no es estadísticamente significativo
#La probabilidad de tener desnutrición crónica cuando tengo acceso un espacio de lavado /
# es menor con respecto a no tener acceso a tener un espacio de lavado. El estimador es estadísticamente /
#sígnificativo, el signo de la variable es concluyente y se puede decir que es el signo adecuado

# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)


# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
predictions_class == y_test


# 100 folds:
kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    

       # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)


plt.hist(accuracy_scores, bins=30, edgecolor='black')
# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

######################################################################################3
#Ejerciio 3

#Definimos variables categóricas
variables_categoricas = ['espacio_lavado']

#Generamos una copia de la base original
transformador = StandardScaler()
datos_escalados2 =poblacion_objetivo .copy()

#Limpiamos la variable dcronica para realizar el análiss de estimación
datos_escalados2 = datos_escalados2[~datos_escalados2["dcronica"].isna()]
datos_escalados2 = datos_escalados2[~datos_escalados2["espacio_lavado"].isna()]

# Variables ficticias de las categóricas
datos_dummies = pd.get_dummies(datos_escalados2, columns=variables_categoricas, drop_first=True)

# Definimos las variables dependientes e independientes
X = datos_dummies[['espacio_lavado_1.0']]
y = datos_dummies["dcronica"]
#Asigno los pesos con el factor de expansión
weights = datos_dummies['fexp_nino']

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

#Estimación del modelo
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

#La probabilidad de tener desnutrición crónica cuando tengo acceso un espacio de lavado /
# es menor con respecto a no tener acceso a tener un espacio de lavado. El estimador es estadísticamente /
#sígnificativo, el signo de la variable es concluyente y se puede decir que es el signo adecuado
#No cambia mucho con respecto al modelo de entrem¿namiento


# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)
# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
predictions_class == y_test


kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)
print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")


# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)
plt.hist(accuracy_scores, bins=30, edgecolor='black')
# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()


#¿Qué sucede con la precisión promedio del modelo cuando se utiliza el conjunto de datos filtrado?
#0.80 Antes era  # y ahora es 0.815 Incrementa en 1,5 con respecto a sa la precision promedio anterior
# (Incremento o disminuye ¿Cuanto?)
#¿Qué sucede con la distribución de los coeficientes beta en comparación con el ejercicio anterior? 
#(Incrementa o disminuye ¿Cuanto?)
#Los coeficiente beta aumentan  de casi 20 a casi 30 , pero los coeficiente beta(en numero de picos ) disminuyen en numero con respecto al anterior