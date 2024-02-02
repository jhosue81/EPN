#Trabajo Grupal
#Treicy Diaz
#Yojaira Farinango
#Jhosue Cuichán


from ipaddress import summarize_address_range
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xlrd as x
import pandas as pd
from matplotlib.ticker import FuncFormatter
# Instalamos primero en cmd pip install xlrd

df_contries = pd.read_excel(r"C:\Users\Hp\OneDrive\Documentos\CURSO PYTHON\API_NY.GDP.PCAP.KD.ZG_DS2_es_excel_v2_6508525.xls", sheet_name = "Metadata - Countries")
df_index = pd.read_excel(r"C:\Users\Hp\OneDrive\Documentos\CURSO PYTHON\API_NY.GDP.PCAP.KD.ZG_DS2_es_excel_v2_6508525.xls",sheet_name="Data",skiprows=  3)

# Obtengo los indicadores de America Latina

america_latina = df_contries[df_contries["Region"] == "América Latina y el Caribe (excluido altos ingresos)"]

# Unimos la tabla que contiene solo a los países de América Latina
base = america_latina.merge(df_index,on= ["Country Name" , "Country Code"],how = "left")

# Seleccionamos la tasa de crecimiento para todos los países de América Latina y el Caribe en el 2020 
# obtenemos la media en m_columnaseleccionada
m_columnaseleccionada = base['2020'].mean()

# ¿ Como ha evolucionado la tasa de crecimiento a lo largo del tiempo en América Latina y el Caribe?
# Esto podemos mostrarlo a través de un gráfico de series de tiempo
# Establecer 'Country Name' como índice
base = base.set_index('Country Name')


# Transponer el DataFrame para que los años estén en el índice y los países sean columnas
base_T = base.transpose()

base_T.drop(base_T.index[0:9], axis=0, inplace=True)

correlation_matrix = base_T.corr()

print(correlation_matrix)

print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")

# Mostrar el mapa de calor
plt.show()


for col in base_T.columns:
    plt.plot(base_T.index, base_T[col], marker='o', label=col)  # Trazar las observaciones para cada individuo


# Ajustar las etiquetas del eje x e y, y el título del gráfico
plt.xlabel("Años")
plt.ylabel('PIB per capita')
plt.title('Gráfico de Líneas de Observaciones de Individuos a lo Largo del Tiempo')

# Agregar una leyenda
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xticks(base_T.index[::2] , rotation="vertical")


# Mostrar el gráfico
plt.show()