import pandas as pd
import os
import csv
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

archivo_csv = os.getcwd() + "/../Escritorio/-data-mining/clima.csv"

datos=pd.read_csv(archivo_csv)

datos.head()

# muestra las columnas que traer 
datos.columns

# verificar si los valores estan en blando y toma todas las filas y todas las columnas
datos.isnull().any().any()

# nos muestra cuantos valores vienen en blanco en cada columna
datos.isnull().sum()

# mostrar las filas con valores faltantes
datos[datos.isnull().any(axis=1)]
# Nan lo coloca como los valores en blanco

del datos['number']

datos

# Limpiando los valores en blanco
filas_antes=datos.shape[0]
datos=datos.dropna()

filas_despues=datos.shape[0]
print("El número de filas eliminadas son: {}".format(filas_antes-filas_despues))

filas_antes

filas_despues

datos_limpio=datos.copy()

# nos devuelve si la condicion es verdadera o falsa
datos_limpio['humedad_alta']=(datos_limpio['relative_humidity_3pm']>24.99)
datos_limpio

# nos devuelve un dato numerico
datos_limpio['humedad_alta']=(datos_limpio['relative_humidity_3pm']>24.99)*1
print(datos_limpio['humedad_alta'])

datos_limpio

y=datos_limpio[['humedad_alta']].copy()
y

datos_limpio['relative_humidity_3pm'].head()

# head trae los pirmeros cinco resultados
y.head()

tiempo='9am'

caracteristica=list(datos_limpio.columns[datos_limpio.columns.str.contains(tiempo)])

caracteristica.remove('relative_humidity_9am')
caracteristica

x=datos_limpio[caracteristica].copy()

# 1/3 de la data va utlizarse para entreno y 2/3 para pruebas
x_entreno, x_prueba, y_entreno, y_prueba = train_test_split(x,y,test_size=0.33, random_state=32)

type(x_entreno)
type(y_prueba)

# resumen de los datos estadisticos mas basicos
y_prueba.describe()
x_entreno.describe()

# Arbol de decision

# primer atributo las hojas que se van a tener
clasficador_humedad=DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

# entrenando
clasficador_humedad.fit(x_entreno,y_entreno)

type(clasficador_humedad)

prediccion = clasficador_humedad.predict(x_prueba)
type(prediccion)

#tomando los primeros 10 registros
prediccion[:10]

y_prueba[['humedad_alta']][:10]

accuracy_score(y_prueba, y_pred = prediccion)


mean_squared_error(y_prueba, y_pred = prediccion)
