# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:52:07 2025

@author: manel
"""

import numpy as np
import matplotlib.pyplot as plt

print("\n\n----INICIO ud2.py----\n\n")
#%% Sección inicial
a = np.array([1,2,3,4])
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(a.shape)
print(b)
print(b.shape)

c = np.arange(1.0, 10.5, 0.5)
print(c)

zeros_array = np.zeros((2,3)) # a) Array de ceros
print("\nArray de ceros (2x3):\n", zeros_array)
ones_array = np.ones((3,2)) # b) Array de unos
print("\nArray de unos (3x2):\n", ones_array)
indentidad = np.identity(4) # c) Array identidad
print("\nMatriz identidad (4x4):\n", indentidad)
random_array = np.random.random((2,3)) # Valores aleatorios entre 0 y 1
print("\nArray random (aleatorio):\n", random_array)
random_sample = np.random.ranf((2,2)) # Random_sample() / ranf()
print("\nArray aleatorio ranf():\n", random_sample)
enteros_random = np.random.randint(10,50, (3,3)) # Enteros aleatorios entre 10 y 50
print("\nArray de enteros aleatorio randint():\n", enteros_random)

#%% Sección dos, operaciones matemáticas
print("Enteros Radom :\n")
x = np.array([[1,2], [3,4]])
# ctrl + alt + down duplicar linear
y = np.array([[6,7], [8,9]])
print(x)
print(y)

print("\nSuma x + y: \n", x + y)
print("\nResta x - y: \n", x - y)
print("\nMultiplicación x * y: \n", x * y)
print("\nProducto escalar np.dot(x,y):\n", np.dot(x,y))
print("\nDivisión x / y: \n", x / y)

#%% Matplotlib

x = [1,2,3,4]
y = [10,20,25,30]
plt.plot(x, y, marker='o')
plt.title("ejemplo de gráfico")
plt.xlabel("eje x")
plt.ylabel("eje y")

plt.show()

fig, axs = plt.subplots(2,2, figsize=(8,4))
axs[0,0].plot(x, y, marker='*')
axs[0,0].set_title('Linea')
axs[0,1].bar(x, y, color="y")
axs[0,1].set_title('Barras')
axs[1,0].scatter(x, y, color='red')
axs[1,0].set_title('Dispersión')
axs[1,1].hist(y, bins=4, color='purple')
axs[1,1].set_title('Histograma')

#%% Sección 2, 
x = [1,2,3,4]
y = [10,20,25,30]
plt.plot(x, y, marker='o')
plt.title("ejemplo de gráfico")
plt.xlabel("eje x")
plt.ylabel("eje y")
plt.xlim(0, 5)
plt.ylim(0,505)
plt.show()

fig, axs = plt.subplots(2,2, figsize=(8,4))
axs[0,0].plot(x, y, marker='*')
axs[0,0].set_title('Linea')

axs[0,0].set_xlim(0,5)
axs[0,0].set_ylim(0,35)

axs[0,1].bar(x, y, color="y")
axs[0,1].set_title('Barras')
axs[1,0].scatter(x, y, color='red')
axs[1,0].set_title('Dispersión')
axs[1,1].hist(y, bins=4, color='purple')
axs[1,1].set_title('Histograma')

#%% pandas
import pandas as pd
ruta = r"C:\Users\Mañana\Documents\akoarguel\Sistemas-de-Aprendizaje-Automatico\Prácticas\UNIDAD 2\Ejemplos clase - spyder\unsdg_2002_2021.csv"
df = pd.read_csv(ruta, sep=',')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns.values)
# print(df.index)
print("Ordenando valores de las fechas")
print(df.sort_values("dt_year", ascending=True))
print("usando iloc[] para la primera fila")
print(df.iloc[1])
print("usando loc[] para dt_year")
# df_2002= df.loc[df["dt_year"]==2002]
# print(df_2002)
print(df.drop_duplicates())
print(df.drop_duplicates(subset='dt_year'))
print("Corrigiendo NAN")
print(df.dropna(how='any'))#elimina si hay algun NAN
print(df.dropna(how='all'))#elimina todos los NAN
print(df.fillna(0))#remplaza los nan por 0.0
 
 
#%% SimpleImputer
 
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#cargar el dataset
 
ruta = r"C:\Users\Mañana\Documents\akoarguel\Sistemas-de-Aprendizaje-Automatico\Prácticas\UNIDAD 2\Ejemplos clase - spyder\unsdg_2002_2021.csv"
df = pd.read_csv(ruta, sep=',')

#Seleccionar categoria y crear dataset
dfObjetivo = df[["country","level_of_development"]]
dfObjetivo = dfObjetivo.drop_duplicates()
 
#prueba con df normal
print(df.head())
#prueba con df objetivo
print(dfObjetivo.head())
 
dfCaracteristicas = df.drop(columns=["dt_year","dt_date","region"])
#agrupar por pais y calcular la media
dfCaracteristicas = dfCaracteristicas.groupby("country").mean(numeric_only=True)
print (dfCaracteristicas)
 
imputador = SimpleImputer(missing_values=np.nan, strategy="mean")
dfImputado = imputador.fit_transform(dfCaracteristicas)
#Convertirlo a dataframe
 
dfImputado = pd.DataFrame(dfImputado,index=dfCaracteristicas.index,
                          columns=dfCaracteristicas.columns)
 
#Prueba con dfImputado
print(dfImputado.head())
 
#estandalizar los datos
escalador = StandardScaler()
dfEstandarizado = escalador.fit_transform(dfImputado)
 
#convertirlo a dataframe
X = pd.DataFrame(dfEstandarizado,index=dfCaracteristicas.index,
                               columns=dfCaracteristicas.columns)
 
 
#Preparamos la variable objetivo Y
Y = dfObjetivo.set_index("country")
 
#Caracteristicas estandarizadas (x)
print(X.head())
 
#Nivel de desarrollo (y)
print(Y.head())
 
#dfObjetivo sera como Y pero  con otro index
print(dfObjetivo.head())
 
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size= 0.30,
    random_state=42)
 
print("\n------Tamaños de los conjuntos")
print("\n------X_train", X_train.shape)
print("\n------X_test", X_test.shape)
print("\n------Y_train", Y_train.shape)
print("\n------Y_test", Y_test.shape)
# print("\n------Y_train", Y_train)
# print("\n------X_train", X_train)

#%% Practica VIII - WALMART

import pandas as pd
import random

dfDatos = pd.DataFrame()

def cargarDatos():
    dfDatos = pd.read_csv( r"C:\Users\Mañana\Documents\akoarguel\Sistemas-de-Aprendizaje-Automatico\Prácticas\UNIDAD 2\Ejemplos clase - spyder\Walmart_Sales.csv")
    
    valoresPosiblesWeeklyRains = ["Ninguna", "Pocas", "Medias", "Muchas"]
    weekly_Rains = []
    for i in range(dfDatos.shape[0]):
        weekly_Rains.append(valoresPosiblesWeeklyRains[random.randint(0, 3)])
        
    dfDatos["Weekly_Rains"] = weekly_Rains
    
    valoresPosiblesWeeklyDiscounts = ["Carnes", "Pescados", "Restos"]
    weekly_Discounts = []
    for i in range(dfDatos.shape[0]):
        weekly_Discounts.append(valoresPosiblesWeeklyDiscounts[random.randint(0, 2)])
        
    dfDatos["Weekly_Discounts"] = weekly_Discounts
    
    dfDatos = dfDatos[dfDatos.Store == 1]
    return dfDatos

dfDatos = cargarDatos()
print("\n\nTipo de datos: ")
print(type(dfDatos))
print("\n\nDataframe dfDatos-------------------\n")
print(dfDatos)
print("\n\nVer si hay algun null---------------\n")
print(dfDatos.isnull())
print("\n\nLa suma de cuantos null hay---------\n")
print(dfDatos.isnull().sum())

dfDatos= dfDatos.dropna(subset=['Store', 'Date', 'Weekly_Sales'])
dfDatos = dfDatos.reset_index(drop=True)
print("\n\nDataframe dfDatos tras dropna--------:\n")
print(dfDatos)
# Imputaciones de valores faltantes
# Holiday_Flag = 0
dfDatos['Holiday_Flag'] = dfDatos['Holiday_Flag'].fillna(0)
# Temperature -> media
media_temp = dfDatos['Temperature'].mean()
dfDatos['Temperature'] = dfDatos['Temperature'].fillna(media_temp)
# Fuel_Price -> mediana
mediana_fuel = dfDatos['Fuel_Price'].median()
dfDatos['Fuel_Price'] = dfDatos['Fuel_Price'].fillna(mediana_fuel)
# CPI -> moda
moda_cpi = dfDatos['CPI'].mode()[0]
dfDatos['CPI'] =  dfDatos['CPI'].fillna(moda_cpi)
# Unemployment -> Q1
q1_unemployment = dfDatos['Unemployment'].quantile(0.25)
dfDatos['Unemployment'] = dfDatos['Unemployment'].fillna(q1_unemployment)

print("\nNulos después de imputación: ")
print(dfDatos.isnull().sum())
print("----")
print(moda_cpi)
print(mediana_fuel)
print(media_temp)

from sklearn.model_selection import train_test_split

df_X = pd.DataFrame(dfDatos, columns=[
    'Store', 'Date', 'Holiday_Flag', 'Termperature', 'Fuel_Price', 'CPI',
    'Unemployment', 'Weekly_Rains', 'Weekly_Discounts'])
df_Y = pd.DataFrame(dfDatos, columns=['Weekly_Sales'])

df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(df_X, df_Y,
                                            test_size=0.2, random_state=100)
### test_size= 0.2 (20% de prueba)
print("\nCantidad de filas y columnas de X: ", df_X.shape)
print("\nCantidad de filas y columnas de X_train: ", df_X_train.shape)
print("\nCantidad de filas y columnas de X_test: ", df_X_test.shape)

print("\nCantidad de filas y columnas de Y: ", df_Y.shape)
print("\nCantidad de filas y columnas de Y_train: ", df_Y_train.shape)
print("\nCantidad de filas y columnas de Y_train: ", df_Y_test.shape)

print("\n\nDataFrame df_X: ")
# (Usamos .head() para mostrar los 5 primeros)
print(df_X)
print("\n\nDataFrame df_X_train: ")
print(df_X_train)
print("\n\nDataFrame df_X_test: ")
print(df_X_test)

#%% Ejercicio 2 walmart - train_test_split
"""
1 Carga el archivo walmart.csv en un DataFrame llamado dfTiendas. 
    Elimina todas las filas que contengan valores nulos en las columnas ‘Store’, 
    ‘Weekly_Sales’ y ‘Fuel_Price’. Después, reinicia el índice del DataFrame 
    para que las filas queden correctamente numeradas. Por último, muestra el
    DataFrame resultante después de la limpieza.
"""
import pandas as pd

dfTiendas = pd.DataFrame()
dfTiendas = pd.read_csv( r"C:\Users\Mañana\Documents\akoarguel\Sistemas-de-Aprendizaje-Automatico\Prácticas\UNIDAD 2\Ejemplos clase - spyder\Walmart_Sales.csv")
dfTiendas= dfTiendas.dropna(subset=['Store', 'Weekly_Sales', 'Fuel_Price'])
dfTiendas = dfTiendas.reset_index(drop=True)
print("\n\nDataframe dfDatos tras dropna--------:\n")
print(dfTiendas)
"""
2 A partir del DataFrame del ejercicio anterior, realiza imputaciones personalizadas:
    - Sustituye los valores nulos en ‘Holiday_Flag’ por 1.
    - Sustituye los valores nulos en ‘CPI’ por la lo más frecuente de esa columna.
    - Sustituye los valores nulos en ‘Temperature’ por el valor mínimo (min) de la columna.
"""
# Holiday_Flag = 1
dfTiendas['Holiday_Flag'] = dfTiendas['Holiday_Flag'].fillna(1)
# CPI -> moda
moda_cpi = dfTiendas['CPI'].mode()[0]
### te devuelve el primer valor de un array (porque pueden estar empatados)
dfTiendas['CPI'] =  dfTiendas['CPI'].fillna(moda_cpi)
# Temperature -> valor minimo
minimo = dfTiendas['Temperature'].min()
dfTiendas['Temperature'] = dfTiendas['Temperature'].fillna(minimo)

print("\nNulos después de imputación: ")
print(dfTiendas.isnull().sum())
print("----")
print(moda_cpi)
print(minimo)

"""
3 Realiza una última imputación para sustituir los valores nulos en ‘Unemployment’ 
    por el cuartil 3 (Q3) de esa columna. Actualiza el DataFrame y muestra
    cuántos nulos quedan por columna. Verifica que no quedan datos ausentes.
"""
# Unemployment -> Q1
q3_unemployment = dfTiendas['Unemployment'].quantile(0.25)
dfTiendas['Unemployment'] = dfTiendas['Unemployment'].fillna(q3_unemployment)
"""
4 A partir del DataFrame dfTiendas ya limpio y sin valores nulos, crea dos 
    nuevos DataFrames:
        •	df_X con columnas ['Store', 'Holiday_Flag', 'Temperature', 
                                'Fuel_Price', 'CPI', 'Unemployment']
        •	df_Y con la columna ['Weekly_Sales']
    Una vez creados, utiliza la función train_test_split() para dividir los 
    datos en 85% entrenamiento y 15% test
"""
...
#%% Ejercicio 1 walmart
"""
1. Carga walmart.csv, crea dos columnas categóricas (no numéricas) nuevas
distintas, por ejemplo: Customer_Flow: ["Bajo", "Medio", "Alto", "Muy_Alto"].
Una columna de dos valores y otra de cuatro
"""


import pandas as pd
import random

dfDatos = pd.DataFrame()

dfDatos = pd.read_csv( r"C:\Users\Mañana\Documents\akoarguel\Sistemas-de-Aprendizaje-Automatico\Prácticas\UNIDAD 2\Ejemplos clase - spyder\Walmart_Sales.csv")
    
valoresPosiblesCustomerFlow = ["Bajo", "Medio", "Alto", "Muy_Alto"]
valoresPosiblesCustomerFlow2 = ["NO", "SI"]
  
# Ejercicio 2 walmart

"""
Define una función llamada insertarColumnas que use bucles para insertar las
columnas que has creado en el dataframe y que utilice solo los datos de la 
‘Store’ número 2. Imprime.el datraframe resultante
"""

def insertarColumnas(vpcf1, vpcf2, dfDatos):
    customer_flow = []
    for i in range(dfDatos.shape[0]):
        customer_flow.append(vpcf1[random.randint(0, 3)])
    dfDatos["Ventas"] = customer_flow
    
    customer_flow2 = []
    for i in range(dfDatos.shape[0]):
        customer_flow2.append(vpcf2[random.randint(0, 1)])
    dfDatos["Partidad al LOL"] = customer_flow2
    
    dfDatos = dfDatos[dfDatos.Store == 2]
    return dfDatos

dfDatos = insertarColumnas(valoresPosiblesCustomerFlow, valoresPosiblesCustomerFlow2, dfDatos)
print(dfDatos.shape[0])
print("\n\nTipo de datos:")
print(type(dfDatos))
print("\n\nDataframe dfDatos----------:\n")
print(dfDatos)


# Ejercicio 3 walmart
"""
Verifica que tu dataFrame no tiene ningún dato nulo mostrándo la suma de los
null
"""

print("\n\nVer si hay algun null----------:\n")
print(dfDatos.isnull())
print("\n\nLa suma de cuantos null hay----------:\n")
print(dfDatos.isnull().sum())


