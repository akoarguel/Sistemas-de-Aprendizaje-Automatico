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
print("\n\nDataframe dfDatos-------------------")
print(dfDatos)
print("\n\nVer si hay algun null---------------")
print(dfDatos.isnull())
print("\n\nLa suma de cuantos null hay---------")
print(dfDatos.isnull().sum())

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



