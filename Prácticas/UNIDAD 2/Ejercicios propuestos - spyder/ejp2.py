# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:05:08 2025

@author: manel
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.impute import SimpleImputer
import pandas as pd

#%% Ejercicio 15
x = np.random.randint(0,90, (3,1))
y = np.random.randint(8,512, (3,1))
plt.plot(x, y, color='y', marker='o')
plt.title("Gráfico ejercicio 1")
plt.xlabel("eje x")
plt.ylabel("eje y")
plt.show()

#%% Ejercicio 16
meses = ["enero", "febrero", "marzo"]
ventas = np.random.randint(1000,2000, (3,1))

plt.plot(meses, ventas, marker='o')
plt.title("Ventas mensuales")
plt.xlabel("MESES")
plt.xlim(-1,3)
plt.ylim(900, 2100)
plt.show()

#%% Ejercicio 17
meses = ["enero", "febrero", "marzo"]
ventas = np.random.randint(1000,2000, (3,1))
x = np.random.randint(0,90, (3,1))
y = np.random.randint(8,512, (3,1))

fig, axs = plt.subplots(1,2, figsize=(8,3))
axs[0].plot(meses, ventas, "--")
axs[0].set_title("Ventas menusales")
axs[0].set_xlabel("meses")
axs[0].set_xlim(-1,3)
axs[0].set_ylim(900,2100)

axs[1].plot(x, y, linestyle='--', marker='*', color="m")
axs[1].set_title("Grafico ejercicio 1")
axs[1].set_xlabel("eje x")
axs[1].set_xlim(0,90)
axs[1].set_ylim(0,600)

#%% Conjunto de datos scikit-learn
 
datos_diabetes = datasets.load_diabetes()
#datos_boston = datasets.load_boston()
datos_iris = datasets.load_iris()
datos_cancer = datasets.load_breast_cancer()
datos_digits = datasets.load_digits()
 
#elegimos un dataset
datos = datos_iris
print(datos.keys())
print("\nNombres de las caracteristicas -----")
print(datos.feature_names)
print("\nNombres del objetivo -----")
print(datos.target_names)
print("\nNombres de frame -----")
print(datos.frame)
print("\nNombres de filename -----")
print(datos.filename)
print("\n\n\n Description -----")
print(datos.DESCR)
 
print("\nDescripcion datos digits -----")
print(datos_digits.keys())
print(datos_digits.DESCR)
 
 
df_x = pd.DataFrame(datos.data, columns=datos.feature_names)
#index = datos.index
print("\nDataframe de caracteristicas (X)")
print(df_x.head())

#%% Ejercicio 1
'''1 Se pide cargar el conjunto de datos breast_cancer de Scikit-learn, 
convertirlo en un DataFrame de pandas usando los nombres de las 
características como columnas y mostrar por pantalla las cinco 
primeras filas del DataFrame resultante.'''

datos_cancer = datasets.load_breast_cancer()

df =pd.DataFrame(datos_cancer.data, columns=datos_cancer.feature_names)

print("Cinco primeras filas")
print(df.head())


#%% Ejercicio 2
'''A partir del DataFrame anterior, se solicita aplicar un objeto 
SimpleImputer de Scikit-learn configurado con la estrategia "most_frequent"
 para sustituir todos los valores 0.0 del conjunto de datos por el valor 
 más frecuente de cada columna, y guardarlo en un nuevo DataFrame llamado 
 dfImputado. Mostrarlo por pantalla.'''

imputer = SimpleImputer(missing_values=0.0, strategy='most_frequent')
datos_imputados = imputer.fit_transform(df) 

dfImputado=pd.DataFrame(data=datos_imputados, columns=df.columns)
print("Df por pantalla")
print(dfImputado.head())
 
#%% Ejercicio 3
''' Crear un DataFrame llamado dfCancerModificado usando el método .copy() 
en el dfImputado, que elimine la columna "worst symmetry" y ordene el 
DataFrame por la columna "mean texture" en orden ascendente.'''

dfCancerModificado = dfImputado.copy()
dfCancerModificado = dfCancerModificado.drop("worst symmetry", axis=1)
dfCancerModificado = dfCancerModificado.sort_values(by="mean texture", ascending=True)

print("Df Cancer modificado")
print(dfCancerModificado.head())   