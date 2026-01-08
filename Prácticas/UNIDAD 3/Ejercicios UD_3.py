# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 11:02:33 2026

@author: Mañana
"""

### Ejercicios de repaso 1
#%% Ejercicio 1
import pandas as pd

def construir_dataframe():
    ruta = r"C:\Users\Mañana\Documents\akoarguel\Sistemas-de-Aprendizaje-Automatico\Prácticas\UNIDAD 3\cotizacion.csv"
    df_original = pd.read_csv(ruta, sep=';', thousands='.', decimal=',')
    datos = df_original.drop(columns=['Nombre'])
    
    nuevo_df = pd.DataFrame({
        'Minimo': datos.min(),
        'Maximo': datos.max(),
        'Media': datos.mean()
    })
    
    return nuevo_df

resultado = construir_dataframe()
print(resultado)

#%% Ejercicio 2

def generar_dataframe():
    # a) genera un dataframe con los datos de titanic.csv
    ruta = r"C:\Users\Mañana\Documents\akoarguel\Sistemas-de-Aprendizaje-Automatico\Prácticas\UNIDAD 3\titanic.csv"
    df_titanic = pd.read_csv(ruta, sep=',')
    
    # b) imprimir dimensiones, tamaño, ínidice y las últimas lineas del dataframe
    print("\nDimensiones del df: ", df_titanic.shape)
    print("\nTamaño del df: ", df_titanic.size)
    print("\nÍndice del df: ", df_titanic.index)
    print("\nÚltimas 10 líneas: \n", df_titanic.tail(10))
    
    # c) datos del pasajero con identificador 148 con loc[]
    print("\nPasajero con identificador 148: ", df_titanic.loc[148])
    
    # d) mostrar por pantalla las filas pares usando iloc[range(...)]
    filas_pares = df_titanic.iloc[range(0, len(df_titanic), 2)]
    print("\nFilas pares (usando iloc): \n", filas_pares)
    
    # e) nombres de personas de primera clase 'Pclass == 1' ordenadas alfabeticamente
    nombres = df_titanic[df_titanic['Pclass']==1]['Name'].sort_values()
    print("\nNombres de personas de primera clase (ordendas): \n", nombres)
    
    
generar_dataframe()