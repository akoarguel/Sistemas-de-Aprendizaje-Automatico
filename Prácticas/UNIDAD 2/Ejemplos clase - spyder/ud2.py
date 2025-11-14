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