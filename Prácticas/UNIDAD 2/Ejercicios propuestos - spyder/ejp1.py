# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:22:49 2025

@author: manel
"""
import numpy as np

#%% Ejercicio 11
x1 = np.array([713])
x2 = np.array([[1,1,7], [1,1,7]])
x3 = np.zeros((3,3))

print("x1: \n", x1)
print("x2: \n", x2)
print("x3: \n", x3)

#%% Ejercicio 12
identidad = np.identity(5)
print(f"indentidad: \n {identidad}")
array_random = np.random.random((3,3))
print("random: \n", array_random)

#%% Ejercicio 13
y1 = np.random.randint(10,50, (2,2))
y2 = np.random.randint(10,50, (2,2))

print("y1: \n", y1)
print("y2: \n", y2)
print("suma: \n", y1+y2)
print("multiplicación: \n", y1*y2)

#%% Ejercicio 14
z = np.random.randint(0,100, (5,5))
print("Array z: \n", z)
print("Máximo: \n", np.max(z))
print("Mínimo: \n", np.min(z))
print("Media: \n", np.mean(z))
