# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:05:08 2025

@author: manel
"""

import numpy as np
import matplotlib.pyplot as plt

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
