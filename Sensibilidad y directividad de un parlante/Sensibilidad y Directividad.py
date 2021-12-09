# -- coding: utf-8 --
"""
Created on Thu Jul  8 01:00:29 2021

@author: NPass
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from suavizado import suavizado

sens_df = pd.read_excel('Sensibilidad y Directividad - Curvas Smart.xlsx', sheet_name='Sensibilidad')
sens_values = sens_df.to_numpy()

direc_df = pd.read_excel('Sensibilidad y Directividad - Curvas Smart.xlsx', sheet_name='Directividad')
direc_values = direc_df.to_numpy()

# =============================================================================
# SENSIBILIDAD
# =============================================================================
class Sens:
    freq = sens_values[2:,0]
    class sound:
        pink = sens_values[2:,1]
        f_100_500 = sens_values[2:,4]
        f_171 = sens_values[2:,7]
        
# =============================================================================
# DIRECTIVIDAD
# =============================================================================
class Direc:
    freq = direc_values[1:,0]
    class gr: # Grados
        _0 = direc_values[1:,1]
        _15 = direc_values[1:,6]
        _30 = direc_values[1:,11]
        _45 = direc_values[1:,16]
        _60 = direc_values[1:,21]
        _75 = direc_values[1:,26]
        _90 = direc_values[1:,31]
    class coh: # Coherencia
        _0 = direc_values[1:,3]
        _15 = direc_values[1:,8]
        _30 = direc_values[1:,13]
        _45 = direc_values[1:,18]
        _60 = direc_values[1:,23]
        _75 = direc_values[1:,28]
        _90 = direc_values[1:,33]

# =============================================================================
# MEDICIONES DE REFERENCIA
# =============================================================================
class Ref:
    pink = 84
    f_171 = 94.8
    f_100_500 = 87.3

sens = Sens()
direc = Direc()
ref = Ref()

# =============================================================================
# CORRECCION DE LAS CURVAS DE SENSIBILIDAD y DIRECTIVIDAD
# =============================================================================

    # Valor mas cercano en frecuencia a 171 Hz
dif = np.abs(sens.freq - 171)
index_closest_171 = dif.argmin()

    # Diferencia entre la referencia y la curva del Smaart
default_dBSPL = sens.sound.f_171[index_closest_171]
ref_dBSPL = ref.f_171
dif_dBSPL = ref_dBSPL - default_dBSPL

## Corrección de sensibilidad tomando como referencia los dBSPL a 171 Hz
    # Correción de calibración
sens.sound.f_171 = sens.sound.f_171 + dif_dBSPL
sens.sound.f_100_500 = sens.sound.f_100_500 + dif_dBSPL
sens.sound.pink = sens.sound.pink + dif_dBSPL

    # Correción energetica
dif_sens_pink = sens.sound.f_171[index_closest_171] - sens.sound.pink[index_closest_171]
sens.sound.pink = sens.sound.pink + dif_sens_pink

dif_sens_100_500 = sens.sound.f_171[index_closest_171] - sens.sound.f_100_500[index_closest_171]
sens.sound.f_100_500 = sens.sound.f_100_500 + dif_sens_100_500

## Correción de directividad tomando como referencia los dBSPL a 171 Hz
direc.gr._0 = direc.gr._0 + dif_dBSPL
#Normalización con respecto a la directividad en 0º
direc.gr._15 = (direc.gr._15 + dif_dBSPL)/direc.gr._0
direc.gr._30 = (direc.gr._30 + dif_dBSPL)/direc.gr._0
direc.gr._45 = (direc.gr._45 + dif_dBSPL)/direc.gr._0
direc.gr._60 = (direc.gr._60 + dif_dBSPL)/direc.gr._0
direc.gr._75 = (direc.gr._75 + dif_dBSPL)/direc.gr._0
direc.gr._90 = (direc.gr._90 + dif_dBSPL)/direc.gr._0
direc.gr._0 = direc.gr._0/direc.gr._0

#Sonograma
direc_matrix = np.array([direc.gr._90,
                         direc.gr._75,
                         direc.gr._60,
                         direc.gr._45,
                         direc.gr._30,
                         direc.gr._15,
                         direc.gr._0,
                         direc.gr._15,
                         direc.gr._30,
                         direc.gr._45,
                         direc.gr._60,
                         direc.gr._75,
                         direc.gr._90])



def promedio_por_octava(GR):
    oct_bands = np.array([63,125,250,500,1000,2000,4000,8000])
    prom_per_oct = np.empty(len(oct_bands))
    for i in oct_bands:
        finf = i * 2**(-0.5)
        fsup = i * 2**(0.5)
        dif_freq_sup = np.abs(direc.freq - fsup)
        dif_freq_inf = np.abs(direc.freq - finf)
        index_fsup=np.argmin(dif_freq_sup)
        index_finf = np.argmin(dif_freq_inf)
        prom_i = np.mean(GR[index_finf:index_fsup])
        prom_per_oct[np.where(oct_bands==i)[0][0]] = prom_i 
        
    return prom_per_oct

direc_oct_0 = promedio_por_octava(direc.gr._0)
direc_oct_15 = promedio_por_octava(direc.gr._15)
direc_oct_30 = promedio_por_octava(direc.gr._30)
direc_oct_45 = promedio_por_octava(direc.gr._45)
direc_oct_60 = promedio_por_octava(direc.gr._60)
direc_oct_75 = promedio_por_octava(direc.gr._75)
direc_oct_90 = promedio_por_octava(direc.gr._90)

direc_matrix_oct = np.array([direc_oct_90,
                             direc_oct_75,
                             direc_oct_60,
                             direc_oct_45,
                             direc_oct_30,
                             direc_oct_15,
                             direc_oct_0,
                             direc_oct_15,
                             direc_oct_30,
                             direc_oct_45,
                             direc_oct_60,
                             direc_oct_75,
                             direc_oct_90])

gr1= direc_matrix_oct[:,0]
gr2 = direc_matrix_oct[:,1]
gr3= direc_matrix_oct[:,2]
gr8= direc_matrix_oct[:,7]
# Crear eje polar

s = pd.Series(np.arange(1))
theta=np.arange(-np.pi/2,np.pi/2+0.1,np.pi/12)
print(s.head())
print(theta[:10])
# Crear datos

fig = plt.figure(figsize=(8,4))
ax1 = plt.subplot(121, projection = 'polar')
#ax2 = plt.subplot(122)

# Crear subgrafo de coordenadas polares
# También puedes escribir: ax = fig.add_subplot (111, polar = True)

ax1.plot(theta,gr8,linestyle = '--',lw=1)  
ax1.plot(s, linestyle = '--', marker = '.',lw=2)
#ax2.plot(theta,theta*3,linestyle = '--',lw=1)
#ax2.plot(s)
#plt.grid()

# Cree un gráfico de coordenadas polares, el parámetro 1 es el ángulo (sistema en radianes), el parámetro 2 es el valor
# lw → ancho de línea




# =============================================================================
# GRÁFICO DE SENSIBILIDAD
# =============================================================================

"""
A = suavizado(sens.freq,sens.sound.pink,3)
plt.semilogx(A)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Sensibilidad [dB SPL]')
plt.grid()
"""
# =============================================================================
# GRAFICO DE DIRECTIVIDAD
# =============================================================================
# plt.figure(1,[10,5])
# plt.contourf(direc.freq,np.arange(13),direc_matrix)
# #x_label1 = [r"$63$",r"$125$",r"$250$",r"$500$",r"$1,000$",r"$2,000$",r"$4,000$",r"$8,000$",r"$63$",r"$16,000$"]
# #plt.xticks(np.array([10000,10100]),x_label1)
# plt.xlim([40,16000])
# y_label = [r"$-90^o$",r"$-45^o$", r"$0^o$", r"$45^o$",r"$90^o$"]
# #plt.xticks(x_label1)
# plt.yticks(np.array([0,3,6,9,12]),y_label)
# plt.xlabel('Frecuencia [Hz]', fontsize = 14)
# plt.ylabel('Ángulo [$^o$]', fontsize = 14)


"""
plt.semilogx(sens.freq,sens.sound.pink)
plt.semilogx(sens.freq,sens.sound.f_100_500)
plt.semilogx(sens.freq,sens.sound.f_171)
plt.xlim([50,12000])
"""