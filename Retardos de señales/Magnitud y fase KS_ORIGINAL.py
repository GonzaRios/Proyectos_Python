# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:39:35 2020

@author: riosg
"""
import numpy as np
from matplotlib import pyplot as plt

Fs=44100
eje_f = np.linspace(1,22050,int(22050))
#G_sis = abs(np.cos(np.pi*eje_f*(1/Fs)))
#plt.plot(eje_f,G_sis)


N= Fs/eje_f
fase = N + 1/2
plt.plot(eje_f,fase)