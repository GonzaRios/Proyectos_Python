# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:14:06 2020

@author: tinch
"""
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import scipy as sc
from scipy import signal 
from pylab import figure
from Polos_ceros import zplane

f = 5000
T = 3
u = 1/2

Fs = 44100
N = int(np.round(Fs/f,0))
t = int(np.round(Fs*T))
x = np.zeros(t)
y =np.zeros(t)
x[0:N] = np.random.randn(N)
p = np.zeros(len(x))

K = int(u*N)

for i in range(N):
    p[i] = x[i]-x[i-K] 


y[0:N] = p[0:N]
y[N]= p[1]
M = t-N-1

for i in range(M):
    j = i+N+1
    y[j]=p[j]+(1/2)*(y[j-N]+y[j-N-1])

sf.write('KS_extensiones.wav', y, Fs)


w = np.arange(0,np.pi,1/Fs)
z= np.exp(1j*(-w))
Ha= (1+z)/2
Hb= z**N
Ha_b = Ha*Hb
H = 1/(1-(Ha_b[1:]))
Hp = H*(1-z[1:]**(-u*N))
plt.figure('RESPUESTA EN FRECUENCIA')
plt.title('RESPUESTA EN FRECUENCIA')
H = H/np.max(H)
Hp = Hp/np.max(Hp)
plt.plot(w[1:],20*np.log10(H))
plt.plot(w[1:],20*np.log10(Hp))
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud [dB]')
plt.grid()

