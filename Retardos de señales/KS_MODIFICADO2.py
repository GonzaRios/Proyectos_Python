# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:24:55 2020

@author: riosg
"""


import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from pylab import figure



Fs = 44100
T=3
N = int(np.round(Fs/1000,0))
t = int(np.round(Fs*T))
x = np.zeros(t)
y =np.zeros(t)
x[0:N] = np.random.randn(N)
y[0:N] = x[0:N]
y[N]=x[1]
M = t-N-1

for i in range(M):
    j = i+N+1
    y[j]=x[j]+(1/2)*(y[j-N]+y[j-N-1])

A = len(y)

L =  200
R= np.exp(-((np.pi)*L*(1/Fs)))
q= np.zeros(A)

for k in range(A):
    q[k] = (1-R)*y[k]+R*q[k-1]
    

sf.write('KSM2.wav', q, Fs)

w = np.arange(0,np.pi,1/Fs)
z= np.exp(1j*(-w))
Ha= (1+z)/2
Hb= z**N
Ha_b= Ha*Hb
H0 = 1/(1-(Ha_b[1:]))
H0=H0/np.max(H0)
Ha_n=(1-R)/(1-(R*z[1:]))
Ha_n=Ha_n/np.max(Ha_n)
plt.figure('RESPUESTA EN FRECUENCIA')
plt.title('RESPUESTA EN FRECUENCIA')
plt.plot(w[1:],20*np.log10(H0))
plt.plot(w[1:],20*np.log10(Ha_n))
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud [dB]')
plt.grid()
