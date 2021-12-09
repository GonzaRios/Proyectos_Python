# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:39:54 2020

@author: riosg
"""


import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import scipy as sc
from scipy import signal 
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
Q= 0.8
for i in range(M):
    j = i+N+1
    y[j]=x[j]-Q*x[j-1]+Q*y[j-1]+(1-abs(Q))*y[j-N]
    


w = np.arange(0,np.pi,1/Fs)
z= np.exp(1j*(-w))
Ha= (1+z)/2
Hb= z**N
Ha_b= Ha*Hb
H0 = 1/(1-(Ha_b[1:]))
H0=H0/np.max(H0)
Ha_n=(1-abs(Q))/(1-(Q*z))
Ha_b_n= 1/(1-((Ha_n*Hb)[1:]))
Ha_b_n=Ha_b_n/np.max(Ha_b_n)
plt.figure('RESPUESTA EN FRECUENCIA')
plt.title('RESPUESTA EN FRECUENCIA')
plt.plot(w[1:],20*np.log10(H0))
plt.plot(w[1:],20*np.log10(Ha_b_n))
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud [dB]')
plt.grid()
sf.write('KSM_Han.wav', y, Fs)