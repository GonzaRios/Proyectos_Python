import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import scipy as sc
from scipy import signal 
from pylab import figure
from Polos_ceros import zplane


def KS_ORIGINAL(f,T):
 """
 La función en cuestión permite generar un archivo .wav que simula una cuerda pulsada mediante el algoritmo de Karplus-Strong.
 Se cuenta con dos variables de entrada, las cuales permiten modificar tanto la frecuencia fundamental (f),como la duración
 del audio a generar en segundos(T).
 """
 Fs = 44100

 N = int(np.round(Fs/f,0))
 t = int(np.round(Fs*T))
 x = np.zeros(t)
 y =np.zeros(t)
 x[0:N] = np.random.randn(N) # Ráfaga de ruido aleatoria que se utiliza como impulso.
 y[0:N] = x[0:N]
 y[N]=x[1]
 M = t-N-1

 for i in range(M):
    j = i+N+1
    y[j]=x[j]+(1/2)*(y[j-N]+y[j-N-1]) # Ecuación en diferencias del algoritmo.

 sf.write('KS.wav', y, Fs) 
 
 return N

# Si se desea analizar graficamente parámetros tales como Respuesta en frecuencia, Fase, polos y ceros, descomentar la siguiente sección 
# y variar los valores de entrada "f" y "T". 


fs= 44100
f=1000
T=3
N= KS_ORIGINAL(f, T)
w = np.arange(0,np.pi,1/fs)
z= np.exp(1j*(-w))
Ha= (1+z)/2
Hb= z**N
Ha_b= Ha*Hb 
H = 1/(1-(Ha_b[1:]))
plt.figure('RESPUESTA EN FRECUENCIA Y FASE')
plt.subplot(3,1,1)
plt.title('RESPUESTA EN FRECUENCIA')
H=H/np.max(H)
plt.plot(w[1:],20*np.log10(H))
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud  [dB]')
plt.grid()

#FASE 
img= np.imag(H)
re = np.real(H)
fase= np.arctan(img/re)
plt.subplot(3,1,3)
plt.title('FASE')
plt.plot(w[1:],abs(fase))
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Fase <H(e^jw)')
plt.grid()
plt.show()
"""
fs= 44100
N,t= KS_ORIGINAL(1000, 1)
w = np.arange(0,np.pi,1/fs)
z2= np.exp(1j*(-w))
Ha= (1+z2)/2
Hb= z2**N
Ha_b= Ha*Hb 
H = 1/(1-(Ha_b[1:]))
plt.plot(w[1:],abs(H))
"""
"""
z= np.exp(1j*(w))
b= (z**(N+1))
a=((z**(N+1))-(z+0.5))
k , h = sc.signal.freqz(b,a)
plt.figure('RESPUESTA EN FRECUENCIA')
plt.plot(k,abs(h))
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud |H(e^jw)|')
plt.grid()
"""