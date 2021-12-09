import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import scipy as sc
from scipy import signal 
from pylab import figure
from Polos_ceros import zplane

def KS_MODIFICADO(f,T,u,P,L):
 """
 En esta función se aplican tres modificaciones [1] al algoritmo original de Karplus-Strong. Las mismas fueron:
 1-Moving pick : permite elegir posición normalizada (0<u<1) donde se exite la cuerda.
 2-Other Filters in Feedback Loop: cambia el filtro en el lazo de realimentación por un único polo z=P (0<P<1 para conservar la estabilidad del sistema).  
 3-Dynamic-level lowpass filter: modifica la dinámica de la cuerda simulada aplicando un filtro pasabajos de ancho de banda 0<L<fs/2 (Hz).
 
 [1] Jaffe, A.-Smith,J. "Extensions of the Karplus-Strong Plucked-String Algorithm". Computer Music Journal, Vol. 7, No. 2 (Summer, 1983), pp. 56-69.    
 """
 Fs = 44100
 N = int(np.round(Fs/f,0))
 t = int(np.round(Fs*T))
 x = np.zeros(t)
 x[0:N] = np.random.randn(N) # Ráfaga de ruido.
 p = np.zeros(len(x))
 uN = int(u*N)

 for i in range(N):
    p[i]= x[i]-x[i-uN] # Moving pick.
    
 y =np.zeros(t)
 y[0:N] = x[0:N]
 y[N]=x[1]
 M = t-N-1

 for i in range(M):
    j = i+N+1
    y[j]=x[j]-P*x[j-1]+P*y[j-1]+(1-abs(P))*y[j-N] # Other Filters in Feedback Loop
    
 A = len(y)

 
 R= np.exp(-((np.pi)*L*(1/Fs)))
 q= np.zeros(A)

 for k in range(A):
    q[k] = (1-R)*y[k]+R*q[k-1] #Dynamic-level lowpass filter
       

 sf.write('KSM_N.wav', q, Fs)
 return N,Fs,R

# Si se desea analizar graficamente parámetros tales como Respuesta en frecuencia, Fase, polos y ceros, descomentar la siguiente sección 
# y variar los valores de entrada "f","T","u","P" y "L" en sus rangos especificados anteriormente. 

f=1000
T=3
u=0.5
P=0.4
L=200
[N,Fs,R]= KS_MODIFICADO(f,T,u,P,L)
w = np.arange(0,np.pi,1/Fs)
z= np.exp(1j*(-w))
Ha= (1-abs(P))/(1-(P*z))
Hb= z**N
He=1-(z[1:]**(u*N))
HL=(1-R)/(1-R*z[1:])
Ha_b=Hb*Ha
H_sistema= He*(1/(1-(Ha_b[1:])))*HL
H_sistema=H_sistema/np.max(H_sistema)
plt.figure('RESPUESTA EN FRECUENCIA')
plt.subplot(3,1,1)
plt.title('RESPUESTA EN FRECUENCIA')
plt.plot(w[1:],20*np.log10(H_sistema)) 
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud [dB]')
plt.grid()

#FASE 
img= np.imag(H_sistema)
re = np.real(H_sistema)
fase= np.arctan(img/re)
plt.subplot(3,1,3)
plt.plot(w[1:],abs(fase))
plt.title('FASE')
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Fase <H(e^jw)')
plt.grid()
plt.show()
