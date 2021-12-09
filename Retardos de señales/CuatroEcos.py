import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt 
from pylab import figure as figure
from Polos_ceros import zplane
import scipy as sc
from scipy import signal 


def Cuatro_Ecos(a,D):
 """
     La siguiente función permite aplicar un delay de cuatro ecos al archivo de audio que ingresa al sistema,
     y elegir los parámetros "a" y "D", que corresponden a la atenuación y el valor de retardo en muestras,
     respectivamente. Cabe aclarar que "a" se encuentra entre 0 y 1. Además, al tener una Fs de 44k1 muestras por segundo
     y la duración del audio es 1.6s,  para  que los ecos sean perceptibles se recomienda usar valores entre 10k y 17k muestras.
     Por  último, se genera un archivo en formato .wav con el nombre '4ecos.wav' que se guarda en la carpeta donde se este corriendo la función.

 """
 x , fs = sf.read('Midi69.wav') 

 N = len(x)
 y = np.zeros(N)

 for i in range(N):
    y[i]= x[i]+a*x[i-D]+(a**2)*x[i-2*D]+(a**3)*x[i-3*D]+(a**4)*x[i-4*D] #Ecuación en diferencias que caracteriza al sistema. El factor de atenuación 'a' se eleva al número n de repetición.
    
 sf.write('4ecos.wav', y, fs) 
 return a,D,fs


# Si se desea analizar graficamente parámetros tales como Respuesta en frecuencia, Fase, polos y ceros, descomentar la siguiente sección 
# y variar los valores de entrada "a" y "D". En este caso, se recomienda utilizar valores enteros de menor magnitud en la variable "D"
# para una visualización de polos y ceros más clara.

fs=44100
D1 = 5
a1= 0.2
a2=0.5
a3=0.8
# RESPUESTA EN FRECUENCIA

omega = np.arange(0,np.pi,1/fs)
H1= (1-(a1**5*np.exp(1j*(-5*omega*D1))))/(1-a1*np.exp(1j*(-1*omega*D1)))
plt.figure('RESPUESTA EN FRECUENCIA Y FASE')
plt.subplot(3,1,1)
plt.title('RESPUESTA EN FRECUENCIA')
cuadro = 'a='+str(a1)
plt.plot(omega,abs(H1),color = 'blue',linewidth = 2.5,linestyle = '-',label = cuadro )
H2= (1-(a2**5*np.exp(1j*(-5*omega*D1))))/(1-a2*np.exp(1j*(-1*omega*D1)))
cuadro2 = 'a='+str(a2)
plt.plot(omega,abs(H2),color = 'red',linewidth = 2.5,linestyle = '-',label = cuadro2 )
H3= (1-(a3**5*np.exp(1j*(-5*omega*D1))))/(1-a3*np.exp(1j*(-1*omega*D1)))
cuadro3 = 'a='+str(a3)
plt.plot(omega,abs(H3),color = 'green',linewidth = 2.5,linestyle = '-',label = cuadro3 )

plt.legend(loc='lower right')
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud |H(e^jw)|')
plt.grid()

#FASE
img1= np.imag(H1)
real1 = np.real(H1)
fase1 = np.arctan(img1/real1)
img2= np.imag(H2)
real2 = np.real(H2)
fase2 = np.arctan(img2/real2)
img3= np.imag(H3)
real3 = np.real(H3)
fase3 = np.arctan(img3/real3)
plt.subplot(3,1,3)
plt.title('FASE')
plt.plot(omega,fase1,color = 'blue',linewidth = 2.5,linestyle = '-',label =cuadro )
plt.plot(omega,fase2,color = 'red',linewidth = 2.5,linestyle = '-',label =cuadro2 )
plt.plot(omega,fase3,color = 'green',linewidth = 2.5,linestyle = '-',label =cuadro3 )
plt.legend(loc='lower right')
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Fase <H(e^jw)')
plt.grid()
plt.show()

"""
#POLOS Y CEROS

b= np.zeros(5*D+1)
b[0] = 1
b[5*D] = -a**5
c= np.zeros(4*D+1)
c[0]=1
c[D]=-a
c[4*D]= 0
plt.figure('POLOS Y CEROS')
zplane(b,c)
"""
"""
w = np.arange(0,np.pi,1/fs)
z= np.exp(1j*(-w))
b= (1-((a**5)*(z**(5*D))))
c=(1-(a*(z**D)))
k , h = sc.signal.freqz(b,c)
plt.figure('RESPUESTA EN FRECUENCIA2')
plt.plot(k,abs(h))
plt.xlabel('Frecuencia angular [rad/s]')
plt.ylabel('Amplitud |H(e^jw)|')
plt.grid()
"""