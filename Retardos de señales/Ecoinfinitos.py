import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt 
from pylab import figure as figure
from Polos_ceros import zplane
import scipy as sc
from scipy import signal 



def Ecoinfinito(a,d,t):
    """
       La siguiente función permite aplicar un delay infinito de ecos al archivo de audio que ingresa al sistema,
       y elegir los parámetros "a" y "d", que corresponden a la atenuación y el valor de retardo en muestras,
       respectivamente. Cabe aclarar que "a" se encuentra entre 0 y 1 y se recomienda 0.1<d<0.5. Además, al tener una Fs de 44k1 muestras por segundo
       y la duración del audio es 1.6s,  para  que los ecos sean perceptibles se recomienda usar valores entre 10k y 17k muestras.
       Además, permite elegir la duración en segundos 't' como criterio de corte del audio final.
       Si se desea mantener la duración original del audio  ingrese t = 0.
       Por último, se genera un archivo .wav con el nombre 'ecoinfinito.wav' que se guarda en la carpeta donde se este corriendo la función.
    """
    x , fs = sf.read('Midi69.wav')
    D = int(fs*d)
    if t == 0:
        N = len(x)
        y = np.zeros(N)
    else:
        m = t*fs
        x = np.append(x,np.zeros(m-len(x)))
        N = len(x)
        
    y = np.zeros(N)
    for i in range(N):
        
        y[i] = x[i]+a*y[i-D] #Ecuación en diferencia que caracteriza sistema. 
    
    sf.write('ecoinfinito.wav', y, fs) 
    return D
    

#Si se desea analizar los graficos de Respuesta en frecuencia descomentar la siguiente sección
#y variar los valores a,t y el delay en segundos d o bien, comentar la función y elegir  el valor D en muestras manualmente.
# para visualizar de mejor manera el comportamiento de ceros y polos. Se recomienda ingresar manualmente el valor D
# y tomar valores pequeños del mismo, por ej. 1<D<10
    
#Respuesta en frecuencia
fs = 44100
d = 0.1
a1=0.2
a2 = 0.5
a3=0.8
t= 0
#D = Ecoinfinito(a, d, t) #DESCOMENTAR ESTA LÍNEA PARA OBTENER EL VALOR D SEGUN LAS VARIABLES DE ENTRADA
D1=9 #DESCOMENTAR ESTA LÍNEA SI SE QUIERE INGRESAR D MANUALMENTE
omega = np.arange(0,np.pi,1/fs)
H1=  np.exp(1j*omega*D1)/(np.exp(1j*omega*D1)-a1)
plt.figure('RESPUESTA EN FRECUENCIA Y FASE')
plt.subplot(3,1,1)
plt.title('RESPUESTA EN FRECUENCIA')
cuadro = 'a='+str(a1)
plt.plot(omega,abs(H1),color = 'blue',linewidth = 2.5,linestyle = '-',label = cuadro )
H2= np.exp(1j*omega*D1)/(np.exp(1j*omega*D1)-a2)
cuadro2 = 'a='+str(a2)
plt.plot(omega,abs(H2),color = 'red',linewidth = 2.5,linestyle = '-',label = cuadro2 )
H3= np.exp(1j*omega*D1)/(np.exp(1j*omega*D1)-a3)
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
#Polos y ceros
b = np.zeros(D+1)
b[0] = 1
c = np.zeros(D+1)
c[0] = 1
c[D] = -a
plt.figure('POLOS Y CEROS')
zplane(b,c)
"""