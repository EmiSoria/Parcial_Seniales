import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import scipy.signal as sig

#Diseño Filtro FIIR usando ventanas, y para verificar la salida del filto usar la
#convolucion 

fc1 =200 # Frecuencia de corte1
fc2=600   #Frecuencia de corte 2 
Fs =3000 # Frecuencia de muestreo en la toma datos 
L = 500+1 # orden del filtro

#N=512

h = sig.firwin(L, (fc1,fc2), window='hamming', pass_zero='bandpass', fs=Fs)
freq, H = sig.freqz(h, fs=Fs)

#print(freq)
#print(H)


#Probemos si filtra 

# señal en el tiempo $
Tmax=4
t = np.arange(0,Tmax, 1/Fs)
x = np.sin(2*np.pi*300*t)+np.sin(2*np.pi*1500*t)  #Probar cambiando la frecuenciamas alta
N=len(x)
print(N)
freq2 = fft.fftfreq(N,d=1/Fs)
X = fft.fft(x,N)  #X[w]

senial_fft_modX = np.abs(X) / N 

# salida del filtro
y=np.convolve(x,h)
Y = fft.fft(y,N)
senial_fft_modY = np.abs(Y) / N 



# Grafica de Espectros

fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
fig1.suptitle("Espectros de frecuencias", fontsize=18)

#Grafica de la señal temporal y del espectro en frecuencias
# Se grafica la señal temporal
ax1[0].plot(freq2, senial_fft_modX)
ax1[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1[0].set_ylabel('Magnitud [V]', fontsize=15)
ax1[0].set_title('Magnitud de la Respuesta en Frecuencia de señal de entrada', fontsize=15)
ax1[1].set_xlim([-3000, 3000])
ax1[0].grid()

# se grafica la magnitud de la respuesta en frecuencia
ax1[1].plot(freq2, senial_fft_modY )
ax1[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1[1].set_ylabel('Magnitud [V]', fontsize=15)
ax1[1].set_title('Magnitud de la Respuesta en Frecuencia de señal de salida ', fontsize=15)
ax1[1].set_xlim([-3000, 3000])
ax1[1].grid()

plt.show()



#Graficas  

fig1,axs1 = plt.subplots(4,1)
fig1.set_size_inches((15,16))
plt.subplots_adjust(hspace=0.3)


# Respuesta al impulso
ax=axs1[0]
ax.plot(range(len(h)), h)
ax.set_xlabel("$frec$",fontsize=12)
ax.set_ylabel("$hi[w]$",fontsize=12)

# Respuesta en frecuencia del filtro
ax=axs1[1]
ax.plot(freq, np.absolute(H))
ax.set_xlabel("$Frec$",fontsize=12)
ax.set_ylabel("$h[w]$",fontsize=12)

# Señal de Entrada  
ax=axs1[2]
ax.plot(x[0:1000], label="señal de entrada")
ax.set_xlabel("$n$",fontsize=12)
ax.set_ylabel("$x[n]$",fontsize=12)

#Señal de salida
ax=axs1[3]
ax.plot(y[0:2000], label="señal de salida")
ax.set_xlabel("$n$",fontsize=12)
ax.set_ylabel("$y[n]$",fontsize=12)

plt.legend()
plt.grid()
plt.show() 
