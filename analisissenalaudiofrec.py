#Analisis en frecuencia de una señal de audio aplicar un filtro pasabajo a 
# un archivo .wav
# Incluir librerías
from scipy import signal
from scipy import fft
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


filename = 'Alarm01'                       # nombre de archivo
fs, data = wavfile.read(f'{filename}.wav') # frecuencia de muestreo y datos de la señal

#-----------Definición de parámetros temporales:

ts = 1 / fs                             # tiempo de muestreo
N = len(data)                           # número de muestras en el archivo de audio
t = np.linspace(0, N * ts, N)           # vector de tiempo

if len(data.shape) > 1:
    senial = data[:, 0]                 # Si el audio es estereo, se extrae un canal de la pista
else:
    senial = data
#senial = senial * 3300.0 / (2 ** 16 - 1)# se escala la señal a mV (considerando un CAD de 16bits y Vref 3.3V)




freq = fft.fftfreq(N, d=1/fs)   # se genera el vector de frecuencias
senial_fft = fft.fft(senial)    # se calcula la transformada rápida de Fourier

# El espectro es simétrico, nos quedamos solo con el semieje positivo
f = freq[np.where(freq >= 0)]
#senial_fft = senial_fft[np.where(freq >= 0)]

# Se calcula la magnitud del espectro
senial_fft_mod = np.abs(senial_fft) / N     # Respetando la relación de Parceval
# Al haberse descartado la mitad del espectro, para conservar la energía
# original de la señal, se debe multiplicar la mitad restante por dos (excepto
# en 0 y fm/2)
#senial_fft_mod[1:len(senial_fft_mod-1)] = 2 * senial_fft_mod[1:len(senial_fft_mod-1)]


# Se crea una gráfica que contendrá dos sub-gráficos ordenados en una fila y dos columnas
fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
fig1.suptitle(filename, fontsize=18)

#Grafica de la señal temporal y del espectro en frecuencias
# Se grafica la señal temporal
ax1[0].plot(t, senial)
ax1[0].set_xlabel('Tiempo [s]', fontsize=15)
ax1[0].set_ylabel('Tensión [V]', fontsize=15)
ax1[0].set_title('Señal temporal', fontsize=15)
ax1[0].set_xlim([0, ts*N])
ax1[0].grid()

# se grafica la magnitud de la respuesta en frecuencia
ax1[1].plot(freq, senial_fft_mod)
ax1[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1[1].set_ylabel('Magnitud [V]', fontsize=15)
ax1[1].set_title('Magnitud de la Respuesta en Frecuencia', fontsize=15)
ax1[1].set_xlim([-5000, 5000])
ax1[1].grid()

plt.show()


#Implementacion de un filtro

"""
#IIR
s = 360

# Se analiza el orden necesario para cumplir los requisitos utilizando un filtro Butterworth
N, wn = signal.buttord([5, 15], [1, 50], 3, 20, analog=False, fs=fs)
# Se genera el filtro con el orden calculado
sos_iir_1 = signal.butter(N, wn, 'bandpass', analog=False, output='sos', fs=fs)

print("Secciones de orden 2:")
print(sos_iir_1)
"""

#FIR
# Proponemos un orden
#Filtro 1: Frecuencia de corte 500Hz, ventana Hamming y Orden 2001
L1 = 2001 #Orden del filtro 
num_fir_1 = signal.firwin(L1, cutoff=500, window='hamming', pass_zero='lowpass', fs=fs)

#Filtro 2: Frecuencia de corte 500Hz, ventana Hann y Orden 1001
L2 = 1001 #Orden del filtro
num_fir_2 = signal.firwin(L2, cutoff=500, window='hann', pass_zero='highpass', fs=fs)

#Filtro 3: Frecuencia de corte 500Hz, ventana Blackman y Orden 1001
L3 = 1001 #Orden del filtro
num_fir_3 = signal.firwin(L3, cutoff=500, window='blackman', pass_zero='highpass', fs=fs)

#print("Coeficientes del filtro:")
#print(num_fir_1)

# se genera un vector de frecuencias
f = np.logspace(-1, 3, int(1e3))
# se analiza la respuesta en frecuencia de ambos filtros
#f, h_iir_1 = signal.sosfreqz(sos_iir_1, worN=f, fs=fs)
f, h_fir_1 = signal.freqz(num_fir_1, 1, worN=f, fs=fs)
f, h_fir_2 = signal.freqz(num_fir_2, 1, worN=f, fs=fs)
f, h_fir_3 = signal.freqz(num_fir_3, 1, worN=f, fs=fs)

# se grafican ambas respuestas en frecuencia superpuestas
fig1, ax1 = plt.subplots(1, 3, figsize=(20, 7), sharex=True)
fig1.suptitle('Frecuenca de los filtros FIR pasa-alto', fontsize=18)

ax1[0].plot(f, 20*np.log10(abs(h_fir_1)))
ax1[0].set_ylabel('Ganancia [dB]', fontsize=15)
ax1[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1[0].grid(which='both')
ax1[0].legend(loc="lower right", fontsize=15)
ax1[0].set_title('Ventana Hamming, Orden 2001', fontsize=15)
ax1[0].set_xscale('log')
ax1[0].set_xlim([1, 10000])
ax1[0].set_ylim([-80, 10])

ax1[1].plot(f, 20*np.log10(abs(h_fir_2)))
ax1[1].set_ylabel('Ganancia [dB]', fontsize=15)
ax1[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1[1].grid(which='both')
ax1[1].legend(loc="lower right", fontsize=15)
ax1[1].set_title('Ventana Hann, Orden 1001', fontsize=15)
ax1[1].set_xscale('log')
ax1[1].set_xlim([1, 10000])
ax1[1].set_ylim([-80, 10])

ax1[2].plot(f, 20*np.log10(abs(h_fir_3)))
ax1[2].set_ylabel('Ganancia [dB]', fontsize=15)
ax1[2].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1[2].grid(which='both')
ax1[2].legend(loc="lower right", fontsize=15)
ax1[2].set_title('Ventana Blackman, Orden 1001', fontsize=15)
ax1[2].set_xscale('log')
ax1[2].set_xlim([1, 10000])
ax1[2].set_ylim([-80, 10])

plt.show()



#Aplico señal al filtro propuesto y se obtiene la señal filtrada

#Filtro 1: Ventana Hamming, Orden 2001
senial_fir_1 = signal.lfilter(num_fir_1, 1, senial) #Funcion que me permite aplicar el filtro a la señal y devuelve la señal filtrada

#Filtro 2: Ventana Hann, Orden 1001
senial_fir_2 = signal.lfilter(num_fir_2, 1, senial)

#Filtro 3: Ventana Blackman, Orden 1001
senial_fir_3 = signal.lfilter(num_fir_3, 1, senial)

ts = 1 / fs                             # tiempo de muestreo
N = len(senial_fir_1)   

ts = 1 / fs                             # tiempo de muestreo
N = len(senial_fir_2)   

ts = 1 / fs                             # tiempo de muestreo
N = len(senial_fir_3)   

# graficación de las señales Original y filtrada 
fig2, ax2 = plt.subplots(1, 3, figsize=(20, 7), sharex=True)
fig2.suptitle('Señales original y filtradas', fontsize=18)

ax2[0].plot(t, senial, label='Señal original')
ax2[0].plot(t, senial_fir_1, label='Filtro Hamming, Orden 2001', color='g')
ax2[0].set_ylabel('Amplitud [mV]', fontsize=12)
ax2[0].set_xlabel('Tiempo [s]', fontsize=12)
ax2[0].legend(loc="upper right", fontsize=12)
ax2[0].set_title('Filtro FIR con ventana Hamming', fontsize=15)
ax2[0].set_xlim([0, ts*N])
#ax2[0].set_ylim([-1, 1.5])
ax2[0].grid()

ax2[1].plot(t, senial, label='Señal original')
ax2[1].plot(t, senial_fir_2, label='Filtro Hann, Orden 1001', color='r')
ax2[1].set_ylabel('Amplitud [mV]', fontsize=12)
ax2[1].set_xlabel('Tiempo [s]', fontsize=12)
ax2[1].legend(loc="upper right", fontsize=12)
ax2[1].set_title('Filtro FIR con ventana Hann', fontsize=15)
ax2[1].set_xlim([0, ts*N])
#ax2[1].set_ylim([-1, 1.5])
ax2[1].grid()

ax2[2].plot(t, senial, label='Señal original')
ax2[2].plot(t, senial_fir_3, label='Filtro Blackman, Orden 1001', color='b')
ax2[2].set_ylabel('Amplitud [mV]', fontsize=12)
ax2[2].set_xlabel('Tiempo [s]', fontsize=12)
ax2[2].legend(loc="upper right", fontsize=12)
ax2[2].set_title('Filtro FIR con ventana Blackman', fontsize=15)
ax2[2].set_xlim([0, ts*N])
#ax2[2].set_ylim([-1, 1.5])
ax2[2].grid()

plt.show()



# Grafica del Espectro de la señal original
freq = fft.fftfreq(N, d=1/fs)   # se genera el vector de frecuencias
senial_fft_original = fft.fft(senial)    # se calcula la transformada rápida de Fourier
senial_fft_mod_original = np.abs(senial_fft_original) / N

# Espectro de frecuencia de la señal del Filtro 1
freq_fir_1 = fft.fftfreq(len(senial_fir_1), d=1/fs)   # se genera el vector de frecuencias
senial_fft_fir_1 = fft.fft(senial_fir_1)    # se calcula la transformada rápida de Fourier
senial_fft_mod_fir_1 = np.abs(senial_fft_fir_1) / len(senial_fir_1)

# Espectro de frecuencia de la señal del Filtro 2
freq_fir_2 = fft.fftfreq(len(senial_fir_2), d=1/fs)   # se genera el vector de frecuencias
senial_fft_fir_2 = fft.fft(senial_fir_2)    # se calcula la transformada rápida de Fourier
senial_fft_mod_fir_2 = np.abs(senial_fft_fir_2) / len(senial_fir_2)

# Espectro de frecuencia de la señal del Filtro 3
freq_fir_3 = fft.fftfreq(len(senial_fir_3), d=1/fs)   # se genera el vector de frecuencias
senial_fft_fir_3 = fft.fft(senial_fir_3)    # se calcula la transformada rápida de Fourier
senial_fft_mod_fir_3 = np.abs(senial_fft_fir_3) / len(senial_fir_3)



# El espectro es simétrico, nos quedamos solo con el semieje positivo
f = freq[np.where(freq >= 0)]
#senial_fft = senial_fft[np.where(freq >= 0)]

# Se calcula la magnitud del espectro
senial_fft_mod_original = np.abs(senial_fft_original) / N     
senial_fft_mod_fir_1 = np.abs(senial_fft_fir_1) / len(senial_fir_1)     
senial_fft_mod_fir_2 = np.abs(senial_fft_fir_2) / len(senial_fir_2)     
senial_fft_mod_fir_3 = np.abs(senial_fft_fir_3) / len(senial_fir_3)     

#Gráfica de los espectros de frecuencia
fig3, ax3 = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
fig3.suptitle('Espectros de frecuencia de señales originales y filtradas', fontsize=18)

#Grafica de la señal temporal y del espectro en frecuencias
# Se grafica la señal temporal
ax3[0].plot(t, senial_fir_1)
ax3[0].set_xlabel('Tiempo [s]', fontsize=15)
ax3[0].set_ylabel('Tensión [V]', fontsize=15)
ax3[0].set_title('Señal temporal Filtro Hamming, Orden 2001', fontsize=15)
ax3[0].set_xlim([0, ts*N])
ax3[0].grid()

# se grafica la magnitud de la respuesta en frecuencia
ax3[1].plot(freq_fir_1, senial_fft_mod_fir_1)
ax3[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3[1].set_ylabel('Magnitud [V]', fontsize=15)
ax3[1].set_title('Magnitud de la Respuesta en Frecuencia del Filtro Hamming', fontsize=15)
ax3[1].set_xlim([-5000, 5000])
ax3[1].grid()

#Gráfica de los espectros de frecuencia
fig4, ax4 = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
fig4.suptitle('Espectros de frecuencia de señales originales y filtradas', fontsize=18)

# Se grafica la señal temporal
ax4[0].plot(t, senial_fir_2)
ax4[0].set_xlabel('Tiempo [s]', fontsize=15)
ax4[0].set_ylabel('Tensión [V]', fontsize=15)
ax4[0].set_title('Señal temporal del Filtro Hann, Orden 1001', fontsize=15)
ax4[0].set_xlim([0, ts*N])
ax4[0].grid()

# se grafica la magnitud de la respuesta en frecuencia
ax4[1].plot(freq, senial_fft_mod_fir_2)
ax4[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[1].set_ylabel('Magnitud [V]', fontsize=15)
ax4[1].set_title('Magnitud de la Respuesta en Frecuencia del Filtro Hann', fontsize=15)
ax4[1].set_xlim([-5000, 5000])
ax4[1].grid()

#Gráfica de los espectros de frecuencia
fig5, ax5 = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
fig5.suptitle('Espectros de frecuencia de señales originales y filtradas', fontsize=18)

# Se grafica la señal temporal
ax5[0].plot(t, senial_fir_3)
ax5[0].set_xlabel('Tiempo [s]', fontsize=15)
ax5[0].set_ylabel('Tensión [V]', fontsize=15)
ax5[0].set_title('Señal temporal del Filtro Blackman, Orden 1001', fontsize=15)
ax5[0].set_xlim([0, ts*N])
ax5[0].grid()

# se grafica la magnitud de la respuesta en frecuencia
ax5[1].plot(freq, senial_fft_mod_fir_3)
ax5[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax5[1].set_ylabel('Magnitud [V]', fontsize=15)
ax5[1].set_title('Magnitud de la Respuesta en Frecuencia del Filtro Blackman', fontsize=15)
ax5[1].set_xlim([-5000, 5000])
ax5[1].grid()

plt.show()


