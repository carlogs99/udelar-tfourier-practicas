
import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftshift, ifft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

#estilo de las gráficas
plt.style.use('ggplot')



frecuencia_muestreo = 30000 #Frecuencia de muestreo en Hz
frecuencia_linea = 60    #Frecuencia de línea en Hz
muestras_por_ciclo = int(frecuencia_muestreo/frecuencia_linea)

nombres_electrodomesticos = ['Air Conditioner',
                         'Compact Fluorescent Lamp',
                         'Fan',
                         'Fridge',
                         'Hairdryer',
                         'Heater',
                         'Incandescent Light Bulb',
                         'Laptop',
                         'Microwave',
                         'Vacuum',
                         'Washing Machine',
                         'Other']

nombres_abreviados_electrodomesticos = ['AirC','CFL','Fan','Frid','Hair','Heat','ILB','Lapt','MWave','Vacc','Wash', 'Other']


# ubicación del directorio  de la base PLAID que contiene los ".csv"
PLAID_csv_directory = "C:/Users/carlos/Downloads/PLAID/CSV"  #PONER EL CAMINO ADECUADO

# archivo con la metadata
archivo_metadata = './data/meta1_simple.csv'

# ubicación del directorio de la base EDM-IIE que contiene las señales en archivos ".csv"
EDM_IIE_csv_directory = './EDM_IIE/CSV'

# archivo con la metadata completa de la base EDM-IIE
EDM_IIE_archivo_full_metadata = './EDM_IIE/meta1_simple_iie_2019.csv'

#--------------------------------------------------------

# COMPLETAR
# Agregar las funciones de la práctica 1

def cargar_metadata(PLAID=True):
    '''
    Carga la informacion del archivo "meta1_simple.csv" a la variable metadata
    '''
    # COMPLETADO
    ##metadata = np.genfromtxt(archivo_metadata, dtype="int", delimiter=",", skip_header=1) 
    if(PLAID):
        dataframe = pd.read_csv(archivo_metadata)
        metadata = dataframe.to_numpy()
    else:
        dataframe = pd.read_csv(EDM_IIE_archivo_full_metadata)
        metadata = dataframe.to_numpy()
    return metadata

#------------------------------------------------------------------------------
# Todas las funciones 'get' suponen la pre-existencia de la variable 'metadata'
#------------------------------------------------------------------------------

def get_cantidad_electrodomesticos(PLAID=True):
    '''Devuelve la cantidad de electrodomésticos en la base'''
    # COMPLETADO
    return len(cargar_metadata(PLAID))

def get_tipo(id_electrodomestico, PLAID=True):
    '''Devuelve el tipo de electrodoméstico'''
    # COMPLETADO
    if PLAID:
        auxDic = dict(cargar_metadata()[:,0:2])
        tipo = auxDic[id_electrodomestico]
    else:
        tipo = cargar_metadata(PLAID)[id_electrodomestico-1, 1]
    return tipo

def get_casa(id_electrodomestico, PLAID=True):
    '''Devuelve la casa del electrodoméstico'''
    # COMPLETADO
    if PLAID:
        auxDic = dict(cargar_metadata()[:,0:3:2])
        casa = auxDic[id_electrodomestico]
    else:
        casa = cargar_metadata(PLAID)[id_electrodomestico-1, 2]
    return casa
    
def get_nombre(id_electrodomestico, PLAID=True):
    '''Devuelve el nombre del electrodoméstico '''
    # COMPLETADO
    return nombres_electrodomesticos[get_tipo(id_electrodomestico, PLAID)]
    
def get_nombre_abreviado(id_electrodomestico):
    '''Devuelve el nombre abreviado del electrodoméstico '''
    # COMPLETADO
    return nombres_abreviados_electrodomesticos[get_tipo(id_electrodomestico)]

def get_nombre_archivo(id_electrodomestico, PLAID=True):
    '''Devuelve el camino completo al archivo correspondiente al electrodoméstico'''
    # COMPLETADO
    if PLAID:
        nombre = '{}/{}.csv'.format(PLAID_csv_directory, id_electrodomestico)
    else:
        casa_edm = cargar_metadata(PLAID)[id_electrodomestico-1, 2]
        id_edm = cargar_metadata(PLAID)[id_electrodomestico-1, 3]
        instancia_edm = cargar_metadata(PLAID)[id_electrodomestico-1, 4]
        nombre = '{}/casa_{:03d}_id_{:03d}_instancia_{:03d}.csv'.format(EDM_IIE_csv_directory, casa_edm, id_edm, instancia_edm)
    return nombre
    
def get_ids(PLAID=True):
    '''Devuelve un array con los ids de todos los electrodomésticos de la base'''
    # COMPLETADO
    return cargar_metadata(PLAID)[:,0:1]
    
def get_ids_por_tipo(tipo, PLAID=True):
    '''Devuelve los ids correspondientes a cierto tipo'''
    # COMPLETADO
    return [cargar_metadata(PLAID)[a,0] for a in range(len(cargar_metadata(PLAID))) if cargar_metadata(PLAID)[a,1]==tipo]

def get_ids_por_casa(casa, PLAID=True):
    '''Devuelve los ids correspondientes a cierta casa'''
    # COMPLETADO
    return [cargar_metadata(PLAID)[a,0] for a in range(len(cargar_metadata(PLAID))) if cargar_metadata(PLAID)[a,2]==casa]

def cargar_VI_por_ciclos(nombre_archivo, 
                         frecuencia_muestreo=30000,
                         frecuencia_linea=60,
                         ciclos_a_cargar=1e100,  #por defecto se cargan todos los ciclos posibles
                         ciclos_a_saltear=0):    #por defecto se carga desde el inicio
    '''
    Carga un cierto numero de ciclos de una señal I,V guardada en un archivo
    Devuelve las señales I y V como  2 arrays Nx1
    
    Importante: se debe asegurar que se carga un número entero de ciclos (el número final
    de ciclos cargados podría eventualmente ser menor a 'ciclos_a_cargar')
    '''
    
    # COMPLETADO
    I = np.genfromtxt(nombre_archivo, delimiter=',')[:, 0:1].flatten()
    V = np.genfromtxt(nombre_archivo, delimiter=',')[:, 1:2].flatten()
    
    muestras_por_ciclo = int(frecuencia_muestreo/frecuencia_linea)
    cantidad_ciclos_completos = int(len(I)/muestras_por_ciclo)
    
    muestra_inicial = int(ciclos_a_saltear*muestras_por_ciclo)
    muestra_final = min(muestra_inicial + int(ciclos_a_cargar*muestras_por_ciclo), cantidad_ciclos_completos*muestras_por_ciclo)
            
    I = I[muestra_inicial:muestra_final]
    V = V[muestra_inicial:muestra_final]    
    
    return I,V

def generar_vector_tiempo(numero_muestras, frecuencia_muestreo=30000):
    '''
    Genera un vector de tiempo del largo solicitado
    '''
    duracion = (numero_muestras-1)/frecuencia_muestreo   #duración de la señal en segundos
    vector_tiempo = np.linspace(0,duracion,numero_muestras)
    
    return vector_tiempo

def graficar_I_V(I, V, frecuencia_muestreo=30000, fignum=None, color = 'g'):
    '''
    Genera un vector de tiempos T adecuado y grafica por separado
    las señales de corriente I(T) y voltaje V(T) que se le pasan.
    Se supone que I y V son de igual largo
    
    Si se le pasa un fignum, grafica I sobre la figura (fignum) y V sobre la 
    figura (fignum+1). De lo contrario crea dos figuras nuevas.
    '''
    
    T = generar_vector_tiempo(len(I), frecuencia_muestreo)
    
    # COMPLETADO
    
    #Grafica corrientes
    plt.figure(fignum)
    plt.plot(T, I, color)
    plt.title("$I = f(t)$")
    plt.xlabel("Tiempo $t$")
    plt.ylabel("Corriente $I$")
    
    #Grafica voltajes
    if(fignum == None):
        plt.figure()
    else:
        plt.figure(fignum + 1)
    plt.plot(T, V, color)
    plt.title("$V = f(t)$")
    plt.xlabel("Tiempo $t$")
    plt.ylabel("Voltaje $V$")
    
def graficar_diagrama_VI(I, V, fignum=None):
    '''
    Grafica I vs. V 
    
    Si se le pasa un fignum, grafica el diagrama sobre la 
    figura (fignum). De lo contrario crea una figuras nueva.
    '''
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
        
    plt.plot(V,I,'.-')
    plt.title('Diagrama V-I')
    plt.xlabel('V')
    plt.ylabel('I')


def promediar_ciclos(S, frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Promedia los ciclos de la señal S. Si la señal no tiene un número entero de ciclos,
    las muestras del final correspondientes a un ciclo incompleto no se tienen en cuenta
    Devuelve el ciclo promedio.
    Entrada:
        S     array Nx1
    Salida
        ciclo      array Cx1    con C=frecuencia_muestreo/frecuencia_linea
    '''
    
    # COMPLETADO
    
    # Truncamos los ciclos incompletos de S
    C = int(frecuencia_muestreo/frecuencia_linea)
    cantidad_ciclos_completos = int(len(S) / C)
    muestras_a_considerar = cantidad_ciclos_completos * C 
    S_trunc = S[0:muestras_a_considerar]
    
    # Promediamos
    
    ciclo_promedio = []
    
    for x in range(C):
        promedioAux = []
        for n in range(cantidad_ciclos_completos):
            promedioAux.append(S_trunc[x+n*C])
        ciclo_promedio.append(np.mean(promedioAux))
        
    return ciclo_promedio
    
def calcular_indices_para_alinear_ciclo(ciclo):
    '''
    Alinea un ciclo de manera que la señal inicie en el cruce por cero ascendente.
    Devuelve los indices que hacen el ordenamiento
    Ejemplo de uso:
    indices = calcular_indices_para_alinear_ciclo(ciclo)
    ciclo_alineado = ciclo[indices]
    '''
    cantidad_muestras = len(ciclo)
    
    ix = np.argsort(np.abs(ciclo))
    j = 0
    while True:
        if ix[j]<muestras_por_ciclo-1 and ciclo[ix[j]+1]>ciclo[ix[j]]:
            real_ix = ix[j]
            break
        else:
            j += 1
    
    indices_ordenados = np.hstack( (np.arange(real_ix,cantidad_muestras),
                                    np.arange(0,real_ix)) )
    return indices_ordenados

def alinear_ciclo_I_V(ciclo_I, ciclo_V):
    '''
    Devuelve los ciclos I y V alineados tal que la señal 
    de voltaje inicie en el cruce por cero ascendente
    '''
    # COMPLETADO 
    
    I_i_alineado = calcular_indices_para_alinear_ciclo(ciclo_I)
    ciclo_I_alineado=[]
    for x in range(len(ciclo_I)):
        ciclo_I_alineado.append(ciclo_I[I_i_alineado[x]])
        
        
    V_i_alineado = calcular_indices_para_alinear_ciclo(ciclo_V)
    ciclo_V_alineado=[]
    for x in range(len(ciclo_V)):
        ciclo_V_alineado.append(ciclo_V[V_i_alineado[x]])    
    
    '''
    ciclo_I_alineado = [ciclo_I[x] for x in calcular_indices_para_alinear_ciclo(ciclo_I)]
    ciclo_V_alineado = [ciclo_V[x] for x in calcular_indices_para_alinear_ciclo(ciclo_V)]
    '''
    
    return ciclo_I_alineado, ciclo_V_alineado

def get_ciclo_I_V_promedio_alineado(I,V,frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Dadas las señales I y V, calcula los ciclos promedio y los alinea
    Devuelve los ciclos alineados ciclo_I_alineado y ciclo_v_alineado
    '''
    #COMPLETADO
    
    I_promedio = promediar_ciclos(I, frecuencia_muestreo, frecuencia_linea)
    V_promedio = promediar_ciclos(V, frecuencia_muestreo, frecuencia_linea)
        
    return alinear_ciclo_I_V(I_promedio, V_promedio)
    
def generar_vector_frecuencia(numero_muestras, frecuencia_muestreo=30000, centrar_frecuencia=True):
    '''
    Genera un vector de frecuencias del largo especificado
    If centrar_frecuencia==True (por defecto)
        salida es un array   [-Fm/2.....Fm/2-Fm/numero_muestras]
    else
        salida es un array   [0.....Fm-Fm/numero_muestras]
    '''
    step_frecuencia = frecuencia_muestreo/numero_muestras 
    vector_frecuencia = np.arange(0,frecuencia_muestreo,step_frecuencia)
    if centrar_frecuencia:
        vector_frecuencia = vector_frecuencia - frecuencia_muestreo/2
    
    return vector_frecuencia

def graficar_FI_FV(I,V, frecuencia_muestreo=30000, centrar_frecuencia=True, fignum=None):
    '''
    Genera un vector de frecuencia adecuado.
    Grafica el modulo de la transformada de I y de V en figuras separadas
    
    Si se le pasa un fignum, grafica FI sobre la figura (fignum) y FV sobre la 
    figura (fignum+1). De lo contrario crea dos figuras nuevas.
    '''
    numero_muestras = len(I)
    vector_frecuencia = generar_vector_frecuencia(numero_muestras, frecuencia_muestreo, centrar_frecuencia)
    
    # COMPLETADO
    
    #Grafica corrientes
    plt.figure(fignum)
    if centrar_frecuencia:
        plt.plot(vector_frecuencia, np.abs(fftshift(fft(I))), 'r')
    else:
        plt.plot(vector_frecuencia, np.abs(fft(I)), 'r')
        
    plt.title("$abs(fft(I)) = f(f)$")
    plt.xlabel("Frecuencia $f$")
    plt.ylabel("$fft(I)$")
    
    #Grafica voltajes
    if(fignum == None):
        plt.figure()
    else:
        plt.figure(fignum + 1)
        
    if centrar_frecuencia:
        plt.plot(vector_frecuencia, np.abs(fftshift(fft(V))), 'r')
    else:
        plt.plot(vector_frecuencia, np.abs(fft(V)), 'r')
        
    plt.title("$abs(fft(V)) = f(f)$")
    plt.xlabel("Frecuencia $f$")
    plt.ylabel("$fft(V)$")  

def graficar_espectrograma_I_V(I,V, frecuencia_muestreo=30000, largo_ventana=256, fignum=None):
    '''
    Grafica el espectrograma de I y de V en figuras separadas
    
    Si se le pasa un fignum, grafica el espectrograma de I sobre la figura (fignum)
    y el de V sobre la figura (fignum+1). De lo contrario crea dos figuras nuevas.
    '''
    f,t,SI = spectrogram(I,fs=frecuencia_muestreo, nperseg=largo_ventana)
    f,t,SV = spectrogram(V,fs=frecuencia_muestreo, nperseg=largo_ventana)
    
    
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
    
    plt.pcolormesh(t, f, SI)
    plt.title('Espectrograma de I')
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
    
    plt.pcolormesh(t, f, SV)
    plt.title('Espectrograma de V')
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    
# Calcular el factor de distorsión para una señal S
def calcular_THD(S, frecuencia_muestreo=30000, frecuencia_linea=60):
    
    numero_muestras = len(S)
    FS = fft(S)
        
    #np.sqrt(2) para que sea rms,  el dos viene porque tenemos frecuencias positivas y negativas 
    #luego de esto solo vamos a usar las frecuencias positivas
    FS_rms = np.abs(FS)/numero_muestras  * 2/np.sqrt(2)  
    
    step_frecuencia = frecuencia_muestreo/numero_muestras 
    indice_frecuencia_fundamental = int(frecuencia_linea / step_frecuencia)
    
    indices_armonicos = np.arange(2*indice_frecuencia_fundamental,
                                  numero_muestras//2,
                                  indice_frecuencia_fundamental) 
    
    distorsion = np.sqrt(np.sum(FS_rms[indices_armonicos]**2)) / FS_rms[indice_frecuencia_fundamental]
    return distorsion
    

def calcular_potencia_media(I,V, frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Calcula la potencia media
    '''
    # COMPLETADO
    potencia_media = np.sum(I*V)/len(I)
    return potencia_media
       
def calcular_potencia_IEEE_1459_2010(I,V, frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Calcula la potencia para señales I,V que pueden ser no-sinusoidales.
    Se supone que las señales I y V tienen un número entero de períodos.
    
    Los cálculos se realizan en frecuencia. Para esta implementación se consideran únicamente la frecuencia
    fundamental y sus armónicos. No se tienen en cuenta otras componentes de frecuencia intermedias.
    
    La función devuelve: S, S_11, S_H, S_N, P, P_11, P_H, Q_11, D_I, D_V, D_H, N, THD_V, THD_I
    S        Apparent power
    S_11     Fundamental apparent power
    S_H      Harmonic apparent power
    S_N      Non-fundamental apparent power
    P        Active power
    P_11     Fundamental active power
    P_H      Harmonics active power
    Q_11     Fundamental reactive power
    D_I      Current distortion power
    D_V      Voltage distortion power
    D_H      Harmonic distortion power
    N        Non-active apparent power
    THD_V    Total harmonic distortion for voltage
    THD_I    Total harmonic distortion for current
    '''
    
    # COMPLETADO
    
    numero_muestras = len(I)
    FI = fft(I)
    FV = fft(V)
    FI_rms = np.abs(FI)/numero_muestras  * 2/np.sqrt(2) 
    FV_rms = np.abs(FV)/numero_muestras  * 2/np.sqrt(2)
    
    step_frecuencia = frecuencia_muestreo/numero_muestras 
    indice_frecuencia_fundamental = int(frecuencia_linea / step_frecuencia)
    indices_armonicos = np.arange(2*indice_frecuencia_fundamental,
                                  numero_muestras//2,
                                  indice_frecuencia_fundamental) 
   
    theta1 = np.angle(FV[indice_frecuencia_fundamental])-np.angle(FI[indice_frecuencia_fundamental])

    #S_11:
    S_11 = FI_rms[indice_frecuencia_fundamental]*FV_rms[indice_frecuencia_fundamental]
    
    #THD_V:
    THD_V = calcular_THD(V, frecuencia_muestreo=frecuencia_muestreo, frecuencia_linea=frecuencia_linea)
    
    #THD_I:
    THD_I = calcular_THD(I, frecuencia_muestreo=frecuencia_muestreo, frecuencia_linea=frecuencia_linea)
    
    #S_H:
    S_H = S_11*THD_V*THD_I
    
    #P:
    P = calcular_potencia_media(I,V,frecuencia_muestreo=frecuencia_muestreo, frecuencia_linea=frecuencia_linea)
    
    #P_11:
    P_11 = S_11*np.cos(theta1)
    
    #P_H:
    P_H = P - P_11
    
    #Q_11:
    Q_11 = S_11*np.sin(theta1)
    
    #D_I:
    D_I = S_11*THD_I
    
    #D_V:
    D_V = S_11*THD_V
    
    #D_H:
    D_H = np.sqrt(abs(S_H**2-P_H**2))
    
    #S_N:
    S_N = np.sqrt(D_I**2+D_V**2+S_H**2)
    
    #S:
    S = np.sqrt(S_11**2+S_N**2)
    
    #N:
    N = np.sqrt(abs(S**2-P**2))
    
    return S, S_11, S_H, S_N, P, P_11, P_H, Q_11, D_I, D_V, D_H, N, THD_V, THD_I



