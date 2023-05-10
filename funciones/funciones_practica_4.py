import os
import numpy as np
from scipy.fftpack import fft, fftshift, ifft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from datetime import datetime

from funciones.util import plot_confusion_matrix
from funciones.funciones_practica_1 import *

# importar funciones del módulo de aprendizaje sklearn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as ABC

def clasificar(clasificador, caracteristicas, etiquetas, subconjuntos ):
    '''
    Recibe un clasificador ya creado, las caracteristicas, 
    las etiquetas y los indicadores de subconjuntos. 
    Devuelve las tasas de acierto y las predicciones
    '''
    
    # Si se usa una sóla característica, forzar que sea un vector columna
    if caracteristicas.ndim == 1:
        caracteristicas = caracteristicas[:,np.newaxis]
    
    # cantidad de subconjuntos
    cantidad_subconjuntos = len(np.unique(subconjuntos))
    
    # inicializar arrays para guardar los resultados
    accuracies = np.empty((cantidad_subconjuntos))
    y_predictions = np.empty((caracteristicas.shape[0]), dtype=np.uint8)

    start = datetime.now()
    #para cada subconjunto
    for i in range(cantidad_subconjuntos):
        id_subconjunto = np.unique(subconjuntos)[i]
        print('%d/%d fold...\t tiempo: %ds'%(id_subconjunto,cantidad_subconjuntos,(datetime.now()-start).seconds), end='\r', flush=True)

        # separar los datos en entrenamiento y test
        indices_test = np.where(subconjuntos==id_subconjunto)[0]
        indices_train = np.where(subconjuntos!=id_subconjunto)[0]

        X_train = caracteristicas[indices_train,:]
        y_train = etiquetas[indices_train]

        X_test = caracteristicas[indices_test,:]
        y_test = etiquetas[indices_test]

        # entrenar el clasificador
        clf = clasificador # solo un nombre más corto
        clf.fit(X_train,y_train)

        #obtener las predicciones sobre el conjunto de test
        y_pred=clf.predict(X_test)
        # obtener la tasa de acierto
        acc = clf.score(X_test,y_test)

        # guardar predicciones y tasa de acierto para el fold
        accuracies[i]=acc
        y_predictions[indices_test] = y_pred
    
    return accuracies, y_predictions, clf


def mostrar_performance(accuracies, y_predictions, etiquetas):
    print('Acierto medio = {:.2f}'.format(np.mean(accuracies)*100))

    # Graficar non-normalized confusion matrix
    y_pred = y_predictions.astype(int)
    y_test = etiquetas.astype(int)

    plot_confusion_matrix(y_test, y_pred, 
                          classes=nombres_electrodomesticos,
                          title='Matriz de confusión')

# VI IMAGE
#
# Adaptado de:
# [1] Gao, Jingkun, et al. "Plaid: a public dataset of high-resoultion 
# electrical appliance measurements for load identification research: 
# demo abstract." proceedings of the 1st ACM Conference on Embedded 
# Systems for Energy-Efficient Buildings. ACM, 2014.
# 


def get_img_from_VI(V, I, width, hard_threshold=False, para=.5):
    '''Get images from VI, hard_threshold, set para as threshold to cut off,5-10
    soft_threshold, set para to .1-.5 to shrink the intensity'''
    
    d = V.shape[0]
    # doing interploation if number of points is less than width*2
    if d<2* width:
        assert False
        newV = np.hstack([V, V[0]])
        newI = np.hstack([I, I[0]])
        oldt = np.linspace(0,d,d+1)
        newt = np.linspace(0,d,2*width)
        I = np.interp(newt,oldt,newI)
        V = np.interp(newt,oldt,newV)
    # get the size resolution of mesh given width
    d_c = (np.amax(I) - np.amin(I)) / width
    d_v = (np.amax(V) - np.amin(V)) / width
    
    #  find the index where the VI goes through in current-voltage axis
    ind_c = np.floor((I-np.amin(I))/d_c).astype(int)
    ind_v = np.floor((V-np.amin(V))/d_v).astype(int)
    ind_c[ind_c==width] = width-1
    ind_v[ind_v==width] = width-1  # ok
    
    Img = np.zeros((width,width))
    
    for i in range(len(I)):
        Img[ind_c[i],width-ind_v[i]-1] += 1 # why backwards?
    
    if hard_threshold:
        Img[Img<para] = 0
        Img[Img!=0] = 1
        return Img
    else:
        return (Img/np.max(Img))**para        

def calcular_caracteristicas_PLAID():
    
    # Calcular las características dadas por la
    # función calcular_potencia_IEEE_1459_2010 
    ids = get_ids()

    # inicializar arrays
    TIPO = np.empty( (len(ids)), dtype=np.uint8)
    CASA = np.empty( (len(ids)), dtype=np.uint8)
    DATOS_IEEE = np.empty( (len(ids), 14))

    for i in range(len(ids)):
        id_elec = ids[i][0]          
        print('Calculando {:04d}/{:04d}'.format(id_elec, len(ids)), end='\r', flush=True)

        #COMPLETAR código
        I,V = cargar_VI_por_ciclos(get_nombre_archivo(id_elec), 
                                   ciclos_a_saltear=50,
                                   ciclos_a_cargar=60)

        S, S_11, S_H, S_N, P, P_11, P_H, Q_11, D_I, D_V, D_H, N, THD_V, THD_IR = \
                calcular_potencia_IEEE_1459_2010(I,V, frecuencia_muestreo=30000, frecuencia_linea=60)

        TIPO[i] = get_tipo(id_elec)
        CASA[i] = get_casa(id_elec)
        DATOS_IEEE[i,0] = S
        DATOS_IEEE[i,1] = S_11
        DATOS_IEEE[i,2] = S_H
        DATOS_IEEE[i,3] = S_N
        DATOS_IEEE[i,4] = P
        DATOS_IEEE[i,5] = P_11
        DATOS_IEEE[i,6] = P_H
        DATOS_IEEE[i,7] = Q_11
        DATOS_IEEE[i,8] = D_I
        DATOS_IEEE[i,9] = D_V
        DATOS_IEEE[i,10] = D_H
        DATOS_IEEE[i,11] = N
        DATOS_IEEE[i,12] = THD_V
        DATOS_IEEE[i,13] = THD_IR
        
    # Calcular las características FP_fun y FP_tot
    FP_fun = np.empty(len(ids))
    FP_tot = np.empty(len(ids))

    for i in range(len(ids)):
        id_elec = ids[i][0]          
        print('Calculando {:04d}/{:04d}'.format(id_elec, len(ids)), end='\r', flush=True)

        FP_fun[i] = DATOS_IEEE[i,5] / DATOS_IEEE[i,1] # FP_fun = P_11/S_11
        FP_tot[i] = DATOS_IEEE[i,4] / DATOS_IEEE[i,0] # FP_tot = P/S


    ancho_imagen_VI = 16
    hard_th = True

    # inicializar arrays
    IMG_VI = np.empty( (len(ids), ancho_imagen_VI**2))

    # COMPLETAR código

    for i in ids:
        nombre_de_archivo = get_nombre_archivo(i[0])
        I,V = cargar_VI_por_ciclos(nombre_de_archivo, ciclos_a_saltear=50)
        IMG_VI[i-1,:] = np.ravel(get_img_from_VI(V,I, width=ancho_imagen_VI, hard_threshold=hard_th))

        print('Calculando {:04d}/{:04d}'.format(int(i), len(ids)), end='\r', flush=True)

    factores = np.transpose(np.vstack((FP_fun, FP_tot)))
    todas_juntas = np.hstack((factores, DATOS_IEEE))
    caracteristicas = np.hstack((todas_juntas, IMG_VI))
    
    return TIPO, CASA, DATOS_IEEE, factores, IMG_VI, caracteristicas;

def calcular_caracteristicas_EDMIIE():
    
    # Calcular las características dadas por la
    # función calcular_potencia_IEEE_1459_2010 
    ids = get_ids(PLAID=False)

    # inicializar arrays
    TIPO = np.empty( (len(ids)), dtype=np.uint8)
    CASA = np.empty( (len(ids)), dtype=np.uint8)
    DATOS_IEEE = np.empty( (len(ids), 14))

    for i in range(len(ids)):
        id_elec = ids[i][0]          
        print('Calculando {:04d}/{:04d}'.format(id_elec, len(ids)), end='\r', flush=True)

        #COMPLETAR código
        I,V = cargar_VI_por_ciclos(get_nombre_archivo(id_electrodomestico=id_elec, PLAID=False), ciclos_a_saltear=10, ciclos_a_cargar=60, frecuencia_muestreo=25000, frecuencia_linea=50)

        S, S_11, S_H, S_N, P, P_11, P_H, Q_11, D_I, D_V, D_H, N, THD_V, THD_IR = \
                calcular_potencia_IEEE_1459_2010(I,V, frecuencia_muestreo=25000, frecuencia_linea=50)

        TIPO[i] = get_tipo(id_elec, PLAID=False)
        CASA[i] = get_casa(id_elec, PLAID=False)
        DATOS_IEEE[i,0] = S
        DATOS_IEEE[i,1] = S_11
        DATOS_IEEE[i,2] = S_H
        DATOS_IEEE[i,3] = S_N
        DATOS_IEEE[i,4] = P
        DATOS_IEEE[i,5] = P_11
        DATOS_IEEE[i,6] = P_H
        DATOS_IEEE[i,7] = Q_11
        DATOS_IEEE[i,8] = D_I
        DATOS_IEEE[i,9] = D_V
        DATOS_IEEE[i,10] = D_H
        DATOS_IEEE[i,11] = N
        DATOS_IEEE[i,12] = THD_V
        DATOS_IEEE[i,13] = THD_IR
        
    # Calcular las características FP_fun y FP_tot
    FP_fun = np.empty(len(ids))
    FP_tot = np.empty(len(ids))

    for i in range(len(ids)):
        id_elec = ids[i][0]          
        print('Calculando {:04d}/{:04d}'.format(id_elec, len(ids)), end='\r', flush=True)

        FP_fun[i] = DATOS_IEEE[i,5] / DATOS_IEEE[i,1] # FP_fun = P_11/S_11
        FP_tot[i] = DATOS_IEEE[i,4] / DATOS_IEEE[i,0] # FP_tot = P/S


    ancho_imagen_VI = 16
    hard_th = True

    # inicializar arrays
    IMG_VI = np.empty( (len(ids), ancho_imagen_VI**2))

    # COMPLETAR código

    for i in ids:
        nombre_de_archivo = get_nombre_archivo(id_electrodomestico=i[0], PLAID=False)
        I,V = cargar_VI_por_ciclos(nombre_de_archivo, ciclos_a_saltear=50, frecuencia_muestreo=25000, frecuencia_linea=50)
        IMG_VI[i-1,:] = np.ravel(get_img_from_VI(V,I, width=ancho_imagen_VI, hard_threshold=hard_th))

        print('Calculando {:04d}/{:04d}'.format(int(i), len(ids)), end='\r', flush=True)

    factores = np.transpose(np.vstack((FP_fun, FP_tot)))
    todas_juntas = np.hstack((factores, DATOS_IEEE))
    caracteristicas = np.hstack((todas_juntas, IMG_VI))
    
    return TIPO, CASA, DATOS_IEEE, factores, IMG_VI, caracteristicas;

