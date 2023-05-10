import os
import sys
import numpy as np
from scipy.fftpack import fft, fftshift, ifft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

from ctypes import *
from funciones.dwfconstants import *
import time


def adquirir_un_canal(cantidad_de_muestras, frecuencia_de_muestreo=25000, 
                     rango_canal_0=10,
                     generar_sinusoide=False):
    '''
    Adquiere la señal conectada al canal ch0 del osciloscopio.
    
    Opcionalmente genera una sinusoide por el primer canal del generador de señales por 
    si se quiere enviar esta señal al osciloscopio.
    
    Devuelve las muestras adquiridas como un array de numpy.
    '''
    
    # COMPLETAR
    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    elif sys.platform.startswith("darwin"):
        dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    else:
        dwf = cdll.LoadLibrary("libdwf.so")

    #declare ctype variables
    hdwf = c_int()
    sts = c_byte()
    hzAcq = c_double(frecuencia_de_muestreo)
    nSamples = cantidad_de_muestras
    rgdSamples_ch0 = (c_double*nSamples)()
    rgdSamples_ch1 = (c_double*nSamples)()
    
    cAvailable = c_int()
    cLost = c_int()
    cCorrupted = c_int()
    fLost = 0
    fCorrupted = 0

    
    #open device
    print("Opening first device")
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(str(szerr.value))
        print("failed to open device")
        quit()

    if generar_sinusoide:
        print("Generating sine wave...")
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_bool(True))
        dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
        dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(1000))
        dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(2))
        dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))

    #set up acquisition
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(rango_canal_0))
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(rango_canal_1))
    
    
    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length

    #wait at least 2 seconds for the offset to stabilize
    print('Adquisición dentro de 2s ...', end=None)
    time.sleep(1)
    print('1s ...', end=None)
    time.sleep(1)
    
    
    

    #print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))

    print('INICIO ! ')
    
    cSamples = 0

    while cSamples < nSamples:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed) :
            # Acquisition not yet started.
            continue

        dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))

        cSamples += cLost.value

        if cLost.value :
            fLost = 1
        if cCorrupted.value :
            fCorrupted = 1

        if cAvailable.value==0 :
            continue

        if cSamples+cAvailable.value > nSamples :
            cAvailable = c_int(nSamples-cSamples)

        dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(rgdSamples_ch0, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data
        dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples_ch1, sizeof(c_double)*cSamples), cAvailable) # get channel 2 data
        cSamples += cAvailable.value

    if generar_sinusoide:
        dwf.FDwfAnalogOutReset(hdwf, c_int(0))
    
    dwf.FDwfDeviceCloseAll()

    print("Grabación finalizada")
    if fLost:
        print("AVISO: Pérdida de muestras! Puede deberse a una frecuencia de muestreo excesiva")
    if fCorrupted:
        print("AVISO: Corrupción de muestras! Puede deberse a una frecuencia de muestreo excesiva")
    
    muestras_ch0 = np.fromiter(rgdSamples_ch0, dtype = np.float)
    muestras_ch1 = np.fromiter(rgdSamples_ch1, dtype = np.float)
        
    return muestras_ch0


def adquirir_dos_canales_con_cuenta_regresiva_v1(cantidad_de_muestras, frecuencia_de_muestreo=25000,
                                             rango_canal_0=10, rango_canal_1=10,
                                             generar_sinusoide=False,
                                             amplitud_sinusoide=1,
                                             salida_digital_ajustable=False,
                                             salida_digital_0=0,
                                             salida_digital_1=0,
                                             voltaje_saturacion=4.5):
    '''
    Adquiere las señales conectadas a los canales ch0 y ch1 del osciloscopio.
    
    Opcionalmente genera una sinusoide por el primer canal del generador de señales por 
    si se quiere enviar esta señal al osciloscopio.
    
    Devuelve las muestras adquiridas como dos arrays de numpy.
    '''
    
    # COMPLETAR
    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    elif sys.platform.startswith("darwin"):
        dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    else:
        dwf = cdll.LoadLibrary("libdwf.so")

    global cAvailable, cLost, cCorrupted, fLost, fCorrupted

    #declare ctype variables
    hdwf = c_int()
    sts = c_byte()
    hzAcq = c_double(frecuencia_de_muestreo)
    nSamples = cantidad_de_muestras
    rgdSamples_ch0 = (c_double*nSamples)()
    rgdSamples_ch1 = (c_double*nSamples)()
    
    cAvailable = c_int()
    cLost = c_int()
    cCorrupted = c_int()
    fLost = 0
    fCorrupted = 0

    
    #open device----------------------------------------------------------
    print('---------------------------------------------------------------')
    print("Abriendo el dispositivo AD2")
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(str(szerr.value))
        print("ERROR: No se pudo abrir el dispositivo")
        quit()

    # POWER Vss, Vdd -----------------------------------------------------
    print('Prendiendo fuentes +5,-5')
    # set up analog IO channel nodes
    # enable positive supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(True)) 
    # set voltage to 5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(5.0)) 
    # enable negative supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(0), c_double(True)) 
    # set voltage to -5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(1), c_double(-5.0)) 
    # master enable
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))


    # GENERADOR ---------------------------------------------------------
    if generar_sinusoide:
        print("Generando sinusoide de amplitud {} V".format(amplitud_sinusoide))
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_bool(True))
        dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
        dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(50))
        dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(amplitud_sinusoide))
        dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))

    
    # DIGITAL IO --------------------------------------------------------
    dwRead = c_uint32()  # para leer las entradas digitales
    #chequear valores
    salida_digital_0 = 0 if salida_digital_0==0 else 1
    salida_digital_1 = 0 if salida_digital_1 == 0 else 1
    if salida_digital_ajustable:
        print('Salidas digitales se van a ajustar con la señal del canal 0')
        salida_digital_0 = 0
        salida_digital_1 = 0
    else:
        print('Salidas digitales fijas')
    mascara_salidas_digitales = salida_digital_1*2 + salida_digital_0

    # enable output/mask on 8 LSB IO pins, from DIO 0 to 7
    dwf.FDwfDigitalIOOutputEnableSet(hdwf, c_int(0x00FF)) 
    # set value on enabled IO pins
    dwf.FDwfDigitalIOOutputSet(hdwf, c_int(mascara_salidas_digitales))
    # fetch digital IO information from the device 
    dwf.FDwfDigitalIOStatus (hdwf) 
    # read state of all pins, regardless of output enable
    dwf.FDwfDigitalIOInputStatus(hdwf, byref(dwRead)) 

    #print(dwRead as bitfield (32 digits, removing 0b at the front)
    print('Salidas digitales sin ajustar [D1 D0] = [{}]'.format(bin(dwRead.value)[2:].zfill(16)[14:]))
    
    # PREPARACION DE LA ADQUISICION -------------------------------------
    #set up acquisition
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(rango_canal_0))
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(rango_canal_1))

    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length

    def adquirir_muestras(num_muestras):
        global cAvailable, cLost, cCorrupted, fLost, fCorrupted
        cSamples = 0
        while cSamples < num_muestras:
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
            if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed):
                # Acquisition not yet started.
                continue
            dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
            cSamples += cLost.value
            if cLost.value:
                fLost = 1
            if cCorrupted.value:
                fCorrupted = 1
            if cAvailable.value == 0:
                continue
            if cSamples + cAvailable.value > num_muestras:
                cAvailable = c_int(num_muestras - cSamples)

            dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(rgdSamples_ch0, sizeof(c_double) * cSamples),
                                       cAvailable)  # get channel 1 data
            dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples_ch1, sizeof(c_double) * cSamples),
                                       cAvailable)  # get channel 2 data
            cSamples += cAvailable.value

        if fLost:
            print("AVISO: Pérdida de muestras! Puede deberse a una frecuencia de muestreo excesiva")
        if fCorrupted:
            print("AVISO: Corrupción de muestras! Puede deberse a una frecuencia de muestreo excesiva")
        muestras_ch0 = np.fromiter(rgdSamples_ch0, dtype=np.float)[:num_muestras]
        muestras_ch1 = np.fromiter(rgdSamples_ch1, dtype=np.float)[:num_muestras]
        return(muestras_ch0, muestras_ch1)




    # SONDEO PARA DETERMINAR LA GANANCIA ADECUADA--------------------------
    if salida_digital_ajustable:
        print('---------------------------------------------------------------')
        print('Calibrando la ganancia adecuada ...')
        time.sleep(1)
        dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))  #inicio de la adquisici'on
        muestras_ch0, muestras_ch1 = adquirir_muestras(frecuencia_de_muestreo * 1 ) # adquisici'on de 1 segundo
        detected_amplitude = np.percentile(muestras_ch0, 99.9)
        orden_ganancia = int(np.log10(voltaje_saturacion/detected_amplitude))
        if orden_ganancia >=3:
            salida_digital_0 = 0
            salida_digital_1 = 1
        elif orden_ganancia >=2:
            salida_digital_0 = 1
            salida_digital_1 = 0
        else:
            salida_digital_0 = 0
            salida_digital_1 = 0
        mascara_salidas_digitales = salida_digital_1 * 2 + salida_digital_0
        # setear las nuevas salidas digitales
        dwf.FDwfDigitalIOOutputSet(hdwf, c_int(mascara_salidas_digitales))
        # fetch digital IO information from the device
        dwf.FDwfDigitalIOStatus(hdwf)
        # read state of all pins, regardless of output enable
        dwf.FDwfDigitalIOInputStatus(hdwf, byref(dwRead))

        print('Amplitud detectada: {:.3f} V'.format(detected_amplitude))
        print('Orden de ganancia determinada: ~ {}'.format( 10 ** int(mascara_salidas_digitales+1)))
        print('Salidas digitales ajustadas   [D1 D0] = [{}]'.format( bin(dwRead.value)[2:].zfill(16)[14:] ) )


    #wait at least 2 seconds for the offset to stabilize
    # print('Adquisición dentro de 3s ... ')
    # time.sleep(1)
    print('---------------------------------------------------------------')
    print('Iniciando adquisición en 2s ...  ')
    time.sleep(1)
    print('Iniciando adquisición en 1s ...  ')
    time.sleep(1)
    
    #print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
    print('INICIO !', end=None)

    muestras_ch0, muestras_ch1 = adquirir_muestras(nSamples)


    if generar_sinusoide:
        dwf.FDwfAnalogOutReset(hdwf, c_int(0))
    
    dwf.FDwfDeviceCloseAll()

    print("Grabación finalizada")


    detected_amplitude = np.percentile(muestras_ch0, 99.9)
    if detected_amplitude>voltaje_saturacion:
        print('ERROR: Ganancia excesiva. Señal del canal 0 saturada !!!')

    return muestras_ch0, muestras_ch1, mascara_salidas_digitales


def adquirir_dos_canales_con_cuenta_regresiva_v2(cantidad_de_muestras=50000, frecuencia_de_muestreo=25000,
                                              rango_canal_0=10, rango_canal_1=10,
                                              generar_sinusoide=False,
                                              amplitud_sinusoide=1,
                                              salida_digital_ajustable=False,
                                              salida_digital_0=0,
                                              salida_digital_1=0,
                                              voltaje_saturacion=4.5):
    '''
    Adquiere las señales conectadas a los canales ch0 y ch1 del osciloscopio.

    Opcionalmente genera una sinusoide por el primer canal del generador de señales por
    si se quiere enviar esta señal al osciloscopio.

    Devuelve las muestras adquiridas como dos arrays de numpy.
    '''

    # COMPLETAR
    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    elif sys.platform.startswith("darwin"):
        dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    else:
        dwf = cdll.LoadLibrary("libdwf.so")

    global cAvailable, cLost, cCorrupted, fLost, fCorrupted

    # declare ctype variables
    hdwf = c_int()
    sts = c_byte()
    hzAcq = c_double(frecuencia_de_muestreo)
    nSamples = cantidad_de_muestras
    rgdSamples_ch0 = (c_double * nSamples)()
    rgdSamples_ch1 = (c_double * nSamples)()

    cAvailable = c_int()
    cLost = c_int()
    cCorrupted = c_int()
    fLost = 0
    fCorrupted = 0

    # open device----------------------------------------------------------
    print('---------------------------------------------------------------')
    print("Abriendo el dispositivo AD2")
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(str(szerr.value))
        print("ERROR: No se pudo abrir el dispositivo")
        quit()

    # POWER Vss, Vdd -----------------------------------------------------
    print('Prendiendo fuentes +5,-5')
    # set up analog IO channel nodes
    # enable positive supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(True))
    # set voltage to 5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(5.0))
    # enable negative supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(0), c_double(True))
    # set voltage to -5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(1), c_double(-5.0))
    # master enable
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))

    # GENERADOR ---------------------------------------------------------
    if generar_sinusoide:
        print("Generando sinusoide de amplitud {} V".format(amplitud_sinusoide))
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_bool(True))
        dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
        dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(50))
        dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(amplitud_sinusoide))
        dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))

    # DIGITAL IO --------------------------------------------------------
    dwRead = c_uint32()  # para leer las entradas digitales
    # chequear valores
    salida_digital_0 = 0 if salida_digital_0 == 0 else 1
    salida_digital_1 = 0 if salida_digital_1 == 0 else 1
    if salida_digital_ajustable:
        print('Salidas digitales se van a ajustar con la señal del canal 0')
        salida_digital_0 = 0
        salida_digital_1 = 0
    else:
        print('Salidas digitales fijas')
    mascara_salidas_digitales = salida_digital_1 * 2 + salida_digital_0

    # enable output/mask on 8 LSB IO pins, from DIO 0 to 7
    dwf.FDwfDigitalIOOutputEnableSet(hdwf, c_int(0x00FF))
    # set value on enabled IO pins
    dwf.FDwfDigitalIOOutputSet(hdwf, c_int(mascara_salidas_digitales))
    # fetch digital IO information from the device
    dwf.FDwfDigitalIOStatus(hdwf)
    # read state of all pins, regardless of output enable
    dwf.FDwfDigitalIOInputStatus(hdwf, byref(dwRead))

    # print(dwRead as bitfield (32 digits, removing 0b at the front)
    print('Salidas digitales sin ajustar [D1 D0] = [{}]'.format(bin(dwRead.value)[2:].zfill(16)[14:]))

    # PREPARACION DE LA ADQUISICION -------------------------------------
    # set up acquisition
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(rango_canal_0))
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(rango_canal_1))

    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples / hzAcq.value))  # -1 infinite record length

    def adquirir_muestras(num_muestras):
        global cAvailable, cLost, cCorrupted, fLost, fCorrupted
        cSamples = 0
        while cSamples < num_muestras:
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
            if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed):
                # Acquisition not yet started.
                continue
            dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
            cSamples += cLost.value
            if cLost.value:
                fLost = 1
            if cCorrupted.value:
                fCorrupted = 1
            if cAvailable.value == 0:
                continue
            if cSamples + cAvailable.value > num_muestras:
                cAvailable = c_int(num_muestras - cSamples)

            dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(rgdSamples_ch0, sizeof(c_double) * cSamples),
                                       cAvailable)  # get channel 1 data
            dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples_ch1, sizeof(c_double) * cSamples),
                                       cAvailable)  # get channel 2 data
            cSamples += cAvailable.value

        if fLost:
            print("Pérdida de muestras! Reduzca la frecencia")
        if fCorrupted:
            print("Corrupción de muestras! Reduzca la frecencia")
        muestras_ch0 = np.fromiter(rgdSamples_ch0, dtype=np.float)[:num_muestras]
        muestras_ch1 = np.fromiter(rgdSamples_ch1, dtype=np.float)[:num_muestras]
        return (muestras_ch0, muestras_ch1)

    # SONDEO PARA DETERMINAR LA GANANCIA ADECUADA--------------------------
    if salida_digital_ajustable:
        print('---------------------------------------------------------------')
        print('Calibrando la ganancia adecuada ...')



        for mascara_salidas_digitales in [2,1,0]:
            # setear las nuevas salidas digitales
            dwf.FDwfDigitalIOOutputSet(hdwf, c_int(mascara_salidas_digitales))
            time.sleep(1)
            dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))  # inicio de la adquisici'on
            muestras_ch0, muestras_ch1 = adquirir_muestras(int(frecuencia_de_muestreo*0.5))  # adquisici'on de 0.5 segundo
            detected_amplitude = np.percentile(muestras_ch0, 99.9)
            if detected_amplitude < voltaje_saturacion:
                #encontramos la ganancia adecuada ya que no está saturada
                break;
        # fetch digital IO information from the device
        dwf.FDwfDigitalIOStatus(hdwf)
        # read state of all pins, regardless of output enable
        dwf.FDwfDigitalIOInputStatus(hdwf, byref(dwRead))

        print('Orden de ganancia determinada: ~ {}'.format( 10 ** int(mascara_salidas_digitales+1)))
        print('Salidas digitales ajustadas   [D1 D0] = [{}]'.format( bin(dwRead.value)[2:].zfill(16)[14:] ) )

    # wait at least 2 seconds for the offset to stabilize
    # print('Adquisición dentro de 3s ... ')
    # time.sleep(1)
    print('---------------------------------------------------------------')
    print('Iniciando adquisición en 2s ...  ')
    time.sleep(1)
    print('Iniciando adquisición en 1s ...  ')
    time.sleep(1)

    # print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
    print('INICIO !', end=None)
    muestras_ch0, muestras_ch1 = adquirir_muestras(nSamples)

    if generar_sinusoide:
        dwf.FDwfAnalogOutReset(hdwf, c_int(0))

    dwf.FDwfDeviceCloseAll()

    print("Grabación finalizada")

    detected_amplitude = np.percentile(muestras_ch0, 99.9)
    if detected_amplitude > voltaje_saturacion:
        print('ERROR: Ganancia excesiva. Señal del canal 0 saturada !!!')

    return muestras_ch0, muestras_ch1, mascara_salidas_digitales

# solo para test
if __name__ == '__main__':
    cantidad_de_muestras = 50000
    frecuencia_de_muestreo = 25000
    salida_digital_ajustable = True
    amplitud_sinusoide = 0.1
    D0=1
    D1=1
    muestras_ch0, muestras_ch1, mascara_salidas_digitales = adquirir_dos_canales_con_cuenta_regresiva_v2(cantidad_de_muestras,
                                                                           frecuencia_de_muestreo=frecuencia_de_muestreo,
                                                                          rango_canal_0=10, rango_canal_1=10,
                                                                          generar_sinusoide=True,
                                                                          amplitud_sinusoide=amplitud_sinusoide,
                                                                          salida_digital_ajustable=salida_digital_ajustable,
                                                                          salida_digital_0=D0,
                                                                          salida_digital_1=D1,
                                                                          voltaje_saturacion=4.5)

    print('===========================================================================')
    print('Mascara de salidas digitales final: {}'.format(mascara_salidas_digitales))
    print('Salidas digitales [D1 D0] = [{} {}]'.format(mascara_salidas_digitales//2, np.mod(mascara_salidas_digitales,2)))

    # graficar
    plt.figure(figsize=(8, 2))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(muestras_ch0, 'b', label='ch0')
    plt.plot(muestras_ch1, 'r', label='ch1')
    plt.legend()
    ax1.title.set_text('Toda la adq.')

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(muestras_ch0, 'b.', label='ch0')
    plt.plot(muestras_ch1, 'r.', label='ch1')
    ax2.title.set_text('Ultimos 4 ciclos')
    plt.xlim(cantidad_de_muestras - frecuencia_de_muestreo / 50 * 4, cantidad_de_muestras)

    plt.show()

