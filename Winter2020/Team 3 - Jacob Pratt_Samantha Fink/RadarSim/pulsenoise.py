import numpy as np


def noiseBandwidth(BW, NF=1):
    kB = 1e-23
    T = 291
    BW = 1/PW
    NoisePWR = kB*T*BW*NF
    Noise_dBm = 10*np.log10(NoisePWR/1e-3)
    return NoisePWR, Noise_dBm

def addNoiseToPulseTrain(p_train, BW, NF=1):
    # add the noise in.
    kB = 1e-23
    T = 291
    #BW = 1/PW
    NoisePWR = kB*T*BW*NF
    noise = awgn_like(p_train, pwr = NoisePWR)
    signal_noise = noise + p_train
    return signal_noise

def awgn_like(signal, pwr = 1e-3):
    '''
    creates a noise vector like signal at a particular power level
    '''
    noisepower=pwr
    noise=noisepower*(np.random.uniform(-1,1,size=len(signal)))
    return noise
