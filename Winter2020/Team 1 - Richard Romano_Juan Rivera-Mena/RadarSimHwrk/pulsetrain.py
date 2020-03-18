import numpy as np
import scipy

def createPulseTrain(A=1, fc=2, k=1, PRI=2, dt_k=20, K_pulses = 10):
    '''
    A:     amplitude
    fc:    center frequency
    k:     number of periods/wavelengths per Pulse Width
    PRI:   Time between pulses
    dt_k:  Samples per period of fc (1/fc)/dt_k
    K_pulses: how many unit pulses are created (N-observations)

    return signal, PW, dt


    NO ERROR CHECKING!!!!
    Creates a simple RADAR/SONAR pulse train used for homework
    '''
    print('=*'*40)
    print('Amplitude {:.02f} dBm, fc:{}, k:{}, PRI:{}, dt_k:{}, K_pulses:{} '.format((10*np.log10(A/1e-3)), fc, k, PRI, dt_k, K_pulses))

    dt = (1/fc)/dt_k # 1/fc is period of center carrier, 
    # this is the center carrier divided by numbers of samples to give us sample length in time
    t_vector = np.arange(0,PRI,dt) # One Gates worth of samples
    len_PRI = len(t_vector)
    PW = (1/fc)*k
    PWidx = np.int(PW/dt)
    mask_v = np.zeros_like(t_vector)
    mask_v[:PWidx] = 1
    mask_v[0]=0 # makes the pulse nice.

    # Multiple inside sin function Ramp(t)
    freq_v = A*np.cos(2. * np.pi*np.int(fc)*t_vector*scipy.signal.sawtooth(t_vector)) 
    unit_signal = mask_v*freq_v

    signal = None # clear the signal
    signal = unit_signal
    idx = 0
    #print(idx,K_pulses)
    for idx in range(int(K_pulses)):
        #print(idx,K_pulses)
        signal = np.concatenate((signal, unit_signal), axis=None)
    return signal, PW, dt, len_PRI


def timeShift(p_train, Range,vp, dt, len_PRI):
    sample_shift = np.int(2*Range/vp/dt)
    print(sample_shift)
    p_train = np.roll(p_train, sample_shift)
    return p_train


def RadarEquationAdv(Pt, Gt, Range, RCS, Gr, Lambda, dB=False):
    '''
    Uses radar equation to calculate signal received in Watts and dB
    input: Pt (Watts)
            Gt (Scalar) if dB = True converts to scalar
            Range (m)
            RCS (m^2)
            Gr (Scalar) if dB = True converts to a scalar
            Lambda(meters/cycle)
            dB True/False Tells if Gt,Gr are passed as Scalars or dB
    return: watts,dBm
    '''
    if dB == True: # convert to scalar
        Gt = 10*np.log10(Gt/10)
        Gr = 10*np.log10(Gr/10)

    PowerRadiated = Pt*Gt
    PowerDensityTx = Pt*Gt/(4*np.pi*Range**2)
    PowerIntercepted = PowerDensityTx * RCS
    PowerDensityReceived = PowerIntercepted * 1/(4*np.pi*Range**2)
    Ae = Gr * Lambda**2/(4*np.pi)
    PowerReceived = PowerDensityReceived * Ae
    Pr_dBm = 10*np.log10(PowerReceived/1e-3)
    return PowerReceived, Pr_dBm


def calcRmax(Pavg,Gt,Gr,Lambda, BW, RCS=1, T=291, NF = 1,L=1, SNRmin=1):
    kB = 1.38e-23
    denom = Pavg*Gt*Gr*(Lambda**2)*RCS
    numer = ((4*np.pi)**3)*kB*T*BW*NF*L*SNRmin
    Rmax = (denom/numer)**(1/4)
    return Rmax
