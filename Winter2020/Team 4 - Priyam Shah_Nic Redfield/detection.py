import numpy as np
from math import factorial
import scipy
from scipy import signal

def envelope(signal):
    """
    Fast envelop detection of real signal using thescipy hilbert transform
    Essentially adds the original signal to the 90d signal phase shifted version
    similar to I&Q
    """
    signal = scipy.signal.hilbert(signal)
    envelope = abs(signal)# np.sqrt(signal.imag**2 + signal.real**2)
    return envelope