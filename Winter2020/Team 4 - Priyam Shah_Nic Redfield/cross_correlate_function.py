import numpy as np
import scipy
from scipy import signal as sig
from scipy.signal import chirp 




def cross_correlation(function, PRI):
     t = np.linspace(0, PRI, 10000)
     kernel = chirp(t,f0=500,f1=8000,t1=PRI, method='linear')
     print('Length of function is {}'.format(len(function)))
     print('Length of kernel is {}'.format(len(kernel)))
     c_c = sig.correlate(function, kernel, mode='same',method='direct')
     return c_c
  
