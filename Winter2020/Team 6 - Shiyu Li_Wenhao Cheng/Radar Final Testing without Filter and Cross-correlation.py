import pyaudio
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')
from scipy import signal as sig
import math
from scipy import stats
import statistics
from statistics import median
from scipy.signal import find_peaks
#import scipy.io.wavfile

chunk = 5*6615 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
fs=44100
PRI=0.15


p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                frames_per_buffer = chunk)

print('recording')

all = []

#recording time
i=0
for i in range(0,6):
    data = stream.read(chunk, exception_on_overflow = False)
    all.append(data)
    i+=1
print('done recording')

stream.close()
p.terminate()

print(len(data))

numpydata = np.frombuffer(data, dtype=np.int16)
plt.plot(numpydata)
plt.show()

print(numpydata)
numpydata=numpydata.reshape(5,int(PRI*fs))

print(numpydata)

#Roll the maximum value of each row to 0
from numpy.lib.stride_tricks import as_strided
def custom_roll(arr, r_tup):
    m = np.asarray(r_tup)
    arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].copy() #need `copy`
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    result = as_strided(arr_roll, (*arr.shape, n), (strd_0 ,strd_1, strd_1))

    return result[np.arange(arr.shape[0]), (n-m)%n]

max_index= numpydata.argmax(axis=1)
print(max_index)

signal_roll = custom_roll(numpydata, -max_index)

print(signal_roll)



# https://blog.csdn.net/weixin_30716725/article/details/95158490?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

# plot data
fig, axs=plt.subplots(6)
axs[0].plot(signal_roll[0,:])
axs[1].plot(signal_roll[1,:])
axs[2].plot(signal_roll[2,:])
axs[3].plot(signal_roll[3,:])
axs[4].plot(signal_roll[4,:])
axs[5].plot(np.sum(signal_roll,axis=0))
signal=np.sum(signal_roll,axis=0)
print(signal)
print(len(signal))
plt.show()




#envelop signal
plt.plot(signal)
signal_env= sig.hilbert(signal)
signal_envelope = abs(signal_env)
plt.plot(signal_envelope)

plt.show()

# Find SNR
n=5
su_Prob=0.9
Noise=statistics.median([signal_envelope])
S_max=-Noise*np.log(0.1)/math.sqrt(n)
height=su_Prob*S_max
print(height)


vp=343

SNRmin=1
RCS=1
n=5
R_unamb = PRI*vp/2


K_pulses = 5 # how many PRI's get simulated
dt_k = 5 # how many samples per fc period (Tc)



peaks, _ = find_peaks(signal_envelope, height=su_Prob*S_max)

nsamps = len(signal_envelope)
x = np.linspace(0,R_unamb, nsamps)
dx = R_unamb/(len(signal_envelope))


plt.plot(x/1e3, signal_envelope)
plt.plot(peaks*dx/1e3,signal_envelope[peaks], 'x')
plt.xlabel('Distance in km')
plt.ylabel('Power in Watts')
plt.show()


#   #######################
import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt


# In[2]:

#kernelRate, KernelData=scipy.io.wavfile.read('PulseTrain_chirp_1000_10_05.wav')
#print(kernelRate)
#print(KernelData)











