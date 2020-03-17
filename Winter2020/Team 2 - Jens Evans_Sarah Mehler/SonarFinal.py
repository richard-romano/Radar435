# Code for working sonar using Matrix One
# ECE 435/535
# Jens Evans, Sarah Mehler

# Initialize libraries
import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import correlate
from scipy.io import wavfile as wfile

# Define functions
def envelope(signal):
    '''
    Fast envelop detection of real signal using thescipy hilbert transform
    Essentially adds the original signal to the 90d signal phase shifted version
    similar to I&Q
    '''
    signal = sig.hilbert(signal)
    envelope = abs(signal)# np.sqrt(signal.imag**2 + signal.real**2)
    return envelope

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Import audio for cross correlation
samplerate, ChirpKernel = wfile.read('C:\\Users\\jense\\ChirpTrain.wav', mmap=False)

#Number of observations
N = 10

# PRI length in Seconds
PRI=0.236
CHUNKSIZE = int(44100*PRI) # fixed chunk size
rate=44100

# Initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=CHUNKSIZE)

# Initialize data array
data = []

# listen
print('Listening...')
x = 0
while x < N:
    values = stream.read(CHUNKSIZE)
    data.append(values)
    x = x+1

# record
data = np.array(data)
MicData = np.frombuffer(data, dtype=np.int16)
stream.stop_stream()
stream.close()
p.terminate()
print('Done.')

#Process Data
# Envelope Hilbert Transform
MicData = butter_bandpass_filter(MicData, 3.5e3, 6.5e3, fs=44100, order=1)
MicData = envelope(MicData)
#Reshape
MicData = MicData.reshape(N,int(x*CHUNKSIZE/N))

# Store Envelope without Cross Correlation for later Comparison
GatedEnvelope=np.sum(MicData,axis=0)
max_value = max(GatedEnvelope)
max_index = np.argmax(GatedEnvelope)
GatedEnvelope = np.roll(GatedEnvelope,-1*max_index)

# Cross Correlation
# Setup Kernel to same length as PRI
ChirpKernel = ChirpKernel[0:len(MicData[0,:])]
ChirpKernel = np.roll(ChirpKernel,int(len(ChirpKernel)/2))
#Normalize Chirpkernel for a better correlation
ChirpKernel = ChirpKernel*(np.amax(MicData)/np.amax(ChirpKernel))
#Test Point
plt.figure(1)
plt.plot(ChirpKernel)
# Find Envelope of ChirpKernel
ChirpKernel = envelope(ChirpKernel)

# Compares ChirpKernel pre-envelope vs. envelope vs. MicData
plt.plot(ChirpKernel)
plt.plot(MicData[5,:])
plt.show()

# Cross-correlate signals row by row
x=0
while x < N:
    MicData[x,:] = sig.correlate(MicData[x,:],ChirpKernel,mode='same')/N
    x=x+1

#Gate Data
GatedData=np.sum(MicData,axis=0)

# Find Max
max_value = max(GatedData)
max_index = np.argmax(GatedData)

GatedData = np.roll(GatedData,-1*max_index)
MicData = np.roll(MicData,-1*max_index)

#Index Peaks
peaks, _ = find_peaks(GatedData, height=2*np.median(GatedData), distance=200)
dx=343/(rate*2)
peakdistance=dx*peaks

print('The Number of target is', (len(peakdistance)))
print("The distance to targets in meters is")
for i in range(len(peakdistance)-1):
    print("\n Target %d: %3f km" %(i, peakdistance[i]))

# Create a distance vector so things are in meters /2 is to compensate
# for the return trip
t=np.arange(0,len(GatedData))*dx

# plot data
plt.figure(2)
plt.title('Sonar Return Data')
plt.xlabel('Distance in Meters')
plt.ylabel('Gain in dB')
plt.semilogy(t,GatedData)
plt.semilogy(peaks*dx,GatedData[peaks], 'x')
plt.show()