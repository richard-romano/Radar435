#Generate FM Chirp

from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import numpy as np

# Create the sine wave
amplitude = np.iinfo(np.int16).max #Setting sine output from -32k to 32k
A = amplitude

num_secs = .05

t = np.linspace(0, num_secs, 12000)
w = A * chirp(t, f0=1, f1=6000, t1=num_secs, method='logarithmic' )

plt.plot(t,w)
plt.title("Logarithmic Chirp, f(0) = 1Hz, f(50ms) = 6kHz")
plt.xlabel('Seconds (s)')
plt.show()

