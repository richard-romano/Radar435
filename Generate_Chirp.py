#Generate FM Chirp

from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import numpy as np

num_secs = .05

t = np.linspace(0, num_secs, 5001)
w = chirp(t, f0=1, f1=6000, t1=num_secs, method='logarithmic' )

plt.plot(t,w)
plt.title("Linear Chirp, f(0) = 1, f(10) = 6kHz")
plt.xlabel('Seconds (s)')
plt.show()
