#Generate FM Chirp

from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import numpy as np

num_secs = 10

t = np.linspace(0, num_secs, 5001)
w = chirp(t, f0=4000, f1=6000, t1=num_secs, method='linear' )

plt.plot(t[:500],w[:500])
plt.title("Linear Chirp, f(0) = 1, f(10) = 10")
plt.xlabel('Seconds (s)')
plt.show()