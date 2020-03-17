# Radar signal simulator
# ECE 435/535

'''
Method: -- this is a script..
1 - Create a pulse train from Transmitter
2 - Generate a list of targets, (Range, RCS) ('Setup the testing environment')
3 - Generate return pulses for each of the targets into a single train
4 - Attenuate 1 to reasonable power level
5 - Add output from step 3 to output of step 4
6 - Add AGWN to 5 (AGWN = Additive Gaussian White Noise)
7 - Apply detection method
'''

from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig

# Custom libraries
import pulsetrain as pt
import pulsenoise as pn
import detection as det

# Student Library
import lastname as GPD # Initials (George P. Burdell)


# setup Radar
Pavg = 100e3        # Basic Power level output of the radar
Gt = 15             # Scalar Gain of TX antenna
Gr = Gt             # Scalar Gain of RX antenna  if Gr == Gt same antenna
fc = 40e6           # Carrier Frequency, Center Frequency
vp = 3e8            # Phase Velocity of the EM wave
NF = 1              # Receiver Noise Figure
T  = 1/fc           # period of one Carrier Frequency
#derived values
Lambda = vp/fc

# Setup Time portion
PRF = 500           # Pulses per second (hertz)
PRI = 1/PRF         # Pulse Repetition Interval (seconds)
R_unamb = PRI *vp/2 # Unambiguous Range

#Num cycles per pulse packet
k = 100             # k cycles of fc in the pulse packet
PW = k*T            # k cycles * Period of fc
BW = 1/PW           # Bandwidth of the RADAR Pulse
# error check
if PW >= PRI:
    print('Error: Pulse width much too long -- PRI: {}, PW = {}'.format(PRI, PW))

# calculate maximum range with SNR = 1, n Observations = 1
SNRmin = 1
RCS = 1
Rmax = pt.calcRmax(Pavg,Gt,Gr,Lambda, BW, SNRmin = SNRmin, RCS = RCS) #, RCS, T, NF = 1,L=1, SNRmin=1)
print('Rmax(SNR:{}, RCS:{}) \t= {:.02f} km'.format(SNRmin, RCS, Rmax/1e3))
print('R unambiguous \t\t= {:.02f}km'.format(R_unamb/1e3))

# setup the test enviroment
num_targets = 10

target_ranges = np.random.randint(Rmax//4,Rmax,num_targets)
target_rcs = np.random.randint(1,1000,num_targets)


# Now we have the Radar configuration, and simulated targets. Next we build the radar returns

# build base Pulse train as origin from the transmitter
K_pulses = 20 # how many PRI's get simulated
dt_k = 20 # how many samples per fc period (Tc)
# create the base pulse train
# we will cheat here and make the Main pulse -100dBm
dBm = -100 #dBm
scalar = 1e-3 * np.power(10,(dBm/10))

main_train, PW, dt, len_PRI = pt.createPulseTrain(A=scalar,fc = fc, k=k, PRI=PRI, dt_k=dt_k, K_pulses = K_pulses)


#main_train_noise = pn.addNoiseToPulseTrain(main_train, 1/PW, NF =10)


# Now we create the returns...
main_trace = np.zeros_like(main_train) # return without TX

for idx, target_range in enumerate(target_ranges):

    pwr, dbm = pt.RadarEquationAdv(Pavg, Gt, target_range, RCS, Gr, Lambda, dB=False)
    print(':: idx: {} Power at RX {} dBm @ range: {} rmax {}'.format(idx,(10*np.log10(Pavg/1e-3)),
                                                                     target_range, R_unamb ))
    p_train, PW, dt, len_PRI = pt.createPulseTrain(A=pwr,fc = fc, k=k, PRI=PRI,
                                                   dt_k=dt_k, K_pulses = np.int(K_pulses))
    # time shift to correct spot
    p_train = pt.timeShift(p_train, target_range,vp, dt, len_PRI)
    main_trace = main_trace + p_train

#-------------------------------
print(len(main_train), len(main_trace))

# Normalize main_train to 1


# -------------------------------
# now we add the two systems together.
# Add noise to the pulse traing
main_trace = main_trace + main_train

main_trace = pn.addNoiseToPulseTrain(main_trace,1/PW)

# -------------------------------
# Detection Section
# Envelope detect the signals
main_trace_env = det.envelope(main_trace)

# provide n observation effects
n_obs_main_trace_env = main_trace_env.reshape(K_pulses+1, len_PRI)
# add them all together
n_obs_main_trace_env = n_obs_main_trace_env.sum(axis=0)


# make the distance vector
nsamps = len(n_obs_main_trace_env)
x = np.linspace(0,R_unamb, nsamps)
dx = R_unamb/(len(n_obs_main_trace_env))

from scipy.signal import find_peaks

dBm = -100 #dBm
scalar = 1e-3 * np.power(10,(dBm/10))
#height = scalar



peaks, _ = find_peaks(n_obs_main_trace_env, height=scalar)
# peaks is a vector of peak
plt.semilogy(x/1e3,n_obs_main_trace_env)
plt.semilogy(peaks*dx/1e3,n_obs_main_trace_env[peaks], 'x')
plt.show()
