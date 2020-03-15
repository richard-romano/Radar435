import detection as det
print('Imported Detection.py')
import butterworth as butt
print('Imported Butterworth.py')
import pyaudio
print('Imported PyAudio')
import os
print('Imported OS')
import numpy as np
print('Imported NumPy')
from matplotlib import pyplot as plt
print('Imported PyPlot from MatPlotLib')
import scipy
print('Imported SciPy')
from scipy import signal as sig
print('Imported Signal from SciPy')
from scipy.signal import butter, lfilter
print('Imported Butter and LFilter from SciPy.Signal')
plt.switch_backend('Qt4Agg')
import enveloping as env
print('Imported enveloping.py')
import cross_correlate_function as ccf
print('Imported xcorr.py')
#------------------------------------------------------------------
#------------------------------------------------------------------ MAIN SETUP
#------------------------------------------------------------------
fc = 2000           # Carrier Frequency, Center Frequency
vp = 344            # Phase Velocity of the wave
T  = 1/fc           # period of one Carrier Frequency
#derived values
Lambda = vp/fc

# Setup Time portion
PRF = 4         # Pulses per second (hertz)
PRI = .25         # Pulse Repetition Interval (seconds)
R_unamb = PRI *vp/2 # Unambiguous Range

#Num cycles per pulse packet
k = 100            # k cycles of fc in the pulse packet
PW = k*T            # k cycles * Period of fc
BW = 1/PW           # Bandwidth of the RADAR Pulse
K_pulses = 20
dt_k = 40
#------------------------------------------------------------------
#------------------------------------------------------------------ PYAUDIO
#------------------------------------------------------------------
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 #samples per second
fs = RATE
RECORD_SECONDS = K_pulses*PRI

#WAVE_OUTPUT_FILENAME = "recording.wav"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)

#clear display of ALSA errors
clear = lambda: os.system('clear')
clear()

print("Recording for {} seconds...".format(round(RECORD_SECONDS,1)))

frames = []
for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
    audio_in = stream.read(CHUNK)
    frames.append(audio_in)

stream.stop_stream()
stream.close()
p.terminate()
print("Complete!")
print('')
#------------------------------------------------------------------
#------------------------------------------------------------------ RAW DATA MANIPULATION
#------------------------------------------------------------------
t_vector = np.arange(0,PRI,1/fs)
len_PRI = len(t_vector)

frames = np.array(frames)
audio_in = np.frombuffer(frames, dtype=np.int16)
len_audio = len(audio_in)
len_env = len_PRI*(K_pulses)

#the length of the audio doesn't match the reshaping of the envelope detection
#this fixes that, but it's a janky solution
if len_audio < len_env: #if the audio length is too short, pad it with the median value of the audio
    audio_in = np.pad(audio_in, int(((len_PRI*(K_pulses)-len_audio)/2)), mode = 'median')
else: #if the audio length is too long, remove samples from the beginning where there's a weird artifact anyway
    audio_in = audio_in[len_audio-len_env::]

nsamps = len_env
'''
plt.semilogy(audio_in)
plt.title('Raw Audio Input')
plt.xlabel('sample number')
plt.ylabel('Power (dB)')
plt.show()
'''
print('**Tx Parameters Input OK')

#------------------------------------------------------------------
#------------------------------------------------------------------ FILTER RAW SIGNAL FOR fc
#------------------------------------------------------------------

#Tx pulse detection to shift Tx to left side of plots (happens at ~600th sample)
roll = 0
for idx in range(1000,nsamps):
    if audio_in[idx] > 1.5*10e4:
        roll = idx
        break

print('What type of analysis?')
print('1. Cross Correlation (takes a long time)')
print('2. Envelope Detection (low resolution)')
cc_or_env = int(input('Enter Selection: '))

#make a blank list and add in each peak as these conditions are met (note: Tx peak will be first pulse detected)
#maxima = list()

if cc_or_env == 1:   
    fc_BP_filt_start = (0.25-0.1)*fc
    fc_BP_filt_stop = (4+1)*fc
    filt_trace = butt.butter_bandpass_filter(audio_in, fc_BP_filt_start, fc_BP_filt_stop, fs, order=1)
    print('**Filtered audio for {} Hz to {} kHz'.format(fc_BP_filt_start, fc_BP_filt_stop/1000))
    print('')
    
    filt_trace_roll = np.roll(filt_trace,-roll) #roll the filt_trace using the above roll NEW
    filt_trace_reshape = filt_trace_roll.reshape(K_pulses, len_PRI) #reshapE into an array of K_pulses by len_PRI #NEW
    filt_trace_sum = filt_trace_reshape.sum(axis=0) #filt_trace summed up together #NEW

    x = np.linspace(0,R_unamb, len(filt_trace_sum))
#     plt.semilogy(x,filt_trace_sum) #NEW
#     plt.xlabel('Distance (m)')
#     plt.title('Filtered/Gated/Summed Signal')
#     plt.ylim(10e2, 10e5)
#     plt.show() #NEW
    
    print('')
    print('**Cross-correlating, please wait...')
    c_c = ccf.cross_correlation(filt_trace_sum, PRI) #changed to cross_correlation of filt_trace_sum
    print('Cross-correlation complete!')

    c_c_env = det.envelope(c_c)

    print('Cross-correlated Ranging')
    plt.semilogy(x,c_c_env)
    plt.title('Cross-correlation')
    plt.xlabel('Distance (m)') 
    #plt.ylim(10e2, 10e5)
    plt.show()
    
    #triggerdB = int(input('Input Threshold (dB): '))
#     #trigger = 10**((triggerdB)/10)
#     
#     for idx in range(1, len(c_c)-1):
#         if c_c[idx] > c_c[idx-1] and\
#         c_c[idx] > c_c[idx+1] and\
#         c_c[idx] > trigger:
#             maxima.append(idx)


elif cc_or_env == 2:
    # Filter signal for fc
    fc_BP_filt_start = (0.75)*fc
    fc_BP_filt_stop = (1.25)*fc
    filt_trace = butt.butter_bandpass_filter(audio_in, fc_BP_filt_start, fc_BP_filt_stop, fs, order=1)
    print('**Filtered audio for {} Hz to {} kHz'.format(fc_BP_filt_start, fc_BP_filt_stop/1000))
    print('')
    # Envelope
    filt_env, n_obs_main_trace_env = env.enveloping(audio_in, K_pulses, len_PRI, filt_trace, R_unamb, BW, fs)
    #triggerdB = int(input('Input Threshold (dBm): '))
    #trigger = 10**((triggerdB-30)/10)

#     for idx in range(1, len(filt_env)-1):
#         if filt_env[idx] > filt_env[idx-1] and\
#         filt_env[idx] > filt_env[idx+1] and\
#         filt_env[idx] > trigger:
#             maxima.append(idx)

#------------------------------------------------------------------
#------------------------------------------------------------------ PEAK DETECTION
#------------------------------------------------------------------

#this is an array of all the sample numbers where a peak occurs above the threshold.
#maxima = np.array(maxima)


#------------------------------------------------------------------
#------------------------------------------------------------------ DISPLAY OUTPUT
#------------------------------------------------------------------
#establish the noise floor for later calculations
noisefloor = np.average(filt_trace)
noisefloordBm = np.average(10*np.log10(np.abs(filt_trace)/1e-3))

#clear display for neato readout
clear = lambda: os.system('clear')
clear()

# #headers for neato readout
# print('Noise Floor at {} dB'.format(round(noisefloordBm,1)))
# print('{} Targets found above {} dB threshold:'.format(maxima.size-1,triggerdB))
# print('')
# print('      Range      Amplitude    Detection Positivity')
# 
# #average this number of samples before & after each detected peak to reduce impact of unexpected minima in the envelope waveform
# 
# for idx in range(1, len(maxima)): #note: start this index at 1 (not 0) to skip over the Tx pulse
#     T_Range = vp*(maxima[idx]/fs)/(2) #range in m
#     T_Power = np.average(n_obs_main_trace_env[(maxima[idx]-dt_k):(maxima[idx]+dt_k)])
#     T_PowerdBm = 10*np.log10(np.abs(T_Power))
#     SNRpeak = T_Power/noisefloor
#     SNRpeakdB = 10*np.log10(SNRpeak)
#     T_Conf = np.round(100*(1-np.exp(-SNRpeak*np.sqrt(K_pulses))),1)
# 
#     print('{: >2}: {: >7} m |{: >7} dB |{: >7}% confidence'.format(idx, np.round(T_Range,2), np.round(T_PowerdBm,2), T_Conf))
#     