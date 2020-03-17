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
#------------------------------------------------------------------
#------------------------------------------------------------------ ENVELOPE, GATE, & SUM (RAW SIGNAL)
#------------------------------------------------------------------
def enveloping(audio_in, K_pulses, len_PRI, filt_trace, R_unamb, BW, fs):
    # Envelope detect the signals
    main_trace_env = det.envelope(audio_in)
    print('**Envelope created for raw audio input')

    nsamps = len(audio_in)
    x = np.linspace(0,R_unamb, nsamps)

    #Tx pulse detection to shift Tx to left side of plots (happens at ~600th sample)
    roll = 0
    for idx in range(1000,nsamps):
        if main_trace_env[idx] > 5e3:
            roll = idx
            break
        
    main_trace_env = np.roll(main_trace_env,-roll)
 

    # Gate the signal & sum them up for n observation effects
    n_obs_main_trace_env = main_trace_env.reshape(K_pulses, len_PRI)

    #Display all pulses together before adding (they should line up)
    for idx in range(0, K_pulses):
        plt.subplot(K_pulses+1,1,idx+1)
        plt.semilogy(n_obs_main_trace_env[idx,::])
        #plt.title('Gated Pulses to be Summed')
        plt.ylabel('{}'.format(idx))
    plt.show()

    # add them all together
    n_obs_main_trace_env = n_obs_main_trace_env.sum(axis=0)
    print('**Gated and summed raw audio input')


    #------------------------------------------------------------------
    #------------------------------------------------------------------ ENVELOPE, GATE, & SUM (FILTERED SIGNAL)
    #------------------------------------------------------------------

    # Redo envelope detection on filtered signal
    filt_trace_env = det.envelope(filt_trace)
    print('**Envelope created for filtered audio input')

    '''
    plt.semilogy(x,filt_trace_env)
    plt.ylim(10e1, 10e5)
    plt.title('Filtered Trace Envelope')
    plt.xlabel('samples')
    plt.ylabel('Power (dB)')
    plt.show()
    '''

    # Redo gating and summing
    filt_trace_env = np.roll(filt_trace_env,-roll)
    n_obs_filt_trace_env = filt_trace_env.reshape(K_pulses, len_PRI)
    n_obs_filt_trace_env = n_obs_filt_trace_env.sum(axis=0)
    print('**Gated and summed filtered audio input')
    #build x-axis
    nsamps = len(n_obs_main_trace_env)
    x = np.linspace(0,R_unamb, nsamps)

    #plot the filtered signal over the unfiltered one to show noise reduction
    #plt.rcParams['figure.figsize'] = [20, 10]
    plt.plot(x,10*np.log10(n_obs_main_trace_env/1e-3), label='Unfiltered')
    plt.plot(x,10*np.log10(n_obs_filt_trace_env/1e-3), label='f$_c$-Filtered')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='best')
    plt.title('Bandpass Filter Signal for f$_c$ to drop noise floor')
    plt.xlabel('Distance (m)')
    plt.ylabel('Power (dBm)')

    plt.show()

    #------------------------------------------------------------------
    #------------------------------------------------------------------ THRESHOLDING
    #------------------------------------------------------------------
    

    #------------------------------------------------------------------
    #------------------------------------------------------------------ FILTERING GATED/SUMMED ENVELOPES
    #------------------------------------------------------------------
    #filter the envelope to smooth out envelope. This will degrade the amplitude but all I need is the sample number for each peak
    BW_BP_filt_start = 1e-20 #lazy lowpass
    BW_BP_filt_stop = BW*0.75
    filt_env = butt.butter_bandpass_filter(n_obs_filt_trace_env, BW_BP_filt_start, BW_BP_filt_stop, fs, order=1 )
    return filt_env, n_obs_main_trace_env;
    print('**Filtered Detected Envelope for {} to {} Hz'.format(BW_BP_filt_start, BW_BP_filt_stop))

    #make a plot for the double-filtered signal used to detect peaks and show the trigger threshold
    #plt.rcParams['figure.figsize'] = [20, 10]
    plt.plot(x, 10*np.log10(np.abs(filt_env)/1e-3),label='BW-Filtered')
    plt.axhline(y=10*np.log10(trigger/1e-3), xmin=0, xmax=len(filt_env), linewidth=1, color = 'r', linestyle='dotted', label='Trigger Threshold')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='best')
    plt.title('Detected Peaks (post-filtering)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Power (dB)')
    plt.show()


    ##*** STOP FUNCTION