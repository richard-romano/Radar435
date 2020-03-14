# Samantha Fink
# ECE 431
# HW Test File for HW #5

# Jupyter Modifications:

#1. Add probability of detection function
	# Probability of failure function: Wi = e^(-(P/Po)*sqrt(N))
		# P = signal, Po = noise, N = number of observations
		# WANT: Probability of detection, -> P(detect) = 1 - Wi
		# Will need 'math' library in order to use exp()

import math

#Test values
N = 10
SNR = 0.73
exponent = -(SNR * math.sqrt(N))
Wi = math.exp(exponent)
print(Wi)
solution = 1 -Wi
print ('Solution = {}'.format(solution))

##########################
#PART ONE: DETERMINE SNR
##########################

###Signal variables, using values from #6 on midterm
#maxRange = 35e3 #max range of radar in km
#PRI = 500 #pulse repetition interval, in pps
#fc = 80e6 #center frequency, in Hz
#PW = 10e-6 #pulse width, in seconds
#Pavg = 100e3 #transmitter power, in KW
#RCS = 15 #radar cross section, in m^2
#Gt = 20 #transmitter gain; scalar
#Gr = 20 #receiver gain; scalar

#Calculate wavelength and Ae
#wavelength = 3e8/fc
#Ae = (Gr*wavelength**2)/(4*math.pi)
#print('Wavelength = {}m'.format(wavelength))
#print('Ae = {}m^2'.format(Ae))

#Calculate Power density at target and of the reflected wave
pDensTxRx = (1/(4*math.pi*(maxRange**2)))**2 #power density at target and 
#reflected wave; simpified for signal eqn

#Calculate signal
signal = Gt*Pavg*pDensTxRx*RCS*Ae
print('Signal = {}'.format(signal))


### Noise variables
#NF = 7.3 #noise factor; scalar
#T = 291 #temperature, in kelvin
#L = 1 #external noise, generally 3-10dB. Using 1 for simulation

#Calculate noise
#beta = 1/PW
print('Bandwidth = {} s'.format(beta))
#noise = const.Boltzmann*beta*NF*L*T
print('Noise = {}'.format(noise))

### Calculate SNR
#SNR = signal/noise
#print('\n'+'SNR = {}'.format(SNR))

########################################
#PART TWO: DETERMINE NOISE PROBABILITY
########################################

### Probability variables
#N = 10 #number of observations; scalar

#Determine false probability detection using above parameters

SNR = 1 
N = 1
Wi = math.exp(-((SNR)*math.sqrt(N)))
print('\n')
print('\033[1m'+'*** FALSE PROBABILITY RESULTS ***'+'\033[0m')
print('False positive probability = {}'.format(Wi))

#Test script, using values from #10 on midterm
#N = 10
#SNR = 0.73
#exponent = -(SNR * math.sqrt(N))
#Wi = math.exp(exponent)
#print ('Test solution = {}'.format(Wi))