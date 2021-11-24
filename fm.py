import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as wavf
from math import pi

sample_rate = 882000 # Hz
center_freq = 95.3e6 # Hz
num_samps = 20000 # number of samples returned per call to rx()

import matplotlib.pyplot as plt

def setup_pluto(sample_rate, center_freq, num_samps):
    sdr = adi.Pluto()
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(sample_rate)
    sdr.rx_buffer_size = num_samps
    return sdr

# Setup pluto and baseband filter
sdr = setup_pluto(sample_rate, center_freq, num_samps)
b, a = signal.iirfilter(10, 0.2, btype='lowpass')

# Collect samples
x = []
for i in range(1000):
    x.extend(sdr.rx())

# Filter baseband samples
y = signal.lfilter(b, a, x)

# Demodulate fm
fm_demod = []
prev_fm = 0    
for i in range(len(y)):        
    fm_demod.append(np.angle(np.conjugate(prev_fm) * y[i]) * 2 * pi * 75000)
    prev_fm = y[i]

# Normalise and Downsample
fm_demod = np.array(fm_demod)   
fm_demod = fm_demod / np.amax(fm_demod)
audio = signal.decimate(fm_demod, 20)    

# Write to file
wavf.write("out.wav", 44100, audio)

# Plot recovered audio
plt.figure()
plt.plot(audio)

# Plot audio fft
plt.figure()
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(audio))))

# Plot baseband spectrograph
plt.figure()
plt.specgram(y, NFFT=1024, Fs=sample_rate)
plt.colorbar()
plt.show()


