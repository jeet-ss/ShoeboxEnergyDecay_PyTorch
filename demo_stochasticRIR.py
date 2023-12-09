import torch
import matplotlib.pyplot as plt
import numpy as np
import control

from shoebox.findAbsCoeffsFromRT import find_abs_coeffs_from_rt
from util.stochasticRIR import stochastic_rir
from shoebox.filterRIR import filter_rir
#from shoebox.computeEchogramsMics import compute_echograms_mics
#from shoebox.renderMicRIRs import render_mic_rirs

# Set the seed for reproducibility
#torch.manual_seed(1)
np.random.seed(1)

# Define parameters for example room
L = torch.tensor([4.0, 5.0, 3.0])
c = 343.0
V = torch.prod(L)

# Define params
fs = 48000
rec = L * torch.tensor([0.41, 0.23, 0.41])      # relative Receiver position [x y z]
src = L * torch.tensor([0.82, 0.64, 0.55])      # relative Source position [x y z]'
rt60 = torch.tensor([1.0, 0.8, 0.7, 0.6, 0.5, 0.4]) * 2.0       # per octave band
nBands = len(rt60)
##

beta = torch.sqrt(1 - find_abs_coeffs_from_rt(L, rt60)[0])      
beta = control.db2mag(control.mag2db(beta) + 0.1 * (torch.rand_like(beta) - 0.5))
maxTime = torch.max(rt60).item()
limitsTime = torch.ones(nBands) * maxTime

band_centerfreqs = torch.zeros(nBands)
band_centerfreqs[0] = 125.0     # lowest octave band
for it in range(1, nBands):
    band_centerfreqs[it] =  2.0 * band_centerfreqs[it-1]
# band_centerfreqs[1:] = 2.0 * band_centerfreqs[:-1]

# Stochastic RIR
h_temp2 = torch.zeros((int(maxTime * fs), nBands))
for it in range(nBands):
    h_temp2[:, it] = stochastic_rir(maxTime, beta[it], L, c, fs)[0]

#torch.save(h_temp2, 'h_temp2.pt')
h_stochastic = filter_rir(h_temp2, band_centerfreqs, fs)
#torch.save(h_stochastic, 'h_stochastic.pt')

plt.figure()
plt.plot(h_stochastic.detach())
plt.show()

# # Image Sources
# abs_echograms, rec_echograms, echograms = compute_echograms_mics(L, src, rec, 1 - beta ** 2, limitsTime)
# h_temp = render_mic_rirs(abs_echograms, band_centerfreqs, fs)
# h_ism = h_temp[:len(h_stochastic)]

# # Combined
# cuton = int(0.10 * fs)
# h_combined = torch.cat([h_ism[:cuton], h_stochastic[cuton:]])

# # Plotting
# def print_spectrogram(waveform, fs, title):
#     plt.figure()
#     plt.specgram(waveform.numpy(), Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.title(title)
#     plt.colorbar(label='Power (dB)')
#     plt.show()

# print_spectrogram(h_ism, fs, 'ISM Spectrogram')
# print_spectrogram(h_stochastic, fs, 'Stochastic Spectrogram')
# print_spectrogram(h_combined, fs, 'Combined Spectrogram')

# # Sound
# torchaudio.save('./results/ism.wav', h_ism.unsqueeze(0), fs)
# torchaudio.save('./results/stochastic.wav', h_stochastic.unsqueeze(0), fs)
# torchaudio.save('./results/combined.wav', h_combined.unsqueeze(0), fs)
