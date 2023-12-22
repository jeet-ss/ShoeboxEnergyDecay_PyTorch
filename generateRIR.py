import torch
#import matplotlib.pyplot as plt
#import numpy as np

from shoebox.findAbsCoeffsFromRT import find_abs_coeffs_from_rt
from util.stochasticRIR import stochastic_rir
from shoebox.filterRIR import filter_rir


# Set the seed for reproducibility
#torch.manual_seed(1)
#np.random.seed(1)

def mag2db(arr):
    return 20 * arr.log10_()

def db2mag(arr):
    return 10**(arr/20)


def generate_stochasticRIR(L, c=343, fs = 48000):
    # return frequency dependent RIR
    V = torch.prod(L)   # L catBack
    # Define params
    fs = 48000
    # rec = L * torch.tensor([0.41, 0.23, 0.41])      # relative Receiver position [x y z]
    # src = L * torch.tensor([0.82, 0.64, 0.55])      # relative Source position [x y z]'
    rt60 = torch.tensor([1.0, 0.8, 0.7, 0.6, 0.5, 0.4]) * 2.0       # per octave band
    nBands = len(rt60)
    # # 
    beta = torch.sqrt(1 - find_abs_coeffs_from_rt(L, rt60)[0])      
    beta = db2mag(mag2db(beta) + 0.1 * (torch.rand_like(beta) - 0.5))
    maxTime = torch.max(rt60).item()
    # limitsTime = torch.ones(nBands) * maxTime
    # #
    band_centerfreqs = torch.zeros(nBands)
    band_centerfreqs[0] = 125.0     # lowest octave band
    for it in range(1, nBands):
        band_centerfreqs[it] =  2.0 * band_centerfreqs[it-1]
    # 
    #
    #stochastic_rir(maxTime, beta[it], L, c, fs)[0]
    #
    # Stochastic RIR
    h_temp2 = torch.zeros((int(maxTime * fs), nBands))
    for it in range(nBands):
        h_temp2[:, it] = stochastic_rir(maxTime, beta[it], L, c, fs)[0]

    h_stochastic = filter_rir(h_temp2, band_centerfreqs, fs)

    return h_stochastic