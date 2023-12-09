import torch

from shoebox.findAbsCoeffsFromRT import find_abs_coeffs_from_rt
from util.stochasticRIR import stochastic_rir
from shoebox.filterRIR import filter_rir
from generate_StochasticRIR import generate_stochastic_rir



def mag2db(arr):
    return 20 * arr.log10_()

def db2mag(arr):
    return 10**(arr/20)


def generate_frequencyRIR(L, c=343, fs = 48000):
    V = torch.prod(L)   # L catBack
    # Define params
    fs = 48000
    rec = L * torch.tensor([0.41, 0.23, 0.41])      # relative Receiver position [x y z]
    src = L * torch.tensor([0.82, 0.64, 0.55])      # relative Source position [x y z]'
    rt60 = torch.tensor([1.0, 0.8, 0.7, 0.6, 0.5, 0.4]) * 2.0       # per octave band
    nBands = len(rt60)
    # # 
    beta = torch.sqrt(1 - find_abs_coeffs_from_rt(L, rt60)[0])      
    beta = db2mag(mag2db(beta) + 0.1 * (torch.rand_like(beta) - 0.5))
    maxTime = torch.max(rt60).item()
    limitsTime = torch.ones(nBands) * maxTime
    # #
    band_centerfreqs = torch.zeros(nBands)
    band_centerfreqs[0] = 125.0     # lowest octave band
    for it in range(1, nBands):
        band_centerfreqs[it] =  2.0 * band_centerfreqs[it-1]
    # Stochastic RIR
    
    h_temp2 = torch.zeros((int(maxTime * fs), nBands))
    K_values = torch.zeros((nBands, 3))
    for it in range(nBands):
        Kx = torch.log(torch.prod(beta[it][0:2])) / L[0]
        Ky = torch.log(torch.prod(beta[it][2:4])) / L[1]
        Kz = torch.log(torch.prod(beta[it][4:6])) / L[2]
        K_values[it, :] = torch.cat((Kx.view(-1), Ky.view(-1), Kz.view(-1)))
        #h_temp2[:, it] = stochastic_rir(maxTime, beta[it], L, c, fs)[0]
        h_temp2[:, it] = generate_stochastic_rir(Kx=Kx, Ky=Ky, Kz=Kz)


    return h_temp2, K_values