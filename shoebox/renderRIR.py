import torch
import numpy as np

from shoebox.langrange import lagrange

def render_rir(echogram, endtime, fs, FRACTIONAL=None, RAND_IMS=None, rand_ims_dels=None):
    """
    Samples the echogram to a specified sample rate.

    Parameters:
    - echogram: dict, echogram structure
    - endtime: float, time in seconds for the desired length of the impulse response
    - fs: int, sampling rate
    - FRACTIONAL: bool, whether to use fractional interpolation
    - RAND_IMS: bool, whether to randomize image source positions
    - rand_ims_dels: torch.Tensor or None, random time delays for image sources

    Returns:
    - IR: torch.Tensor, impulse response
    """

    if RAND_IMS is None:
        RAND_IMS = 0

    if FRACTIONAL is None:
        FRACTIONAL = 1

    # number of reflections inside the time limit
    # max used for 'last' in find
    idx_trans = torch.max(np.argwhere(echogram['time'] < endtime)).item()

    if FRACTIONAL:
        # get Lagrange interpolating filter of order 100 (filter length 101)
        order = 100
        L_frac = order + 1
        h_offset = 50
        h_idx = torch.arange(1 - 51, L_frac - 51)
        
        # make a filter table for quick access for quantized fractional samples
        fractions = torch.arange(0, 1.01, 0.01)  # 1.01 for extra one
        H_frac = lagrange(order, 50 + fractions)

        # initialize array
        tempIR = torch.zeros((int(np.ceil(endtime * fs * 1.1)) + 2 * h_offset, echogram['value'].size(1)))

        # render impulse response
        for i in range(1, idx_trans+1):  #(1, idx_trans + 1): satified by [i-1]
            # select appropriate fractional delay filter
            if RAND_IMS:
                refl_idx = torch.floor((echogram['time'][i-1] + rand_ims_dels[i-1]) * fs) + 1
                refl_frac = torch.fmod((echogram['time'][i-1] + rand_ims_dels[i-1]) * fs, 1)  # why mod 1 ? gets array of zeros
            else:
                refl_idx = torch.floor(echogram['time'][i-1] * fs) + 1
                refl_frac = torch.fmod(echogram['time'][i-1] * fs, 1)

            _, filter_idx = torch.min(torch.abs(refl_frac - fractions))
            h_frac = H_frac[:, filter_idx]

            tempIR[h_offset + refl_idx + h_idx, :] += h_frac * echogram['value'][i-1, :]

        IR = tempIR[h_offset:, :]

    else:
        # initialize array
        IR = torch.zeros((int(np.ceil(endtime * fs)), echogram['value'].size(1)))

        # quantized indices of reflections
        if RAND_IMS:
            refl_idx = torch.round((echogram['time'][:idx_trans] + rand_ims_dels[:idx_trans]) * fs) + 1
        else:
            refl_idx = torch.round(echogram['time'][:idx_trans] * fs) + 1

        # sum reflection amplitudes for each index
        for i in range(1, idx_trans + 1):
            IR[refl_idx[i-1].long(), :] += echogram['value'][i-1, :]

    return IR



