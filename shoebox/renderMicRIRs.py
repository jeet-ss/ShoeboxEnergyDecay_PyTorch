import torch
import numpy as np

from shoebox.renderRIR import render_rir
from shoebox.filterRIR import filter_rir

def render_mic_rirs(echograms, band_centerfreqs, fs, endtime=None):
    """
    Render microphone room impulse responses (RIRs).

    Parameters:
    - echograms: torch.Tensor, echograms structure (nSrc x nRec x nBands)
    - band_centerfreqs: torch.Tensor or None, center frequencies for filterbank
    - fs: int, sampling rate
    - endtime: float or None, desired length of the impulse response

    Returns:
    - rirs: torch.Tensor, rendered RIRs (L_tot x nRec x nSrc)
    """

    nRec = echograms.size(1)
    nSrc = echograms.size(0)
    nBands = echograms.size(2)

    # Sample echogram to a specific sampling rate with fractional interpolation
    FRACTIONAL = 1
    # Randomize slightly position of image source to avoid sweeping echoes
    RAND_IMS = 1

    # Decide on the number of samples for all RIRs
    if endtime is None:
        endtime = 0
        for ns in range(nSrc):
            for nr in range(nRec):
                temptime = echograms[ns, nr, 0]['time']
                if temptime > endtime:
                    endtime = temptime

    L_rir = int(np.ceil(endtime * fs * 1.1))
    L_fbank = 1000 if nBands > 1 else 0
    L_tot = L_rir + L_fbank

    # Render responses and apply filterbank to combine different decays at different bands
    rirs = torch.zeros((L_tot, nRec, nSrc))

    for ns in range(nSrc):
        for nr in range(nRec):
            # If randomization/jitter of image source positions is on, precompute
            # random time delays uniformly distributed on -dx:dx
            if RAND_IMS:
                dx_max = 0.1
                dt_max = dx_max / 343  # speed of sound
                dt = 2 * dt_max * torch.rand(len(echograms[ns, nr, 0]['time'])) - dt_max
                dt[0] = 0  # Preserve timing of direct sound
            else:
                dt = 0

            print(f"\nRendering echogram: Source {ns} - Receiver {nr}")

            tempIR = torch.zeros((L_rir, nBands))

            for nb in range(nBands):
                tempIR[:, nb] = render_rir(
                    echograms[ns, nr, nb], endtime, fs, FRACTIONAL, RAND_IMS, dt
                )

            print("Filtering and combining bands")

            #if band_centerfreqs is None:
            if not torch.any(band_centerfreqs):
                rirs[:, nr, ns] = tempIR
            else:
                rirs[:, nr, ns] = filter_rir(tempIR, band_centerfreqs, fs)

    return rirs
