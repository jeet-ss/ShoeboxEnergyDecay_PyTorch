import torch
from shoebox.firwin_torch import firwin_torch
import matplotlib.pyplot as plt

def rir_bands(rir, f_center=None, fs=48000, device='cpu'):
# separate RIR into Frequenc bands
    if not f_center:
        nBands = 6
        f_center = torch.zeros(nBands).to(device=device)
        f_center[0] = 125.0     # lowest octave band
        for it in range(1, nBands):
            f_center[it] =  2.0 * f_center[it-1]
    else:
        nBands = f_center.size(0)

    if nBands == 1:
        return rir
    else:
        # order of filters
        order = 1000
        filters = torch.zeros(order + 1, nBands).to(device=device)

        for i in range(nBands):
            if i == 0:
                fl = 30
                fh = (f_center[i] * f_center[i + 1]).sqrt()
                w = torch.tensor([fl / (fs / 2), fh / (fs / 2)]).to(device=device)
                filters[:, i] = firwin_torch(order + 1, w, pass_zero='bandpass',  idx = i, device=device)
            elif i == nBands - 1:
                fl = (f_center[i] * f_center[i - 1]).sqrt()
                w = torch.tensor([fl / (fs / 2)]).to(device=device)
                filters[:, i] = firwin_torch(order + 1, w, pass_zero='highpass',idx = i, device=device)
            else:
                fl = (f_center[i] * f_center[i - 1]).sqrt()
                fh = (f_center[i] * f_center[i + 1]).sqrt()
                w = torch.tensor([fl / (fs / 2), fh / (fs / 2)]).to(device=device)
                filters[:, i] = firwin_torch(order + 1, w, pass_zero='bandpass',  idx = i, device=device)
        # padding
        temp_rir = torch.cat([rir, torch.zeros(order).to(device=device)])

        #plot filters
        if False:
            plt.figure(4, figsize=(15,8))
            for it in range(filters.size(1)):
                plt.subplot(2,3,it+1)
                plt.plot(filters[:,it].cpu())

        rir_filt = torch.zeros(rir.size(0)).to(device=device)
        for j in range(nBands):
            xy = torch.nn.functional.conv1d(temp_rir.reshape(1, -1), filters[:, j].reshape(1,1,-1), stride=1, padding='valid', bias=None).squeeze()
            rir_filt = torch.vstack((rir_filt, xy))

        rir_filt = rir_filt[1:,:].T
        if False:
            plt.figure(40, figsize=(15,8))
            for it in range(rir_filt.size(1)):
                plt.subplot(2,3,it+1)
                plt.plot(rir_filt[:,it].cpu())
            
        return rir_filt