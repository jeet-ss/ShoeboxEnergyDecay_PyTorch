import torch
#from scipy.signal import fftconvolve, firwin
#import matplotlib.pyplot as plt

from shoebox.firwin_torch import firwin_torch
#from shoebox.firwin_scipy import firwin_scipy

def filter_rir(rir, f_center, fs):
    nBands = rir.shape[1]

    if len(f_center) != nBands:
        raise ValueError('The number of bands should match the number of columns in the 2nd dimension of rir')

    if nBands == 1:
        return rir
    else:
        # order of filters
        order = 1000
        filters = torch.zeros(order + 1, nBands)

        for i in range(nBands):
            if i == 0:
                fl = 30
                fh = (f_center[i] * f_center[i + 1]).sqrt()
                #w = [fl / (fs / 2), (fh / (fs / 2))]
                #filters[:, i] = torch.tensor(firwin_scipy(order + 1, w, pass_zero='bandpass', fs=fs))
                w = torch.tensor([fl / (fs / 2), fh / (fs / 2)])
                filters[:, i] = firwin_torch(order + 1, w, pass_zero='bandpass', fs=fs, idx = i)
                ### confusion in order + 1 in last line 
            elif i == nBands - 1:
                fl = (f_center[i] * f_center[i - 1]).sqrt()
                #w = [fl / (fs / 2)]
                #filters[:, i] = torch.tensor(firwin_scipy(order + 1, w, pass_zero='highpass', fs=fs, idx = i))
                w = torch.tensor([fl / (fs / 2)])
                filters[:, i] = firwin_torch(order + 1, w, pass_zero='highpass', fs=fs, idx = i)
            else:
                fl = (f_center[i] * f_center[i - 1]).sqrt()
                fh = (f_center[i] * f_center[i + 1]).sqrt()
                #w = [fl / (fs / 2), fh / (fs / 2)]
                #filters[:, i] = torch.tensor(firwin_scipy(order + 1, w, pass_zero='bandpass', fs=fs, idx = i))
                w = torch.tensor([fl / (fs / 2), fh / (fs / 2)])
                filters[:, i] = firwin_torch(order + 1, w, pass_zero='bandpass', fs=fs, idx = i)

        temp_rir = torch.cat([rir, torch.zeros(order, nBands)])

        rir_filt = torch.zeros(rir.size(0))
        for j in range(nBands):
            #yx = fftconvolve(temp_rir[:, j].numpy(), filters[:, j].numpy(), mode='valid')
            xy = torch.nn.functional.conv1d(temp_rir[:, j].reshape(1, -1), filters[:, j].reshape(1,1,-1), stride=1, padding='valid', bias=None).squeeze()
            rir_filt = torch.vstack((rir_filt, xy))

        rir_filt = rir_filt[1:,:].T
            
            

        ### maybe wrong the next line
        #rir_filt = torch.tensor([fftconvolve(temp_rir[:, j].numpy(), filters[:, j].numpy(), mode='valid') for j in range(nBands)], requires_grad=True).T
        #rir_filt = torch.tensor([torch.nn.functional.conv1d(temp_rir[:, j].reshape(1, -1), filters[:, j].reshape(1,1,-1), stride=1, padding=0, bias=None).squeeze() for j in range(nBands)], requires_grad=True).T
        rir_full = rir_filt.sum(dim=1)

        return rir_full

# Example usage:
# rir_full_result = filter_rir(torch.tensor(rir), torch.tensor(f_center), fs)
