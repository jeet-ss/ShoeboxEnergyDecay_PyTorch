import torch
#from scipy.stats import norm

def apply_pressure_source(ism):
    len_ = 10000
    M = torch.ceil(len_ / 2).int()
    H1 = 1j * torch.linspace(0, 1, M)
    H = torch.cat([H1, torch.conj(H1[1:]).flip(0)])
    h = torch.fft.ifft(H).float()
    h = torch.roll(h, M, dims=0)
    h = h[(torch.arange(-100, 101) + len_ // 2).int()]

    h /= torch.norm(h)

    ism2 = torch.conv1d(ism.unsqueeze(0), h.unsqueeze(0), padding=(len_ - 1) // 2)[0]

    return ism2