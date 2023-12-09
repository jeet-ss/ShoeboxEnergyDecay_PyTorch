import torch

from util.applyPressureSource import apply_pressure_source
from util.analyticDampingDensity import analytic_damping_density

def generate_stochastic_rir(Kx, Ky, Kz, fs=48000, c=343, use_pressure_source=False):
#(max_time, beta, L, c, fs, use_pressure_source=False):
    # Predefine L 
    L = torch.tensor([3,4,5])
    V = torch.prod(L)
    # 
    max_time = 2.0
    fs = 48000
    c = 343
    # if Ky.requires_grad : Ky.register_hook(lambda x : print("Ky: ", Ky.grad_fn,torch.isnan(x)))
    Kyxz = torch.concatenate((Kx.reshape(-1,1), Ky.reshape(-1,1), Kz.reshape(-1,1)))
    Kxyz = torch.minimum(Kyxz, torch.tensor(-0.0001))  # limit Kxyz to avoid division by 0
    # if Kxyz.requires_grad : Kxyz.register_hook(lambda x : print("Kxyz: ", Kxyz.grad_fn,torch.any(torch.isnan(x))))

    max_sigma = torch.max(Kxyz)
    min_sigma = -torch.norm(Kxyz)
    sigma = torch.linspace(min_sigma.item() - 0.01, max_sigma.item() + 0.01, 1000)
    
    H, p = analytic_damping_density(sigma, Kxyz, V)
    H = H / (4 * torch.pi)  # this is an unexplained tuning factor
    # if H.requires_grad : H.register_hook(lambda x : print("H", H.grad_fn,torch.any(torch.isnan(x))))
    # uniform sampling of damping density for the decay envelope
    time = torch.arange(1, max_time * fs + 1) / fs
    envelope = torch.sqrt(torch.exp(c * time.unsqueeze(1) * sigma) @ H * torch.mean(torch.diff(sigma)))

    # shape noise
    h = envelope * torch.randn(len(time))
    # if h.requires_grad : h.register_hook(lambda x : print("h", h.grad_fn, torch.any(torch.isnan(x))))
    # apply pressure source to match the color, but compensate the energy loss
    if use_pressure_source:
        h = apply_pressure_source(h)

    return h#, envelope, H, sigma