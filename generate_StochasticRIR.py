import torch

from util.applyPressureSource import apply_pressure_source
from util.analyticDampingDensity import analytic_damping_density

def generate_stochastic_rir(Kx, Ky, Kz, fs=48000, c=343, use_pressure_source=False, device='cpu'):
#(max_time, beta, L, c, fs, use_pressure_source=False):
    # Predefine L 
    L = torch.tensor([3,4,5])
    V = torch.prod(L)
    eps=2.2204e-16
    # 
    max_time = 2.0
    fs = 48000
    c = 343
    # if Ky.requires_grad : Ky.register_hook(lambda x : print("Ky: ", Ky.grad_fn,Ky.data, torch.isnan(x)))
    Kyxz = torch.concatenate((Kx.view(-1,1), Ky.view(-1,1), Kz.view(-1,1)))
    Kxyz = torch.minimum(Kyxz, torch.tensor(-0.0001))  # limit Kxyz to avoid division by 0
    # if Kxyz.requires_grad : Kxyz.register_hook(lambda x : print("Kxyz: ", x,Kxyz.grad_fn, Kxyz.data ,torch.any(torch.isnan(x))))

    max_sigma = torch.max(Kxyz)
    min_sigma = -torch.norm(Kxyz)
    sigma = torch.linspace(min_sigma.item() - 0.01, max_sigma.item() + 0.01, 1000).to(device=device)
    
    H, p = analytic_damping_density(sigma, Kxyz, V, device=device)
    H = H / (4 * torch.pi)  # this is an unexplained tuning factor
    # if H.requires_grad : H.register_hook(lambda x : print("H",x[500:510], H.grad_fn,H.data[500:510],torch.any(torch.isnan(x))))
    # uniform sampling of damping density for the decay envelope
    time = torch.arange(1, max_time * fs + 1).to(device=device) / fs
    envelope = torch.sqrt(torch.clamp(torch.exp(c * time.unsqueeze(1) * sigma) @ H * torch.mean(torch.diff(sigma)), min=torch.tensor(eps).to(device=device)) )
    # if envelope.requires_grad : envelope.register_hook(lambda x : print("envelope",x, envelope.grad_fn, envelope.data,torch.any(torch.isnan(x))))
    # shape noise
    h = envelope * torch.randn(len(time)).to(device=device)
    # if h.requires_grad : h.register_hook(lambda x : print("h",x, h.grad_fn, h.data,torch.any(torch.isnan(x))))
    # apply pressure source to match the color, but compensate the energy loss
    if use_pressure_source:
        h = apply_pressure_source(h)

    return h#, envelope, H, sigma