import torch

from util.applyPressureSource import apply_pressure_source
from util.analyticDampingDensity import analytic_damping_density

def generate_stochastic_rir_del(del_Kx, del_Ky, del_Kz, noise_level=-0.001, fs=48000, c=343, use_pressure_source=False, device='cpu'):
    ###
    #
    # This function uses the difference of K's to predict in the correct order
    ####
    # 
    # if noise_level.requires_grad : noise_level.register_hook(lambda x : print("noise_level", noise_level, noise_level.grad_fn,torch.any(torch.isnan(x))))
    #noise_level = 0.0001
    # Predefine L 
    L = torch.tensor([3,4,5])
    V = torch.prod(L)
    # 
    eps = 2.2204e-16
    max_time = 2.0
    fs = 48000
    c = 343
    # derive K's
    Kx = del_Kx
    Ky = del_Ky + del_Kx
    Kz = del_Kz + del_Ky + del_Kx
    # if Ky.requires_grad : Ky.register_hook(lambda x : print("Ky: ", Ky.grad_fn,torch.isnan(x)))
    Kyxz = torch.concatenate((Kx.view(-1), Ky.view(-1), Kz.view(-1)))
    Kxyz = torch.minimum(Kyxz, torch.tensor(-0.0001))  # limit Kxyz to avoid division by 0
    # if Kxyz.requires_grad : Kxyz.register_hook(lambda x : print("Kxyz: ", Kxyz.grad_fn,torch.any(torch.isnan(x))))

    max_sigma = torch.max(Kxyz)
    min_sigma = -torch.norm(Kxyz)
    sigma = torch.linspace(min_sigma.item() - 0.01, max_sigma.item() + 0.01, 1000).to(device=device)
    
    H, p = analytic_damping_density(sigma, Kxyz, V, device=device)
    H = H / (4 * torch.pi)  # this is an unexplained tuning factor
    # if H.requires_grad : H.register_hook(lambda x : print("H", H.grad_fn,torch.any(torch.isnan(x))))
    # uniform sampling of damping density for the decay envelope
    time = torch.arange(1, max_time * fs + 1).to(device=device) / fs
    envv = torch.sqrt(torch.clamp(torch.exp(c * time.unsqueeze(1) * sigma) @ H * torch.mean(torch.diff(sigma)), min=torch.tensor(eps).to(device=device)) )
    envelope = envv #+ torch.exp(noise_level)# if torch.is_tensor(noise_level) else torch.tensor(noise_level)) # torch.exp()
    # if noise_level.requires_grad : noise_level.register_hook(lambda x : print("noise_level2", noise_level, noise_level.grad_fn,torch.any(torch.isnan(x))))
    # envelope = torch.sqrt(torch.exp(c * time.unsqueeze(1) * sigma) @ H * torch.mean(torch.diff(sigma)))
    # if envelope.requires_grad : envelope.register_hook(lambda x : print("envelope", envelope.grad_fn,torch.any(torch.isnan(x))))
    # shape noise
    #h = envelope * torch.randn(len(time)).to(device=device)
    # if h.requires_grad : h.register_hook(lambda x : print("h", h.grad_fn, torch.any(torch.isnan(x))))
    # apply pressure source to match the color, but compensate the energy loss
    if use_pressure_source:
        h = apply_pressure_source(h)

    return envelope