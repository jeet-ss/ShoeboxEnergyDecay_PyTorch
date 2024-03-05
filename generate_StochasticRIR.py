import torch

from util.applyPressureSource import apply_pressure_source
from util.analyticDampingDensity import analytic_damping_density
from optimizer_utils.helpers import db2mag, eps


def generate_stochastic_rir(Kx, Ky, Kz, V=torch.prod(torch.tensor([3,4,5])), noise_level=torch.tensor(-100), fs=48000, c=343, max_time=2.0, use_pressure_source=False, device='cpu', dual_output=False):
    # Predefine L 
    c = 343
    # if Ky.requires_grad : Ky.register_hook(lambda x : print("Ky: ", Ky.grad_fn,Ky.data, torch.isnan(x)))
    Kyxz = torch.concatenate((Kx.view(-1,1), Ky.view(-1,1), Kz.view(-1,1)))
    Kxyz = torch.minimum(Kyxz, torch.tensor(-0.0001))  # limit Kxyz to avoid division by 0
    # if Kxyz.requires_grad : Kxyz.register_hook(lambda x : print("Kxyz: ", x,Kxyz.grad_fn, Kxyz.data ,torch.any(torch.isnan(x))))

    max_sigma = torch.max(Kxyz)
    min_sigma = -torch.norm(Kxyz)
    sigma = torch.linspace(min_sigma.item() - 0.01, max_sigma.item() + 0.01, 1000).view(1,-1).to(device=device)
    
    h, p = analytic_damping_density(sigma, Kxyz, V, device=device)
    H = h# / (4 * torch.pi)  

    time = torch.transpose(torch.arange(1, max_time * fs + 1).view(1,-1), 0,1).to(device=device) / fs
    envelope = db2mag(noise_level) + torch.sqrt(torch.clamp( torch.exp(c * time * sigma) @ torch.transpose(torch.conj(H.view(1,-1)), 0, 1) * torch.mean(torch.diff(sigma)), min=torch.tensor(eps).to(device=device)) )
    #torch.exp(noise_level)
    
    #return envelope
    h1 = envelope * torch.randn(time.size()).to(device=device)
    if dual_output:
        return envelope, h1.squeeze()
    else:
        return h1.squeeze()