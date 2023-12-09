import torch
#from scipy.stats import norm

from util.applyPressureSource import apply_pressure_source
from util.analyticDampingDensity import analytic_damping_density



def stochastic_rir(max_time, beta, L, c, fs, use_pressure_source=False):
    ####
    # Ask about parser in this position
    ###

    V = torch.prod(L)

    Kx = torch.log(torch.prod(beta[0:2])) / L[0]
    Ky = torch.log(torch.prod(beta[2:4])) / L[1]
    Kz = torch.log(torch.prod(beta[4:6])) / L[2]

    #Kxyz = torch.min(torch.tensor([Kx, Ky, Kz]), torch.tensor(-0.0001))  # limit Kxyz to avoid division by 0
    Kyxz = torch.concatenate((Kx.view(-1), Ky.view(-1), Kz.view(-1)))
    Kxyz = torch.minimum(Kyxz, torch.tensor(-0.0001))
    #Kxyz = torch.min(Kyxz, torch.tensor(-0.0001))

    max_sigma = torch.max(Kxyz)
    min_sigma = -torch.norm(Kxyz)
    sigma = torch.linspace(min_sigma.item() - 0.01, max_sigma.item() + 0.01, 1000)

    H, p = analytic_damping_density(sigma, Kxyz, V)
    H /= (4 * torch.pi)  # this is an unexplained tuning factor

    # uniform sampling of damping density for the decay envelope
    time = torch.arange(1, max_time * fs + 1) / fs
    envelope = torch.sqrt(torch.exp(c * time.unsqueeze(1) * sigma) @ H * torch.mean(torch.diff(sigma)))

    # shape noise
    #h = envelope * torch.tensor(norm.rvs(size=(len(time),)))
    h = envelope * torch.randn(len(time))

    # apply pressure source to match the color, but compensate the energy loss
    if use_pressure_source:
        h = apply_pressure_source(h)

    return h, envelope, H, sigma