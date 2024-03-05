import torch
#import numpy as np
#import cmath



def analytic_damping_density(sigma, K, V, device='cpu'):
    ###
    # sigma is also carrying grad
    ###
    Kx, Ky, Kz = K
    # if Kx.requires_grad : Kx.register_hook(lambda x : print("Kx (in analytic): ", Kx.grad_fn,Kx.data, torch.isnan(x)))
    eps=2.2204e-20

    A = -(Kx**2 + Ky**2 + Kz**2)
    # if A.requires_grad : A.register_hook(lambda x : print("A: ",x, A.grad_fn,A.data, torch.any(torch.isnan(x))))
    B = 2 * sigma * Kz
    # if B.requires_grad : B.register_hook(lambda x : print("B: ",x[:10], B.grad_fn,B.data[:10],torch.any(torch.isnan(x))))
    C = Kx**2 + Ky**2 - sigma**2
    # if C.requires_grad : C.register_hook(lambda x : print("C: ",x[:10], C.grad_fn,C.data[:10],torch.any(torch.isnan(x))))
    tmpy = (B**2 - 4 * A * C)
    # remove negative numbers
    #delta = torch.Tensor(np.real([cmath.sqrt(it) for it in tmpy])) #torch.sqrt(B**2 - 4 * A * C)
    delta = torch.sqrt(torch.clamp(tmpy, min=torch.tensor(eps).to(device=device)) )
    # if delta.requires_grad : delta.register_hook(lambda x : print("delta: ",x[:10], delta.grad_fn,delta.data[:10],torch.any(torch.isnan(x))))

    a0 = -torch.sqrt(Kx**2 + Ky**2)
    a1 = Kx
    a2 = Ky
    b = Kz
    c = sigma
    function_p = lambda a: torch.clip((torch.Tensor([[1],[-1]]).to(device=device)*torch.acos(torch.clamp((-c / torch.sqrt(a**2 + b**2)), min=torch.tensor(-0.99999).to(device=device), max=torch.tensor(0.99999).to(device=device)))) - torch.atan(-a/b), min=0, max=torch.pi/2)

    p0 = function_p(a0) #func_p(a0, b , c)
    p1 = function_p(a1) #func_p(a1, b , c)
    p2 = function_p(a2) #func_p(a2, b , c)
    
    function_F = lambda phi: -(1/torch.sqrt(-A)) * torch.asin(torch.clamp(((2 * A * torch.cos(phi) + B)/ (delta + eps)), min=torch.tensor(-0.99999).to(device=device), max=torch.tensor(0.99999).to(device=device)))
    Fint = lambda p_up, p_low:(function_F(p_up) - function_F(p_low))
 
    H0 = Fint(p0[1, :], p0[0, :])
    H1 = Fint(p1[1, :], p1[0, :])
    H2 = Fint(p2[1, :], p2[0, :])
    # if H1.requires_grad : H1.register_hook(lambda x : print("H1: ",x[:10], H1.grad_fn, H1.data[:10], torch.any(torch.isnan(x))))

    #H = (8 / (4 * torch.pi * V)) * (2 * H0 - H1 - H2)
    H = (8 / (V+eps)) * (2 * H0 - H1 - H2)

    # only as return values
    p = torch.transpose(torch.vstack([p0, p1, p2]), 0,1)
    return H, p


