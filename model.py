import torch
import matplotlib.pyplot as plt

from generate_StochasticRIR import generate_stochastic_rir
from generate_StochasticRIR_del import generate_stochastic_rir_del

# RiR model
class RIR_model(torch.nn.Module):
    def __init__(self, Kx=None, Ky=None, Kz=None, volume=None, noise_level=None, max_time=2.0, device='cpu') :
        super().__init__()
        if not [x for x in [Kx, Ky, Kz] if x is None]:
            self.Kx = torch.nn.Parameter(torch.tensor(Kx))
            self.Ky = torch.nn.Parameter(torch.tensor(Ky))
            self.Kz = torch.nn.Parameter(torch.tensor(Kz))            
        else:
            # Random Initialization
            self.Kx = torch.nn.Parameter(torch.randint(-4000, -100, (1,)).float()*0.0001)
            self.Ky = torch.nn.Parameter(torch.randint(-4000, -100, (1,)).float()*0.0001)
            self.Kz = torch.nn.Parameter(torch.randint(-4000, -100, (1,)).float()*0.0001)
        if volume is not None:
            self.V = torch.nn.Parameter(volume if torch.is_tensor(volume) else torch.tensor(volume))
        else:
            self.V = torch.nn.Parameter(torch.randint(100, 800, (1, )).float())
        if noise_level is not None:
            self.noise_level = torch.nn.Parameter(noise_level if torch.is_tensor(noise_level) else torch.tensor(noise_level))
        else:
            self.noise_level = torch.nn.Parameter(torch.randint(-900, -400, (1,)).float()*0.1)
        self.device = device
        self.maxTime = max_time

    def forward(self):
        def noise_hook(grad):
            clip_value = 1e+4
            #clip_neg = -0.5
            gradd = grad.clone()
            # print("INSIDE_GRAD", gradd, self.noise_level)
            return torch.clamp(gradd*10, min=-clip_value, max=clip_value)
        self.noise_level.register_hook(noise_hook)
        def vol_hook(grad):
            clip_value = 1e+5
            clip_neg = 0
            gradd = grad.clone()
            #print("INSIDE_GRAD", gradd, torch.clamp(gradd*10000, min=-clip_value , max=clip_value),  self.V)
            #print("INSIDE_GRAD", gradd, gradd*10000 ,  self.V)
            return torch.clamp(gradd*1000, min=clip_neg ,max=clip_value)
        self.V.register_hook(vol_hook)
        return generate_stochastic_rir(Kx=self.Kx, Ky=self.Ky, Kz=self.Kz,V=self.V, noise_level=self.noise_level,max_time=self.maxTime, device=self.device)
    
# RIR model 2 - for delta_K
class RIR_model_del(torch.nn.Module):
    def __init__(self, del_Kx=None, del_Ky=None, del_Kz=None, noise_level=None, max_time=2.0, device='cpu') : 
        super().__init__()
        if not [x for x in [del_Kx, del_Ky, del_Kz] if x is None]:
            # Reload model with given damping density coefficients
            self.del_Kx = torch.nn.Parameter(torch.tensor(del_Kx))
            self.del_Ky = torch.nn.Parameter(torch.tensor(del_Ky))
            self.del_Kz = torch.nn.Parameter(torch.tensor(del_Kz))
            #
            self.noise_level = torch.nn.Parameter(noise_level if torch.is_tensor(noise_level) else torch.tensor(noise_level))
        else:
            # Random Initialization
            self.kx = torch.randint(-2000, -100, (1,)).float()*0.0001
            self.ky = torch.randint(-2000, -100, (1,)).float()*0.0001
            self.kz = torch.randint(-2000, -100, (1,)).float()*0.0001
            self.kes = torch.sort(torch.cat([self.kx.view(-1), self.ky.view(-1), self.kz.view(-1)]))[0]
            self.del_Kx = torch.nn.Parameter(self.kes[0])
            self.del_Ky = torch.nn.Parameter(self.kes[1] - self.kes[0])
            self.del_Kz = torch.nn.Parameter(self.kes[2] - self.kes[1]) 
            #
            self.noise_level = torch.nn.Parameter(torch.randint(-1000, -600, (1,)).float()*0.01)
        self.device = device
        self.maxTime = max_time

    def forward(self):
        def my_hook(grad):
            clip_value = 1e+6
            gradd = grad.clone()
            # print("INSIDE_GRAD", gradd, self.noise_level)
            return torch.clamp(gradd*100, -clip_value)#, clip_value)
        self.noise_level.register_hook(my_hook)
        return generate_stochastic_rir_del(del_Kx=self.del_Kx, del_Ky=self.del_Ky, del_Kz=self.del_Kz,noise_level=self.noise_level, max_time=self.maxTime, device=self.device)
