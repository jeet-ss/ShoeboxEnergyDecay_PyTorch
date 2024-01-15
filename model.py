import torch
import matplotlib.pyplot as plt

from generate_StochasticRIR import generate_stochastic_rir
from generate_StochasticRIR_del import generate_stochastic_rir_del

# RiR model
class RIR_model(torch.nn.Module):
    def __init__(self, Kx=None, Ky=None, Kz=None, device='cpu') :
        super().__init__()

        if not [x for x in [Kx, Ky, Kz] if x is None]:
            # Reload model with given damping density coefficients
            self.Kx = torch.nn.Parameter(torch.tensor(Kx))
            self.Ky = torch.nn.Parameter(torch.tensor(Ky))
            self.Kz = torch.nn.Parameter(torch.tensor(Kz))            
        else:
            # Random Initialization
            self.Kx = torch.nn.Parameter(torch.randint(-2000, -100, (1,)).float()*0.0001)
            self.Ky = torch.nn.Parameter(torch.randint(-2000, -100, (1,)).float()*0.0001)
            self.Kz = torch.nn.Parameter(torch.randint(-2000, -100, (1,)).float()*0.0001)
        #
        self.device = device
        

    def forward(self):
        return generate_stochastic_rir(Kx=self.Kx, Ky=self.Ky, Kz=self.Kz, device=self.device)
    
# RIR model 2 - for delta_K
class RIR_model_del(torch.nn.Module):
    def __init__(self, del_Kx=None, del_Ky=None, del_Kz=None, noise_level=None, device='cpu') : 
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

    def forward(self):
        def my_hook(grad):
            clip_value = 1e+6
            gradd = grad.clone()
            # print("INSIDE_GRAD", gradd, self.noise_level)
            return torch.clamp(gradd*100, -clip_value)#, clip_value)
        self.noise_level.register_hook(my_hook)
        return generate_stochastic_rir_del(del_Kx=self.del_Kx, del_Ky=self.del_Ky, del_Kz=self.del_Kz,noise_level=self.noise_level, device=self.device)
