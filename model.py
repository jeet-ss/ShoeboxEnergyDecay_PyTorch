import torch
import matplotlib.pyplot as plt

from generateRIR import generate_stochasticRIR
from generate_StochasticRIR import generate_stochastic_rir

class RIR_model(torch.nn.Module):
    def __init__(self,) :
        super().__init__()
        self.Kx = torch.nn.Parameter(torch.randint(-200, -100, (1,)).float()*0.0001)
        self.Ky = torch.nn.Parameter(torch.randint(-200, -100, (1,)).float()*0.0001)
        self.Kz = torch.nn.Parameter(torch.randint(-200, -100, (1,)).float()*0.0001)
        
        # self.L1 = torch.nn.Parameter(torch.randint(3,8,(1,)).float())
        # self.L2 = torch.nn.Parameter(torch.randint(3,8,(1,)).float())
        # self.L3 = torch.nn.Parameter(torch.randint(3,8,(1,)).float())
        

    def forward(self):
        # return generate_stochasticRIR(L=torch.cat((self.L1,self.L2,self.L3)))#.cuda()
        return generate_stochastic_rir(Kx=self.Kx, Ky=self.Ky, Kz=self.Kz)