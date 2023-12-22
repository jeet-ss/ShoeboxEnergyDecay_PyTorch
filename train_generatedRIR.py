import torch
import matplotlib.pyplot as plt

from generateRIR import generate_stochasticRIR
from generate_StochasticRIR import generate_stochastic_rir
from model import RIR_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Trainer():
    def __init__(self, model_, k_vals, betas, lengths):
        pass

    



if __name__ == '__main__':
    ##
    #torch.manual_seed(2)
    labels = generate_stochasticRIR(L=torch.Tensor([4,3,3]))#.cuda())#.cuda()
    # labels = generate_stochastic_rir(Kx=torch.tensor(-0.012),Ky=torch.tensor( -0.016),Kz=torch.tensor( -0.018))
    #l_env = torch.Tensor(ob.envelope(labels.detach())).float().requires_grad_().cuda()
    #l_env = env_maker(labels).float().requires_grad_().cuda()
    #
    mod = RIR_model()#.cuda()
    crit = torch.nn.MSELoss()#.cuda()
    optim = torch.optim.SGD(mod.parameters(),lr=0.001)
    print("Inital Params: \n")
    for name, param in mod.named_parameters():
        if param.requires_grad:
            print( name,': ', param.data)
    # plt.figure(1)
    # plt.plot(labels)
    # #plt.show()
    # y_env = env_maker0(labels)
    # plt.figure(2)
    # plt.plot(y_env)
    #plt.show()
    
    t_l = []
    for i in range(50):
        optim.zero_grad()
        y_hat = mod.forward()
        # x_env = env_maker0(y_hat)
        # y_env = env_maker0(labels)
        l = torch.sum(torch.abs(y_hat - labels))
        # l = crit(labels, y_hat)
        #yy = y_hat.detach().cpu()
        #env_y = ob.envelope(yy).float().requires_grad_().cuda()
        #env_y = env_maker(yy).float().requires_grad_().cuda()
        # l = crit(x_env, y_env)
        l.backward()
        # mod.float()
        optim.step()
        t_l.append(l.detach().cpu())
        if i%2 == 0:print(f'Loss in epoch:{i} is : {l.detach()}')

    
    print("Updated Params: \n")
    for name, param in mod.named_parameters():
        if param.requires_grad:
            print( name,': ', param.data)

    #
    # plt.figure(2)
    # plt.plot(t_l)
    plt.show()