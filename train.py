import torch
import matplotlib.pyplot as plt

from generateRIR import generate_stochasticRIR
from generate_StochasticRIR import generate_stochastic_rir
from model import RIR_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# class RIR_mod(torch.nn.Module):
#     def __init__(self,) :
#         super().__init__()
#         self.L1 = torch.nn.Parameter(torch.randint(2,4,(1,)).float())
#         self.L2 = torch.nn.Parameter(torch.randint(8,10,(1,)).float())
#         self.L3 = torch.nn.Parameter(torch.randint(6,9,(1,)).float())
#         #self.K = torch.nn.Parameter(torch.linspace(0.5,1,100))
#         #self.x = torch.linspace(-1, 1, 100)
        

#     def forward(self):
#         #return self.a*self.x + self.b
#         #return generate_stochasticRIR(L=torch.Tensor([self.L1, self.L2, self.L3]))
#         return generate_stochasticRIR(L=torch.cat((self.L1,self.L2,self.L3)))#.cuda()
    
# def env_maker(arr, filter_ = np.ones(4096)):
#     filter_avg = filter_/np.sum(filter_)
#     sqq = arr**2
#     xs = signal.convolve(sqq.detach(), filter_avg, mode='same')
#     in_db = control.mag2db(xs)
#     return torch.Tensor(in_db)
def mag2db(xcv):
    return 20 * xcv.log10_()

def env_maker0(arr, filter_=torch.ones(2047), gain=None, clip_=None):
    if gain is not None:
        arr = arr * (10**(gain/20))
    filter_avg = filter_/torch.sum(filter_)
    sqq = arr**2
    smoothed = torch.nn.functional.conv1d(sqq.reshape(1,-1), filter_avg.reshape(1,1,-1), bias=None, stride=1, padding='same' ).reshape(-1,)
    #in_db = mag2db(smoothed)
    if clip_ is not None:
        #
        in_db = torch.clip(in_db, min=clip_)
    return smoothed

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
    def pack_hook(x):
        print("Packing", x.size(), x.grad_fn)
        return x

    def unpack_hook(x):
        if torch.any(torch.isnan(x)):
            print("Unpacking Nan: ", x.size(), x.grad_fn)
        else:
            print("Unpacking: ", x.size(), x.grad_fn)
        return x
    
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