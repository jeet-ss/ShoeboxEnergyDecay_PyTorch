import torch

eps=1e-30
# calculate Damping Coefficients, K Values, and Del_K values
def calculate_K(betas_a, lengths_a):
    ''' 
    betas: a 6x1 array of all reflection coefficients
    lengths: 3x1 array of all wall measurements
    '''
    betas = torch.sort(betas_a)[0]
    lengths = torch.sort(lengths_a)[0]
    #print(betas,lengths)
    Kx = torch.log(torch.prod(betas[0:2])) / lengths[0]
    Ky = torch.log(torch.prod(betas[2:4])) / lengths[1]
    Kz = torch.log(torch.prod(betas[4:6])) / lengths[2]
    return Kx, Ky, Kz

# Calculate del K values
def calculate_del_K(Kx, Ky, Kz):
    '''
    Calculates the incremental del values of Damping density coefficients
    It is done to preserve order of the coeeficients.
    '''
    kes = torch.sort(torch.cat([Kx.view(-1), Ky.view(-1), Kz.view(-1)]))[0]
    return kes[0], kes[1] - kes[0], kes[2] - kes[1]

def calculate_K_from_del(del_Kx, del_Ky, del_Kz):
    '''
    Calculate the damping density coefficients back from the incremental del values
    '''
    return [del_Kx, del_Kx + del_Ky, del_Kz + del_Ky + del_Kx]

def mag2db(xcv):
    return 20 * xcv.log10_()

def db2mag(xv):
    return 10**(xv/20)