import torch
import matplotlib.pyplot as plt

# Env maker
def rir_smoothing(inp, filter_len=2047,device='cpu'):
    ## Main Evelope Block
    filter_= torch.ones(filter_len).to(device=device)
    filter_avg = filter_/torch.sum(filter_) # Avg filter
    sqq = inp**2    # Squaring for Power
    # Smoothing 
    smoothed = torch.nn.functional.conv1d(sqq.reshape(1,-1), filter_avg.reshape(1,1,-1), bias=None, stride=1, padding='same' ).reshape(-1,)
    # if smoothed.requires_grad : smoothed.register_hook(lambda x : print("smoothed: ",x[-10:], smoothed.grad_fn,smoothed.data[200:210], torch.any(torch.isnan(x))))
    return smoothed

def envelope_generator(inp, gain=None, clip_=None, normalise=False, title_str = '', title_sup = '', display_plots=False, device='cpu'):
    eps=2.2204e-16
    # Amplify
    # if inp.requires_grad : inp.register_hook(lambda x : print("inp: ", x[200:210],inp.grad_fn,inp.data[200:210], torch.any(torch.isnan(x))))
    if gain is not None:
        arr = inp * (10**(gain/20))
    else:
        arr = inp
    # if arr.requires_grad : arr.register_hook(lambda x : print("arr: ",x[200:210], arr.grad_fn,arr.data[200:210],torch.any(torch.isnan(x))))
    ## Main Evelope Block
    # filter_= torch.ones(filter_len).to(device=device)
    # filter_avg = filter_/torch.sum(filter_) # Avg filter
    # sqq = arr**2    # Squaring for Power
    # # Smoothing 
    # smoothed = torch.nn.functional.conv1d(sqq.reshape(1,-1), filter_avg.reshape(1,1,-1), bias=None, stride=1, padding='same' ).reshape(-1,)
    # if smoothed.requires_grad : smoothed.register_hook(lambda x : print("smoothed: ",x[-10:], smoothed.grad_fn,smoothed.data[200:210], torch.any(torch.isnan(x))))
    sm2 = arr.clone() + eps
    # if sm2.requires_grad : sm2.register_hook(lambda x : print("sm2: ", x[-10:], sm2.grad_fn,sm2.data, torch.any(torch.isnan(x))))
    # if torch.any(sm2<=0) : print("negative in sm2",[it for it in sm2 if it<=0])
    in_db = mag2db(sm2)    # Convert to dB scale
    # if in_db.requires_grad : in_db.register_hook(lambda x : print("in_db: ", x,in_db.grad_fn,in_db.data, torch.any(torch.isnan(x))))

    # Clipping below -150/-100dB
    # if clip_ is not None:
    #     in_db2 = torch.clip(in_db, min=clip_)
    # else:
    #     in_db2 = in_db

    # normalise around zero
    if normalise:
        normalised = in_db - torch.mean(in_db)
    else:
        normalised = in_db
    # if normalised.requires_grad : normalised.register_hook(lambda x : print("normalised: ", x[:10], normalised.grad_fn,normalised.data[:10], torch.any(torch.isnan(x))))
    # Clipping below -150/-100dB
    if clip_ is not None:
        clipped_db = torch.clip(normalised, min=clip_)
    else:
        clipped_db = normalised
    # Plot   
    if display_plots:
        # if title_str == '':
        #     title_str = str(filter_len)
        title_switch = {
        1: "RIR",
        # 2: "Squared RIR",
        2: "Amplified",
        3: "in dB", 
        4: "normalised" ,
        5: "clipped"}
        if device == 'cuda':
            normalised_c = normalised.clone()
            llist = [inp.detach().cpu(), arr.detach().cpu(),in_db.detach().cpu(),normalised_c.detach().cpu(),clipped_db.detach().cpu()]
        else:
            llist = [inp, arr, clipped_db, normalised]
        #
        plt.figure(figsize=(25,4))
        for i, it in enumerate(llist):
            plt.subplot(1,5, i+1)
            plt.plot(it)
            plt.title(title_switch[i+1])#+", Filter: "+title_str)
            if title_sup != '':
                plt.suptitle(title_sup)
        plt.show()
    
    return normalised

def mag2db(xcv):
    return 20 * xcv.log10_()