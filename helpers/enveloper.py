import torch
import matplotlib.pyplot as plt

# Env maker
def envelope_generator(inp, filter_len=2047, gain=None, clip_=None, normalise=False, title_str = '', display_plots=False):
    # Amplify
    if gain is not None:
        arr = inp * (10**(gain/20))
    else:
        arr = inp

    ## Main Evelope Block
    filter_= torch.ones(filter_len)
    filter_avg = filter_/torch.sum(filter_) # Avg filter
    sqq = arr**2    # Squaring for Power
    # Smoothing 
    smoothed = torch.nn.functional.conv1d(sqq.reshape(1,-1), filter_avg.reshape(1,1,-1), bias=None, stride=1, padding='same' ).reshape(-1,)
    sm2 = smoothed.clone()
    in_db = mag2db(sm2)    # Convert to dB scale

    # Clipping below -150/-100dB
    if clip_ is not None:
        in_db2 = torch.clip(in_db, min=clip_)
    else:
        in_db2 = in_db

    # normalise around zero
    if normalise:
        normalised = in_db2 - torch.mean(in_db2)
    else:
        normalised = in_db2

    # Plot   
    if display_plots:
        title_switch = {
        # 1: "RIR",
        # 2: "Squared RIR",
        1: "Smoothed",
        2: "in dB", 
        3: "normalised" }
        #
        plt.figure(figsize=(15,4))
        for i, it in enumerate([smoothed, in_db2, normalised]):
            plt.subplot(1,3,i+1)
            plt.plot(it)
            plt.title(title_switch[i+1]+", Filter: "+title_str)
        plt.show()
    
    return normalised

def mag2db(xcv):
    return 20 * xcv.log10_()