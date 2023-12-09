import torch
from shoebox.ismCoreMtx import ims_coreMtx
from shoebox.recModuleMic import rec_moduleMic
from shoebox.absorptionModule import absorption_module

def compute_echograms_mics(room, src, rec, abs_wall, limits, mic_specs=None):
    nRec = rec.size(0)
    nSrc = src.size(0)
    echograms = torch.zeros(nSrc, nRec, limits)
    abs_echograms = torch.zeros(nSrc, nRec, limits)
    rec_echograms = torch.zeros(nSrc, nRec, limits)

    # limit the RIR by reflection order or by time-limit
    type = 'maxTime'

    # compute echogram due to pure propagation (frequency-independent)
    for ns in range(nSrc):
        for nr in range(nRec):
            print('')
            print(f'Compute echogram: Source {ns + 1} - Receiver {nr + 1}')
            # compute echogram
            echograms[ns, nr] = ims_coreMtx(room, src[ns], rec[nr], type, limits.max().item())

    # apply receiver directivities (this can be omitted too if all omni sensors)
    if mic_specs is not None:
        print('Apply receiver directivities')
        rec_echograms = rec_moduleMic(echograms, mic_specs)
    else:
        rec_echograms = echograms

    # apply boundary absorption
    for ns in range(nSrc):
        for nr in range(nRec):
            print('')
            print(f'Apply absorption: Source {ns + 1} - Receiver {nr + 1}')
            # compute echogram
            abs_echograms[ns, nr] = absorption_module(rec_echograms[ns, nr], abs_wall, limits)

    return abs_echograms, rec_echograms, echograms

# # Mocking ims_coreMtx and rec_moduleMic functions since they are not provided
# def ims_coreMtx(room, src, rec, type, limit):
#     return torch.rand(limit)

# def rec_moduleMic(echograms, mic_specs):
#     return echograms  # Dummy implementation, replace with actual code
