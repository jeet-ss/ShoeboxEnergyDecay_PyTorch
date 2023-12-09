import torch
import numpy as np

def rec_moduleMic(echograms, mic_specs):
    """
    Apply receiver directivity to echogram.

    Parameters:
    - echograms: torch.Tensor, shape (nSrc, nRec), echogram structure
    - mic_specs: torch.Tensor, shape [x_1 y_1 z_1 a_1; ...; x_nRec y_nRec z_nRec a_nRec];
%               x,y,z unit vector showing the orientation of each mic
%               a=0~1 is the directivity coeff d(theta) = a + (1-a)*cos(theta)
%               a=1 is omni, a=0 is dipole, a=0.5 is cardioid, etc...
%               you can also leave empty for omni receivers

    Returns:
    - rec_echograms: torch.Tensor, shape (nSrc, nRec), updated echograms
    """

    nSrc = echograms.size(0)
    nRec = echograms.size(1)

    if mic_specs is None or mic_specs.numel() == 0:
        mic_specs = torch.ones(nRec, 4)

    nSpecs = mic_specs.size(0)

    if nRec != nSpecs:
        raise ValueError('The number of echograms should equal the number of receiver directivities.')
 
    mic_vecs = mic_specs[:, :-1]   ### might be some confusion
    mic_coeffs = mic_specs[:, -1]

    rec_echograms = echograms.clone()

    # Do nothing if all receivers are omnis
    if not torch.all(mic_coeffs == 1):
        for ns in range(nSrc):
            for nr in range(nRec):
                nRefl = len(echograms[ns, nr]['value'])  # confuse if this is dict ??!!

                # Get vectors from source to receiver
                rec_vecs = echograms[ns, nr]['coords']
                rec_vecs /= torch.sqrt(torch.sum(rec_vecs**2, dim=1, keepdim=True))

                mic_gains = mic_coeffs[nr] + (1 - mic_coeffs[nr]) * torch.dot(rec_vecs, ) #torch.einsum('ij,ij->i', rec_vecs, mic_vecs[nr])
                rec_echograms[ns, nr]['value'] = echograms[ns, nr]['value'] * mic_gains

    return rec_echograms
