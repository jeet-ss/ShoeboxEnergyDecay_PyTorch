import torch
import numpy as np

def lagrange(N, delays):
    """
    Returns an order N FIR filter h which implements given delay (in samples).

    Parameters:
    - N: int, order of the FIR filter
    - delays: torch.Tensor, delays in samples

    Returns:
    - h: torch.Tensor, FIR filter coefficients
    """

    n = torch.arange(0, N+1)
    h = torch.ones(N+1, len(delays))

    for l in range(len(delays)):
        for k in range(N+1):
            index = np.argwhere(n != k)
            if np.ndim(index) > 1:
                index = index.squeeze()
            h[index, l] *= (delays[l] - k) / (n[index] - k)

    return h

