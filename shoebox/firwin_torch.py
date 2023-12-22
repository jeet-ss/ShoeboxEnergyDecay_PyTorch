import torch

def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
    if nyq is None and fs is None:
        fs = 2
    elif nyq is not None:
        # if fs is not None:
        #     raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        # msg = ("Keyword argument 'nyq' is deprecated in favour of 'fs' and "
        #        "will be removed in SciPy 1.12.0.")
        # warnings.warn(msg, DeprecationWarning, stacklevel=3)
        fs = 2*nyq
    return fs


def firwin_torch(numtaps, cutoff, width=None, window='hamming', pass_zero=True,
           scale=True, nyq=None, fs=None, idx=None, device='cpu'):
    

    nyq = 0.5 * _get_fs(fs, nyq)

    cutoff = torch.atleast_1d(cutoff) / float(nyq)

    # # Check for invalid input.
    # if cutoff.ndim > 1:
    #     raise ValueError("The cutoff argument must be at most "
    #                      "one-dimensional.")
    # if cutoff.size == 0:
    #     raise ValueError("At least one cutoff frequency must be given.")
    # if cutoff.min() <= 0 or cutoff.max() >= 1:
    #     raise ValueError("Invalid cutoff frequency: frequencies must be "
    #                      "greater than 0 and less than fs/2.")
    # if np.any(np.diff(cutoff) <= 0):
    #     raise ValueError("Invalid cutoff frequencies: the frequencies "
    #                      "must be strictly increasing.")

    # if width is not None:
    #     # A width was given.  Find the beta parameter of the Kaiser window
    #     # and set `window`.  This overrides the value of `window` passed in.
    #     atten = kaiser_atten(numtaps, float(width) / nyq)
    #     beta = kaiser_beta(atten)
    #     window = ('kaiser', beta)

    if isinstance(pass_zero, str):
        if pass_zero in ('bandstop', 'lowpass'):
            if pass_zero == 'lowpass':
                if cutoff.size(0) != 1:
                    raise ValueError('cutoff must have one element if '
                                     'pass_zero=="lowpass", got %s'
                                     % (cutoff.size(),))
            elif cutoff.size(0) <= 1:
                raise ValueError('cutoff must have at least two elements if '
                                 'pass_zero=="bandstop", got %s'
                                 % (cutoff.size(),))
            pass_zero = True
        elif pass_zero in ('bandpass', 'highpass'):
            if pass_zero == 'highpass':
                if cutoff.size(0) != 1:
                    raise ValueError('cutoff must have one element if '
                                     'pass_zero=="highpass", got %s'
                                     % (cutoff.size(),))
            elif cutoff.size(0) <= 1:
                raise ValueError('cutoff must have at least two elements if '
                                 'pass_zero=="bandpass", got %s'
                                 % (cutoff.size(),))
            pass_zero = False
        else:
            raise ValueError('pass_zero must be True, False, "bandpass", '
                             '"lowpass", "highpass", or "bandstop", got '
                             '{}'.format(pass_zero))

    # pass_zero = bool(operator.index(pass_zero))  # ensure bool-like

    # pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    pass_nyquist = torch.bitwise_xor(torch.bitwise_and(cutoff.size(0), torch.tensor(1)).bool(), pass_zero)
    #if pass_nyquist and numtaps % 2 == 0:
    if torch.logical_and(pass_nyquist, torch.fmod(torch.tensor(numtaps), 2)==0):
        raise ValueError("A filter with an even number of coefficients must have zero response at the Nyquist frequency.")

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = torch.hstack((torch.tensor([0.0] * pass_zero).to(device=device), cutoff, torch.tensor([1.0] * pass_nyquist).to(device=device)))

    # `bands` is a 2-D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = torch.arange(0, numtaps).to(device=device) - alpha
    h = torch.zeros(m.size(0)).to(device=device)
    for left, right in bands:
        temp = torch.sinc(right * m)
        h += right * temp
        h -= left * torch.sinc(left * m)

    # Get and apply the window function.
    # from .windows import get_window
    # win = get_window(window, numtaps, fftbins=False)
    if window == 'hamming':
        win = torch.hamming_window(numtaps).to(device=device)
    else:
        raise ValueError("Only hamming window is Implemented")
    h = h * win
    #
    
    # Now handle scaling if desired.
    if scale:
        # Get the first passband.
        left, right = bands[0]
        if left == 0:
            scale_frequency = 0.0
        elif right == 1:
            scale_frequency = 1.0
        else:
            scale_frequency = 0.5 * (left + right)
        c = torch.cos(torch.pi * m * scale_frequency)
        s = torch.sum(h * c)
        h /= s

    return h
