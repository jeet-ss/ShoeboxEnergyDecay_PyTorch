import torch
import numpy as np

def absorption_module(echogram, alpha, limits=None):
    """
    Apply absorption coefficients to echogram.

    Parameters:
    - echogram: dict, echogram structure
    - alpha: torch.Tensor, absorption coefficients per frequency band and wall
    - limits: torch.Tensor or None, frequency limits for each band

    Returns:
    - abs_echograms: list of dicts, updated echograms per frequency band
    """

    Nbands = alpha.size(0)

    if limits is None:
        abs_echograms = [echogram.copy() for _ in range(Nbands)]
    else:
        abs_echograms = []
        for nb in range(Nbands):
            # max is taken to get the last index
            idx_limit = torch.max(np.argwhere(echogram['time'] < limits[nb])).item() 

            abs_echograms.append({
                'time': echogram['time'][:idx_limit],
                'value': echogram['value'][:idx_limit, :],
                'order': echogram['order'][:idx_limit, :],
                'coords': echogram['coords'][:idx_limit, :]
            })

    for nb in range(Nbands):
        # absorption coefficients for x, y, z walls per frequency
        a_x = alpha[nb, :2]
        a_y = alpha[nb, 2:4]
        a_z = alpha[nb, 4:]  # may end with 4:5
        # reflection coefficients
        r_x = torch.sqrt(1 - a_x)
        r_y = torch.sqrt(1 - a_y)
        r_z = torch.sqrt(1 - a_z)
        # split
        i = abs_echograms[nb]['order'][:, 0]
        j = abs_echograms[nb]['order'][:, 1]
        k = abs_echograms[nb]['order'][:, 2]

        i_even = i[i % 2 == 0]
        i_odd = i[i % 2 != 0]
        i_odd_pos = i_odd[i_odd > 0]
        i_odd_neg = i_odd[i_odd < 0]

        j_even = j[j % 2 == 0]
        j_odd = j[j % 2 != 0]
        j_odd_pos = j_odd[j_odd > 0]
        j_odd_neg = j_odd[j_odd < 0]

        k_even = k[k % 2 == 0]
        k_odd = k[k % 2 != 0]
        k_odd_pos = k_odd[k_odd > 0]
        k_odd_neg = k_odd[k_odd < 0]

        # find total absorption coefficients by calculating the
        # number of hits on every surface, based on the order per dimension
        abs_x = torch.zeros_like(abs_echograms[nb]['time'])
        abs_x[i % 2 == 0] = r_x[0]**(torch.abs(i_even) / 2) * r_x[1]**(torch.abs(i_even) / 2)
        abs_x[(i % 2 != 0) & (i > 0)] = r_x[0]**torch.ceil(i_odd_pos / 2) * r_x[1]**torch.floor(i_odd_pos / 2)
        abs_x[(i % 2 != 0) & (i < 0)] = r_x[0]**torch.floor(torch.abs(i_odd_neg) / 2) * r_x[1]**torch.ceil(torch.abs(i_odd_neg) / 2)

        abs_y = torch.zeros_like(abs_echograms[nb]['time'])
        abs_y[j % 2 == 0] = r_y[0]**(torch.abs(j_even) / 2) * r_y[1]**(torch.abs(j_even) / 2)
        abs_y[(j % 2 != 0) & (j > 0)] = r_y[0]**torch.ceil(j_odd_pos / 2) * r_y[1]**torch.floor(j_odd_pos / 2)
        abs_y[(j % 2 != 0) & (j < 0)] = r_y[0]**torch.floor(torch.abs(j_odd_neg) / 2) * r_y[1]**torch.ceil(torch.abs(j_odd_neg) / 2)

        abs_z = torch.zeros_like(abs_echograms[nb]['time'])
        abs_z[k % 2 == 0] = r_z[0]**(torch.abs(k_even) / 2) * r_z[1]**(torch.abs(k_even) / 2)
        abs_z[(k % 2 != 0) & (k > 0)] = r_z[0]**torch.ceil(k_odd_pos / 2) * r_z[1]**torch.floor(k_odd_pos / 2)
        abs_z[(k % 2 != 0) & (k < 0)] = r_z[0]**torch.floor(torch.abs(k_odd_neg) / 2) * r_z[1]**torch.ceil(torch.abs(k_odd_neg) / 2)

        s_abs_tot = abs_x * abs_y * abs_z
        abs_echograms[nb]['value'] = abs_echograms[nb]['value'] * (s_abs_tot * torch.ones(1, abs_echograms[nb]['value'].size(1) ))
        #(s_abs_tot.unsqueeze(1).expand_as(abs_echograms[nb]['value']))

    return abs_echograms
