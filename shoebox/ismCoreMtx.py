import torch
import torch.nn.functional as F

def ims_coreMtx(room, source, receiver, type, typeValue):
    # INITIALIZE ENVIRONMENT
    l, w, h = room
    src = torch.zeros_like(source)
    rec = torch.zeros_like(receiver)

    # validate source coords
    if torch.any(source < 0 ):
        raise ValueError('Source coordinates should be a positive number')
    if source[0]>l or source[1]>w or source[2]>h:
        raise ValueError('Source coordinates out of bounds')
    
    # move origin to the centre of the room
    src[0] = source[0] - l/2
    src[1] = w/2 - source[1]
    src[2] = source[2] - h/2

    # validate receiver coords
    if torch.any(receiver < 0 ):
        raise ValueError('receiver coordinates should be a positive number')
    if receiver[0]>l or receiver[1]>w or receiver[2]>h:
        raise ValueError('receiver coordinates out of bounds')
    
    # move origin to the centre of the room
    rec[0] = receiver[0] - l/2
    rec[1] = w/2 - receiver[1]
    rec[2] = receiver[2] - h/2

    if type.lower() == 'maxorder':
        maxOrder = typeValue
        echogram = ims_coreN(room, src, rec, maxOrder)
    elif type.lower() == 'maxtime':
        maxDelay = typeValue
        echogram = ims_coreT(room, src, rec, maxDelay)
    else:
        raise ValueError('Wrong type of ims calculation. Type should be either "maxOrder" or "maxTime"')

    # Sort reflections according to propagation time
    echogram['time'], idx = torch.sort(echogram.time, descending=False)
    echogram['value'] = torch.gather(echogram.value, 0, idx)
    echogram['order'] = torch.gather(echogram.order, 0, idx)
    echogram['coords'] = torch.gather(echogram.coords, 0, idx)

    return echogram

def ims_coreN(room, src, rec, N):
    # Speed of sound
    c = 343.0

    # i, j, k indices for calculation in x, y, z respectively
    I, J, K = torch.meshgrid(torch.arange(-N, N + 1), torch.arange(-N, N + 1), torch.arange(-N, N + 1))
    I, J, K = I.flatten(), J.flatten(), K.flatten()

    # Compute total order and select only valid indices up to order N
    s_ord = torch.abs(I) + torch.abs(J) + torch.abs(K)
    valid_indices = s_ord <= N
    I, J, K = I[valid_indices], J[valid_indices], K[valid_indices]

    # Image source coordinates with respect to receiver
    s_x = I * room[0] + (-1) ** I * src[0] - rec[0]
    s_y = J * room[1] + (-1) ** J * src[1] - rec[1]
    s_z = K * room[2] + (-1) ** K * src[2] - rec[2]

    # Distance
    s_d = torch.sqrt(s_x ** 2 + s_y ** 2 + s_z ** 2)

    # Reflection propagation time
    s_t = s_d / c

    # Reflection propagation attenuation - if distance is <1m, set attenuation at 1 to avoid amplification
    s_att = torch.zeros_like(s_d)
    s_att[s_d <= 1] = 1
    s_att[s_d > 1] = 1.0 / s_d[s_d > 1]

    # Write to echogram structure
    reflections = {
        'value': s_att,
        'time': s_t,
        'order': torch.stack([I, J, K], dim=1),
        'coords': torch.stack([s_x, s_y, s_z], dim=1)
    }

    return reflections

def ims_coreT(room, src, rec, maxTime):
    # Speed of sound
    c = 343.0

    # Find order N that corresponds to maximum distance
    d_max = maxTime * c
    Nx = torch.ceil(d_max / room[0]).int().item()
    Ny = torch.ceil(d_max / room[1]).int().item()
    Nz = torch.ceil(d_max / room[2]).int().item()

    # i, j, k indices for calculation in x, y, z respectively
    I, J, K = torch.meshgrid(torch.arange(-Nx, Nx + 1), torch.arange(-Ny, Ny + 1), torch.arange(-Nz, Nz + 1))
    I, J, K = I.flatten(), J.flatten(), K.flatten()

    # Image source coordinates with respect to receiver
    s_x = I * room[0] + (-1) ** I * src[0] - rec[0]
    s_y = J * room[1] + (-1) ** J * src[1] - rec[1]
    s_z = K * room[2] + (-1) ** K * src[2] - rec[2]

    # Distance
    s_d = torch.sqrt(s_x ** 2 + s_y ** 2 + s_z ** 2)

    # Bypass image sources with d > d_max
    valid_indices = s_d < d_max
    I, J, K = I[valid_indices], J[valid_indices], K[valid_indices]
    s_x, s_y, s_z = s_x[valid_indices], s_y[valid_indices], s_z[valid_indices]
    s_d = s_d[valid_indices]

    # Reflection propagation time
    s_t = s_d / c

    # Reflection propagation attenuation - if distance is <1m, set attenuation at 1 to avoid amplification
    s_att = torch.zeros_like(s_d)
    s_att[s_d <= 1] = 1
    s_att[s_d > 1] = 1.0 / s_d[s_d > 1]

    # Write to echogram structure
    reflections = {
        'value': s_att,
        'time': s_t,
        'order': torch.stack([I, J, K], dim=1),
        'coords': torch.stack([s_x, s_y, s_z], dim=1)
    }

    return reflections
