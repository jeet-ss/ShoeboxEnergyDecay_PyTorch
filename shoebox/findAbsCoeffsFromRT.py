import torch
from scipy.optimize import minimize, fmin

def find_abs_coeffs_from_rt(room, rt60_target, abs_wall_ratios=None):
    if abs_wall_ratios is None:
        abs_wall_ratios = torch.ones(6)

    if torch.max(abs_wall_ratios) != 1:
        abs_wall_ratios /= torch.max(abs_wall_ratios)

    n_bands = len(rt60_target)
    rt60_true = torch.zeros_like(rt60_target)
    alpha_walls = torch.zeros((n_bands, 6))
    # funciton to optimize
    funct = lambda a: torch.abs(rt60 - get_rt_sabine(a, room, abs_wall_ratios))
    funct2 = lambda a: torch.abs(rt60 - get_rt_sabine(a, room.detach(), abs_wall_ratios))

    for nb in range(n_bands):
        rt60 = rt60_target[nb]
        x_min = fmin(funct2, 0.0001, disp=False)
        x_min = x_min[0]
        rt60_true[nb] = rt60 + funct(x_min)
        alpha_walls[nb, :] = torch.min(x_min * abs_wall_ratios, torch.ones_like(abs_wall_ratios))

    return alpha_walls, rt60_true


def get_rt_sabine(alpha, room, abs_wall_ratios):
    c = 343.0
    l, w, h = room
    V = l * w * h
    Stot = 2 * (l * w + l * h + w * h)
    alpha = alpha.item()
    alpha_walls = alpha * abs_wall_ratios
    a_x = alpha_walls[0:2]
    a_y = alpha_walls[2:4]
    a_z = alpha_walls[4:6]

    # mean absorption
    a_mean = torch.sum(w * h * a_x + l * h * a_y + l * w * a_z) / Stot
    rt60 = (55.25 * V) / (c * Stot * a_mean)

    return rt60


# # Example usage:
# room_dimensions = torch.tensor([4.0, 5.0, 3.0])
# target_rt60 = torch.tensor([1.0, 0.8, 0.7, 0.6, 0.5, 0.4]) * 2.0
# absorption_ratios = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])  # Adjust these ratios accordingly

# alpha_coeffs, true_rt60 = find_abs_coeffs_from_rt(room_dimensions, target_rt60, absorption_ratios)

# print("Alpha Coefficients:")
# print(alpha_coeffs)
# print("\nTrue RT60 Values:")
# print(true_rt60)
