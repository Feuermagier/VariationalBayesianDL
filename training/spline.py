import torch

# From https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch

def h_poly(t):
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt


def interpolate(x, y, xs):
    x, sort_indxs = torch.sort(x)
    y = y[sort_indxs]
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx