import numpy as np


def scale_signal(x, out_range=(-1, 1)):
    if np.sum(x > 1) > 0:
        return
    domain = 0, 1
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
