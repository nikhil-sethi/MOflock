import numpy as np


def update_corr(x):
    """correction equation for the interval update. weight and bias will depend on CPU"""
    return np.exp(-2 * np.log10(x) + 2)


def brake_decay(r, a, p):
    if r <= 0:
        return 0
    elif 0 < (r * p )< (a / p):
        return r * p
    else:
        return np.sqrt((2 * a * r) - (a / p) ** 2)
