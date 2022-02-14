import numpy as np

__all__ = ['Grid']

class Grid(object):
    def __init__(self, n, d, lb=0, ub=1):
        self.n = n
        self.d = d
        R = lb + (ub - lb) * np.arange(0, n) / (n - 1)
        Xs = np.meshgrid(*[R] * d)
        self.coords = np.stack(Xs, axis=-1)