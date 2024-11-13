from numpy.random import multivariate_normal as mvrn
from scipy.optimize import approx_fprime
import numpy as np


class MeasurementModel:
    def __init__(self, R):
        self.R = R

    def get_R(self, t, x):
        R = self.R(t, x) if callable(self.R) else self.R
        return R

    def get_measurement(self, t, x, noise=False):
        z = x
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z

    def get_H(self, t, x):
        h = lambda state: self.get_measurement(t, state)
        H = approx_fprime(x, h)
        if len(np.shape(H)) == 1:
            H = np.atleast_2d(H)
            if np.shape(H)[1] != len(x):
                H = H.T
        return H


class PosMeas(MeasurementModel):
    def get_measurement(self, t, x, noise=False):
        z = x[:3]
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))

        return z


class RangeMeas(MeasurementModel):
    def get_measurement(self, t, x, noise=False):
        z = np.array([np.linalg.norm(x[:3])])
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z