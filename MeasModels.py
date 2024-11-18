from numpy.random import multivariate_normal as mvrn
from scipy.optimize import approx_fprime
import numpy as np


class MeasurementModel:
    def __init__(self, R, planar=False):
        self.R = R
        self.planar = planar

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
        z = x[:2] if self.planar else x[:3]
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))

        return z


class RangeMeas(MeasurementModel):
    def get_measurement(self, t, x, noise=False):
        pos = x[:2] if self.planar else x[:3]
        z = np.array([np.linalg.norm(pos)])
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z


class RangeAndRArate(MeasurementModel):
    def get_measurement(self, t, x, noise=False):
        z = np.zeros(2)
        r = x[:2] if self.planar else x[:3]
        rmag = np.linalg.norm(r)
        z[0] = rmag
        v = x[2:4] if self.planar else x[3:6]
        vperp = v - np.dot(v, r / rmag)
        z[1] = np.linalg.norm(vperp / rmag)
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z


class DeclinationRA(MeasurementModel):
    def __init__(self, R, planar=False):
        super().__init__(R, planar)
        # cannot be planar
        assert not self.planar

    def get_measurement(self, t, x, noise=False):
        RA = np.atan2(x[1], x[0])
        Decl = np.atan2(x[2], np.linalg.norm(x[:2]))
        z = np.array([RA, Decl])
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z
