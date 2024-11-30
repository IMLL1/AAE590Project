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
    def __init__(self, R=None, planar=False):
        if R is None:
            R = np.diag([100e-3, 100e-3] if planar else [100e-3, 100e-3, 100e-3]) ** 2
        super().__init__(R, planar)

    def get_measurement(self, t, x, noise=False):
        z = x[:2] if self.planar else x[:3]
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))

        return z


class RangeMeas(MeasurementModel):
    def __init__(self, R=None, observer=None, planar=False):
        if R is None:
            R = np.array([[15e-3]]) ** 2
        super().__init__(R, planar)
        if observer is None:
            observer = [0, 0] if self.planar else [0, 0, 0]

        self.rO = np.array(observer)

    def get_measurement(self, t, x, noise=False):
        pos = x[:2] if self.planar else x[:3]
        pos -= self.rO
        z = np.array([np.linalg.norm(pos)])
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z


class RangeAndRate(MeasurementModel):
    def __init__(self, R=None, observer=None, planar=False):
        if R is None:
            R = np.diag([15e-3, 0.20e-3]) ** 2
        super().__init__(R, planar)
        if observer is None:
            observer = [0, 0] if self.planar else [0, 0, 0]

        self.rO = np.array(observer)

    def get_measurement(self, t, x, noise=False):
        z = np.zeros(2)
        r = x[:2] if self.planar else x[:3]
        r -= self.rO
        rmag = np.linalg.norm(r)
        z[0] = rmag
        v = x[2:4] if self.planar else x[3:6]
        vperp = v - np.dot(v, r / rmag)
        z[1] = np.linalg.norm(vperp / rmag)
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z


class DeclinationRA(MeasurementModel):
    def __init__(self, R=None, observer=None):
        if R is None:
            R = np.diag([5e-6, 5e-6]) ** 2
        super().__init__(R, planar=False)

        self.rO = np.zeros(3) if observer is None else np.array(observer)

    def get_measurement(self, t, x, noise=False):
        pos = x[:3] - self.rO

        RA = np.atan2(pos[1], pos[0])
        decl = np.atan2(pos[2], np.linalg.norm(pos))
        z = np.array([RA, decl])
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z


class RangeDeclinationRA(MeasurementModel):
    def __init__(self, R=None, observer=None):
        if R is None:
            R = np.diag([15e-3, 5e-6, 5e-6]) ** 2
        super().__init__(R, planar=False)

        self.rO = np.zeros(3) if observer is None else np.array(observer)

    def get_measurement(self, t, x, noise=False):
        pos = x[:3] - self.rO

        RA = np.atan2(pos[1], pos[0])
        decl = np.atan2(pos[2], np.linalg.norm(pos))
        rho = np.linalg.norm(pos)
        z = np.array([rho, RA, decl])
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z


class WalkerPseudorange(MeasurementModel):
    def __init__(self, R=None, i=0.9599, t: int = 24, p: int = 6, f=20, r_sats=26580):
        if R is None:
            R = np.diag([5e-3] * t) ** 2

        super().__init__(R, planar=False)
        npp = t // p
        df = 2 * np.pi * f / t
        thspace = np.linspace(0, 2 * np.pi, npp, False)
        R1 = lambda th: np.array(
            [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
        )
        R3 = lambda th: np.array(
            [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
        )
        pts = []
        for pn in range(p):
            r = np.array([r_sats, 0, 0])
            Omega = pn * 2 * np.pi / p
            thplane = thspace + df * pn
            for thta in thplane:
                pts.append(R3(thta) @ R1(i) @ R3(Omega) @ r)

        self.pts = pts
        self.n = t

        # generate position from walker

    def get_measurement(self, t, x, noise=False):
        z = np.array([np.linalg.norm(sat - x[:3]) for sat in self.pts])
        if noise:
            z += mvrn(np.zeros_like(z), self.get_R(t, x))
        return z
