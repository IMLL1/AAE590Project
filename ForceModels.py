from numpy.random import multivariate_normal as mvrn
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
import numpy as np


def add_noise(dynamics, t, x, noisevec, tvec):
    idx = np.abs(t - tvec).argmin()
    Qsqrt = np.linalg.cholesky(dynamics.get_Q(t, x))
    w = Qsqrt @ noisevec[idx]
    return dynamics.G @ w


class DynamicsModel:
    def __init__(self, Q, G):
        self.Q = Q
        self.G = G

    def get_Q(self, t, x):
        Q = self.Q(t, x) if callable(self.Q) else self.Q
        return Q

    def get_F(self, t, x):
        f = lambda state: self.get_deriv(t, state)
        F = np.atleast_2d(approx_fprime(x, f))
        if len(np.shape(F)) == 1:
            F = np.atleast_2d(F)
        return F

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        if len(noise_t_vec) == 2:  # if noise
            dx += add_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])
        return dx

    def propagate_x(self, t, x, dt, noise_t_vec=()):
        f = lambda t, x: self.get_deriv(t, x, noise_t_vec)
        x1 = solve_ivp(f, [t, t + dt], x, t_eval=[t + dt]).y.flatten()
        return x1

    def get_truth(self, x0, tk, noise_t_vec=()):
        f = lambda t, x: self.get_deriv(t, x, noise_t_vec)
        xk = list(solve_ivp(f, [tk[0], tk[-1]], x0, t_eval=tk).y.T)
        return xk


class Kepler(DynamicsModel):
    def __init__(self, Q, G, mu):
        super().__init__(Q, G)
        self.mu = mu

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        r = x[:3]
        rmag = np.linalg.norm(r)
        dpos = x[3:6]
        accel = -r * self.mu / rmag**3
        dx[:6] = np.array([*dpos, *accel])

        if len(noise_t_vec) == 2:  # if noise
            dx += add_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])

        return dx


class KeplerPerurbed(DynamicsModel):
    def __init__(self, Q, G, mu, pert_vec):
        super().__init__(Q, G)
        self.mu = mu
        self.pert = np.array(pert_vec)
        # state: x y vx vy mu px py

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        r = x[:3]
        rmag = np.linalg.norm(r)
        dpos = x[3:6]
        accel = -r * self.mu / rmag**3 + self.pert
        dx[:6] = np.array([*dpos, *accel])

        if len(noise_t_vec) == 2:  # if noise
            dx += add_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])

        return dx


class Kepler2D(DynamicsModel):
    def __init__(self, Q, G, mu):
        super().__init__(Q, G)
        self.mu = mu

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        r = x[:2]
        rmag = np.linalg.norm(r)
        dpos = x[2:4]
        accel = -r * self.mu / rmag**3
        dx[:4] = np.array([*dpos, *accel])

        if len(noise_t_vec) == 2:  # if noise
            dx += add_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])

        return dx
