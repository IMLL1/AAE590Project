from numpy.random import multivariate_normal as mvrn
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
import numpy as np


def add_continuous_noise(dynamics, t, x, noisevec, tvec):
    idx = np.abs(t - tvec).argmin()
    Qsqrt = np.linalg.cholesky(dynamics.get_Q(t, x))
    w = Qsqrt @ noisevec[idx]
    return dynamics.G @ w


class DynamicsModel:
    def __init__(self, Q, G, planar=False):
        self.Q = Q
        self.G = G
        self.planar = planar

    def get_Q(self, t, x):
        Q = self.Q(t, x) if callable(self.Q) else self.Q
        return Q

    def get_F(self, t, x):
        f = lambda state: self.get_deriv(t, state)
        F = np.atleast_2d(approx_fprime(x, f))
        if len(np.shape(F)) == 1:
            F = np.atleast_2d(F)
        return F

    def get_STM(self, t, x, dt):
        f = lambda state: self.propagate_x(t, state, dt)
        F = np.atleast_2d(approx_fprime(x, f))
        if len(np.shape(F)) == 1:
            F = np.atleast_2d(F)
        return F

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        if len(noise_t_vec) == 2:  # if noise
            dx += add_continuous_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])
        return dx

    def propagate_x(self, t, x, dt, disc_noise=False, noise_t_vec=()):
        assert not (disc_noise and (len(noise_t_vec) > 0))
        f = lambda t, x: self.get_deriv(t, x, noise_t_vec)
        x1 = solve_ivp(f, [t, t + dt], x, t_eval=[t + dt]).y.flatten()
        if disc_noise:
            x1 += self.G @ mvrn(np.zeros(np.size(self.G, 1)), self.get_Q(t, x))
        return x1

    def get_truth(self, x0, tk, disc_noise=False, noise_t_vec=()):
        x0 = np.array(x0)
        assert not (disc_noise and (len(noise_t_vec) > 0))
        if disc_noise:
            xk = [x0]
            dt = tk[1] - tk[0]
            for i in range(len(tk) - 1):
                t = tk[i]
                xk.append(self.propagate_x(t, xk[-1], dt, disc_noise=disc_noise))
        else:
            f = lambda t, x: self.get_deriv(t, x, noise_t_vec)
            xk = list(solve_ivp(f, [tk[0], tk[-1]], x0, t_eval=tk).y.T)
        return xk


class Linear(DynamicsModel):
    def __init__(self, Q, G, A, planar=False):
        super().__init__(Q, G, planar)
        self.A = A

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = self.A @ x

        if len(noise_t_vec) == 2:  # if noise
            dx += add_continuous_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])

        return dx


class Kepler(DynamicsModel):
    def __init__(self, Q, G, mu, planar=False):
        super().__init__(Q, G, planar)
        self.mu = mu

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        r = x[:2] if self.planar else x[:3]
        rmag = np.linalg.norm(r)
        dpos = x[2:4] if self.planar else x[3:6]
        accel = -r * self.mu / rmag**3
        dx[:6] = np.array([*dpos, *accel])

        if len(noise_t_vec) == 2:  # if noise
            dx += add_continuous_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])

        return dx


class KeplerPerurbed(DynamicsModel):
    def __init__(self, Q, G, mu, pert_vec, planar=False):
        super().__init__(Q, G, planar)
        self.mu = mu
        self.pert = np.array(pert_vec)
        # state: x y vx vy px py

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        r = x[:2] if self.planar else x[:3]
        rmag = np.linalg.norm(r)
        dpos = x[2:4] if self.planar else x[3:6]
        accel = -r * self.mu / rmag**3 + self.pert
        end = 4 if self.planar else 6
        dx[:end] = np.array([*dpos, *accel])

        if len(noise_t_vec) == 2:  # if noise
            dx += add_continuous_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])

        return dx


class KeplerMass(DynamicsModel):
    def __init__(self, Q, G, planar=False):
        super().__init__(Q, G, planar)
        # state: x y vx vy px py mu

    def get_deriv(self, t, x, noise_t_vec=()):
        dx = np.zeros_like(x)
        r = x[:2] if self.planar else x[:3]
        rmag = np.linalg.norm(r)
        dpos = x[2:4] if self.planar else x[3:6]
        muidx = -1
        accel = -r * x[muidx] / rmag**3
        end = 4 if self.planar else 6
        dx[:end] = np.array([*dpos, *accel])

        if len(noise_t_vec) == 2:  # if noise
            dx += add_continuous_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])

        return dx
