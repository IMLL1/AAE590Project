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


class CR3BP(DynamicsModel):
    def __init__(self, Q, G, mu, LU=1, TU=1, planar=False):
        super().__init__(Q, G, planar)
        self.mu = mu
        self.LU = LU
        self.TU = TU
        self.planar = planar

    def get_deriv(self, t, x, noise_t_vec=()):
        X, Y = x[:2] / self.LU
        VX, VY = x[2:4] if self.planar else x[3:5]
        VX /= self.LU / self.TU
        VY /= self.LU / self.TU
        pos = x[:2] / self.LU if self.planar else x[:3] / self.LU
        P1 = np.array([-self.mu, 0]) if self.planar else np.array([-self.mu, 0, 0])
        P2 = np.array([1 - self.mu]) if self.planar else np.array([1 - self.mu, 0, 0])
        r1vec = pos - P1
        r2vec = pos - P2
        r1mag = np.linalg.norm(r1vec)
        r2mag = np.linalg.norm(r2vec)

        ddpos = -(1 - self.mu) * r1vec / r1mag**3 - self.mu * r2vec / r2mag**3
        coriolis = (
            np.array([2 * VY + X, -2 * VX + Y])
            if self.planar
            else np.array([2 * VY + X, -2 * VX + Y, 0])
        )
        ddpos += coriolis

        ddpos *= self.LU / self.TU**2

        dx = np.zeros(4) if self.planar else np.zeros(6)
        velstart = 2 if self.planar else 3
        dx[:velstart] = x[velstart:]
        dx[velstart:] = ddpos

        if len(noise_t_vec) == 2:  # if noise
            dx += add_continuous_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])
        return dx


class CR3BPMassRatio(DynamicsModel):
    def __init__(self, Q, G, LU=1, TU=1, planar=False):
        super().__init__(Q, G, planar)
        self.LU = LU
        self.TU = TU
        self.planar = planar

    def get_deriv(self, t, x, noise_t_vec=()):
        mu = x[-1]
        X, Y = x[:2] / self.LU
        VX, VY = x[2:4] if self.planar else x[3:5]
        VX /= self.LU / self.TU
        VY /= self.LU / self.TU
        pos = x[:2] / self.LU if self.planar else x[:3] / self.LU
        P1 = np.array([-mu, 0]) if self.planar else np.array([-mu, 0, 0])
        P2 = np.array([1 - mu]) if self.planar else np.array([1 - mu, 0, 0])
        r1vec = pos - P1
        r2vec = pos - P2
        r1mag = np.linalg.norm(r1vec)
        r2mag = np.linalg.norm(r2vec)

        ddpos = -(1 - mu) * r1vec / r1mag**3 - mu * r2vec / r2mag**3
        coriolis = (
            np.array([2 * VY + X, -2 * VX + Y])
            if self.planar
            else np.array([2 * VY + X, -2 * VX + Y, 0])
        )
        ddpos += coriolis

        ddpos *= self.LU / self.TU**2

        dx = np.zeros(5) if self.planar else np.zeros(7)
        velstart = 2 if self.planar else 3
        dx[:velstart] = x[velstart : 2 * velstart]
        dx[velstart : 2 * velstart] = ddpos

        if len(noise_t_vec) == 2:  # if noise
            dx += add_continuous_noise(self, t, x, noise_t_vec[0], noise_t_vec[1])
        return dx
