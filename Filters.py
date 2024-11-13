from MeasModels import *
from ForceModels import *
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from numpy.random import multivariate_normal as mvrn

# TODO: add noise to force model propagation


class Filter:
    def __init__(
        self,
        measModel: MeasurementModel,
        dynamicsModel: DynamicsModel,
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        # noise_vec: npt.ArrayLike,
        # t_vec: npt.ArrayLike,
    ):
        self.measModel = measModel
        self.dynamicsModel = dynamicsModel
        self.x = np.array(x0)
        self.P = np.array(P0)
        self.n = len(self.x)
        assert np.shape(self.P) == (self.n, self.n)
        # self.noise_vec = noise_vec
        # self.t_vec = t_vec
        # assert len(noise_vec) == len(t_vec)

    def propagate(self, t, x, dt, noise_t_vec=()):
        return NotImplementedError()

    def update(self, z, t):
        return NotImplementedError()


class EKF(Filter):
    def propagate(self, t, dt, noise_t_vec=()):
        sv0 = np.array([*self.x, *self.P.flatten()])
        nx = len(self.x)

        def joint_DE(t, sv):
            x = sv[:nx]
            P = np.reshape(sv[nx:], (nx, nx))
            F = self.dynamicsModel.get_F(t, x)
            Q = self.dynamicsModel.get_Q(t, x)
            dx = self.dynamicsModel.get_deriv(t, x, noise_t_vec)
            G = self.dynamicsModel.G
            dP = F @ P + P @ F.T + G @ Q @ G.T
            return np.array([*dx, *dP.flatten()])

        sv = solve_ivp(
            joint_DE,
            [t, t + dt],
            sv0,
            t_eval=[t + dt],
        ).y.flatten()
        self.x = sv[:nx]
        self.P = np.reshape(sv[nx:], (nx, nx))

        return self.x, self.P

    def update(self, z, t):
        H = self.measModel.get_H(t, self.x)
        W = H @ self.P @ H.T + self.measModel.get_R(t, self.x)
        C = self.P @ H.T
        K = np.atleast_2d(np.linalg.solve(W.T, C.T).T)
        zhat = self.measModel.get_measurement(t, self.x)
        y = z - zhat
        self.x = self.x + K @ y
        self.P = self.P - C @ K.T - K @ C.T + np.outer(K @ W, K)

        return (
            self.x,
            self.P,
            W,
            y,
        )
