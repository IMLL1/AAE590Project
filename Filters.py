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

    def propagate(self, t, dt, noise_t_vec=()):
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
        self.P = (self.P + self.P.T) / 2

        return self.x, self.P

    def update(self, z, t):
        H = self.measModel.get_H(t, self.x)
        W = H @ self.P @ H.T + self.measModel.get_R(t, self.x)
        C = self.P @ H.T
        K = np.atleast_2d(np.linalg.solve(W.T, C.T).T)
        zhat = self.measModel.get_measurement(t, self.x)
        y = z - zhat
        self.x = self.x + K @ y
        self.P = self.P - C @ K.T - K @ C.T + K @ W @ K.T
        self.P = (self.P + self.P.T) / 2

        return (
            self.x,
            self.P,
            W,
            y,
        )


# UKF uses SUT
class UKF(Filter):
    def __init__(
        self,
        measModel: MeasurementModel,
        dynamicsModel: DynamicsModel,
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        alpha: float = 1e-3,
        kappa: float = 0,
        beta: float = 2,
    ):
        super().__init__(measModel, dynamicsModel, x0, P0)

        # parameters
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
        self.n = len(self.x)
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

        # define weights
        w0m = self.lam / (self.n + self.lam)
        w0c = w0m + (1 - self.alpha**2 + self.beta)
        wi = 1 / (2 * (self.n + self.lam))
        self.wm = [w0m, *[wi] * (2 * self.n)]
        self.wc = [w0c, *[wi] * (2 * self.n)]

    def get_sigmapoints(self):
        X0 = self.x
        chol = np.linalg.cholesky(self.P)
        X_stepm = [
            self.x + np.sqrt(self.n + self.lam) * chol[:, i] for i in range(self.n)
        ]
        X_stepp = [
            self.x - np.sqrt(self.n + self.lam) * chol[:, i] for i in range(self.n)
        ]
        X = [X0, *X_stepm, *X_stepp]
        return X

    def propagate(self, t, dt, noise_t_vec=()):
        Q = self.dynamicsModel.get_Q(t, self.x)
        G = self.dynamicsModel.G
        # sigma points
        Xkm1 = self.get_sigmapoints()
        # propagated sigma points
        Xk = [self.dynamicsModel.propagate_x(t, X, dt, noise_t_vec) for X in Xkm1]
        self.x = np.sum([self.wm[i] * Xk[i] for i in range(1 + 2 * self.n)], axis=0)
        self.P = (
            np.sum(
                [
                    self.wc[i] * np.outer((Xk[i] - self.x), (Xk[i] - self.x))
                    for i in range(1 + 2 * self.n)
                ],
                axis=0,
            )
            + G @ Q @ G.T
        )
        self.P = (self.P + self.P.T) / 2

        return self.x, self.P

    def update(self, z, t):
        # X sigma points
        Xsp = self.get_sigmapoints()
        # Z sigma points
        Zsp = [self.measModel.get_measurement(t, X) for X in Xsp]
        zhat = np.sum([self.wm[i] * Zsp[i] for i in range(1 + 2 * self.n)], axis=0)
        Pz = np.sum(
            [
                self.wc[i] * np.outer((Zsp[i] - zhat), (Zsp[i] - zhat))
                for i in range(1 + 2 * self.n)
            ],
            axis=0,
        ) + self.measModel.get_R(t, self.x)
        Pxz = np.sum(
            [
                self.wc[i] * np.outer((Xsp[i] - self.x), (Zsp[i] - zhat))
                for i in range(1 + 2 * self.n)
            ],
            axis=0,
        )
        K = np.atleast_2d(np.linalg.solve(Pz.T, Pxz.T).T)
        y = z - zhat
        self.x = self.x + K @ y
        self.P = self.P - Pxz @ K.T - K @ Pxz.T + K @ Pz @ K.T
        self.P = (self.P + self.P.T) / 2

        return (
            self.x,
            self.P,
            Pz,
            y,
        )
