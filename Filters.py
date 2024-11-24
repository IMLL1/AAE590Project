from MeasModels import *
from ForceModels import *
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from numpy.random import multivariate_normal as mvrn


class Filter:
    def __init__(
        self,
        measModel: MeasurementModel,
        dynamicsModel: DynamicsModel,
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
    ):
        self.measModel = measModel
        self.dynamicsModel = dynamicsModel
        self.x = np.array(x0)
        self.P = np.array(P0)
        self.n = len(self.x)
        assert np.shape(self.P) == (self.n, self.n)

    def propagate(self, t, dt):
        return NotImplementedError()

    def update(self, z, t):
        return NotImplementedError()


class LinearKalmanFilter(Filter):
    def __init__(self, measModel, dynamicsModel, x0, P0):
        super().__init__(measModel, dynamicsModel, x0, P0)

    def propagate(self, t, dt):
        G = self.dynamicsModel.G
        Q = self.dynamicsModel.get_Q(t, self.x)
        STM = self.dynamicsModel.get_STM(t, self.x, dt)
        self.x = STM @ self.x

        self.P = STM @ self.P @ STM.T + G @ Q @ G.T

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

        return self.x, self.P, W, y


class ExtendedKalmanFilter(Filter):
    def __init__(self, measModel, dynamicsModel, x0, P0):
        super().__init__(measModel, dynamicsModel, x0, P0)

    def propagate(self, t, dt):
        G = self.dynamicsModel.G
        Q = self.dynamicsModel.get_Q(t, self.x)
        STM = self.dynamicsModel.get_STM(t, self.x, dt)
        self.x = self.dynamicsModel.propagate_x(t, self.x, dt)

        self.P = STM @ self.P @ STM.T + G @ Q @ G.T

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

        return self.x, self.P, W, y


# UKF uses SUT
class UnscentedKalmanFilter(Filter):
    def __init__(
        self,
        measModel: MeasurementModel,
        dynamicsModel: DynamicsModel,
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        alpha: float = 1e-3,
        beta: float = 2,
        kappa: float = 0,
    ):
        super().__init__(measModel, dynamicsModel, x0, P0)

        # parameters
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
        self.n = len(self.x)
        self.lam = ((self.alpha**2) * (self.n + self.kappa)) - self.n

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
            self.x - np.sqrt(self.n + self.lam) * chol[:, i] for i in range(self.n)
        ]
        X_stepp = [
            self.x + np.sqrt(self.n + self.lam) * chol[:, i] for i in range(self.n)
        ]
        X = [X0, *X_stepm, *X_stepp]
        return X

    def propagate(self, t, dt):
        Q = self.dynamicsModel.get_Q(t, self.x)
        G = self.dynamicsModel.G
        # sigma points
        Xkm1 = self.get_sigmapoints()
        # propagated sigma points
        Xk = [self.dynamicsModel.propagate_x(t, X, dt) for X in Xkm1]
        self.x = np.sum([self.wm[i] * Xk[i] for i in range(1 + 2 * self.n)], axis=0)
        self.P = (
            sum(
                [
                    self.wc[i] * np.outer((Xk[i] - self.x), (Xk[i] - self.x))
                    for i in range(1 + 2 * self.n)
                ]
            )
            + G @ Q @ G.T
        )
        self.P = (self.P + self.P.T) / 2

        # import matplotlib.pyplot as plt

        # pts = np.random.multivariate_normal(self.x, self.P, 1000)

        # plt.scatter(pts[:, 0], pts[:, 1], c="r")
        # plt.scatter(np.array(Xk)[:, 0], np.array(Xk)[:, 1], c="b")
        # plt.show()
        return self.x, self.P

    def update(self, z, t):
        # X sigma points
        Xsp = self.get_sigmapoints()
        # Z sigma points
        Zsp = [self.measModel.get_measurement(t, X) for X in Xsp]
        zhat = np.sum([self.wm[i] * Zsp[i] for i in range(1 + 2 * self.n)], axis=0)
        Pz = sum(
            [
                self.wc[i] * np.outer((Zsp[i] - zhat), (Zsp[i] - zhat))
                for i in range(1 + 2 * self.n)
            ]
        ) + self.measModel.get_R(t, self.x)
        Pxz = sum(
            [
                # self.wc[i] * np.outer((Xsp[i] - self.x), (Zsp[i] - zhat))
                self.wc[i] * np.outer((Xsp[i] - self.x), (Zsp[i] - zhat))
                for i in range(1 + 2 * self.n)
            ]
        )
        K = np.atleast_2d(np.linalg.solve(Pz.T, Pxz.T).T)
        y = z - zhat
        self.x = self.x + K @ y
        self.P = self.P - Pxz @ K.T - K @ Pxz.T + K @ Pz @ K.T
        self.P = (self.P + self.P.T) / 2

        return self.x, self.P, Pz, y


class CubatureKalmanFilter(Filter):
    def __init__(
        self,
        measModel: MeasurementModel,
        dynamicsModel: DynamicsModel,
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
    ):
        super().__init__(measModel, dynamicsModel, x0, P0)

        # parameters
        n = len(self.x)
        self.xi = np.sqrt(n) * np.block([np.identity(n), -np.identity(n)])
        self.n = n

    def get_sigmapoints(self):
        chol = np.linalg.cholesky(self.P)
        X = [self.x + chol @ self.xi[:, i] for i in range(2 * self.n)]
        return X

    def propagate(self, t, dt):
        Q = self.dynamicsModel.get_Q(t, self.x)
        G = self.dynamicsModel.G
        # sigma points
        Xkm1 = self.get_sigmapoints()
        # propagated sigma points
        Xk = [self.dynamicsModel.propagate_x(t, X, dt) for X in Xkm1]
        self.x = sum(Xk) / (2 * self.n)
        self.P = (
            sum(
                [
                    np.outer((Xk[i] - self.x), (Xk[i] - self.x))
                    for i in range(2 * self.n)
                ]
            )
            / (2 * self.n)
            + G @ Q @ G.T
        )
        self.P = (self.P + self.P.T) / 2

        return self.x, self.P

    def update(self, z, t):
        # X sigma points
        Xsp = self.get_sigmapoints()
        # Z sigma points
        Zsp = [self.measModel.get_measurement(t, X) for X in Xsp]
        zhat = sum(Zsp) / (2 * self.n)
        Pz = sum(
            [np.outer((Zsp[i] - zhat), (Zsp[i] - zhat)) for i in range(2 * self.n)]
        ) / (2 * self.n) + self.measModel.get_R(t, self.x)
        Pxz = sum(
            [np.outer((Xsp[i] - self.x), (Zsp[i] - zhat)) for i in range(2 * self.n)]
        ) / (2 * self.n)
        K = np.atleast_2d(np.linalg.solve(Pz.T, Pxz.T).T)
        y = z - zhat
        self.x = self.x + K @ y
        self.P = self.P - Pxz @ K.T - K @ Pxz.T + K @ Pz @ K.T
        self.P = (self.P + self.P.T) / 2

        return self.x, self.P, Pz, y


# class ExtendedKalmanFilterContinuous(Filter):
#     def __init__(self, measModel, dynamicsModel, x0, P0):
#         super().__init__(measModel, dynamicsModel, x0, P0)
#         print("Using continoous EKF, Q is continuous!\n")

#     def propagate(self, t, dt, noise_t_vec=()):
#         sv0 = np.array([*self.x, *self.P.flatten()])
#         nx = len(self.x)

#         def joint_DE(t, sv):
#             x = sv[:nx]
#             P = np.reshape(sv[nx:], (nx, nx))
#             F = self.dynamicsModel.get_F(t, x)
#             Q = self.dynamicsModel.get_Q(t, x)
#             dx = self.dynamicsModel.get_deriv(t, x, noise_t_vec)
#             G = self.dynamicsModel.G
#             dP = F @ P + P @ F.T + G @ Q @ G.T
#             return np.array([*dx, *dP.flatten()])

#         sv = solve_ivp(
#             joint_DE,
#             [t, t + dt],
#             sv0,
#             t_eval=[t + dt],
#         ).y.flatten()
#         self.x = sv[:nx]
#         self.P = np.reshape(sv[nx:], (nx, nx))
#         self.P = (self.P + self.P.T) / 2

#         return self.x, self.P

#     def update(self, z, t):
#         H = self.measModel.get_H(t, self.x)
#         W = H @ self.P @ H.T + self.measModel.get_R(t, self.x)
#         C = self.P @ H.T
#         K = np.atleast_2d(np.linalg.solve(W.T, C.T).T)
#         zhat = self.measModel.get_measurement(t, self.x)
#         y = z - zhat
#         self.x = self.x + K @ y
#         self.P = self.P - C @ K.T - K @ C.T + K @ W @ K.T
#         self.P = (self.P + self.P.T) / 2

#         return self.x, self.P, W, y
