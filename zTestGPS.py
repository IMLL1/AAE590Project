import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal as rand
from Filters import *
from ForceModels import *
from MeasModels import *
from tqdm import tqdm

np.random.seed(0)


# %% Configurable


def Q(t, x):  # mostly drag
    rmag = np.linalg.norm(x[:2] / 6500)
    return 0.1e-3**2 * np.identity(3) / rmag


G = np.block([np.zeros((3, 3)), np.identity(3), np.zeros((3, 1))]).T
R = np.diag([0.003] * 24)

# x0 = [6750, 0, 0, 0, 6, 8]
x0 = [6450, 0, 0, 0, 4, 6]
P0 = np.diag([*[0.1**2] * 3, *[0.1**2] * 3, 1e3**2])

case = r"EKF/Pos Measurement/$\mu$ Estimation"

dt = 60
propTime = 60 * 60 * 5


# %% setup
# continuous and discrete dt

mu = 3.9861e5  # km3/s2

t = np.arange(0, propTime, dt)  # sec
tcont = np.linspace(t[0], t[-1], 100 * len(t) - 1)  # "continuous" t values
pregenerated_rand = [rand(loc=0, scale=1, size=(len(x0) // 2)) for i in tcont]

x0 = [*x0, mu]
nx = len(x0)
sensor = WalkerPseudorange(R, np.rad2deg(55), 24, 6, 20, 20180, planar=False)
propagator = KeplerMass(Q, G, planar=False)

xhat0 = np.random.multivariate_normal(x0, P0)

kf = ExtendedKalmanFilter(sensor, propagator, xhat0, P0)


# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# for i in range(24):
#     plt.plot(sensor.pts[i][0], sensor.pts[i][1], sensor.pts[i][2], "ko", markersize=4)

# plt.show()

# get truth and measurements
truth = propagator.get_truth(x0, t, (pregenerated_rand, tcont))
z = np.array([sensor.get_measurement(t[k], truth[k], True) for k in range(len(t))])

xhatm = []  # all prior state estiamtes
Pm = []  # all prior covariances
xhatp = [xhat0]  # all posterior state estiamtes
Pp = [P0]  # all posterior covariances

Ws = [np.full(np.shape(R), np.nan)]
epsilons = [np.full(len(np.diag(R)), np.nan)]

# %% filter
dt = t[2] - t[1]
for k in tqdm(range(1, len(t))):
    kf.propagate(t[k - 1], dt)
    xhatm.append(kf.x)
    Pm.append(kf.P)

    _, _, Wk, yk = kf.update(z[k], t[k])

    xhatp.append(kf.x)
    Pp.append(kf.P)

    Ws.append(Wk)
    epsilons.append(yk)


xhat = np.reshape(xhatp, (-1, nx))
truth = np.array(truth)

# do the same thing to xcont to compare
err = xhat - truth
bars = 3 * np.sqrt(np.array([np.diag(P) for P in Pp]))

epsilons = np.array(epsilons)
innovbars = 3 * np.sqrt(np.array([np.diag(W) for W in Ws]))

# %% analysis
# plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

# fig, ax = plt.subplots(3, 3, layout="tight")
# fig.suptitle("Covariance Analysis\n" + case)
# params = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\mu$"]
# units = [*["km"] * 3, *["km/s"] * 3, r"km$^3$/s$^2$"]
# for i in range(len(params)):
#     ax[i // 3, i % 3].plot(t, err[:, i])
#     ax[i // 3, i % 3].step(t, bars[:, i], "-r", lw=1, alpha=0.5)
#     ax[i // 3, i % 3].plot(t, -bars[:, i], "-r", lw=1, alpha=0.5)
#     ax[i // 3, i % 3].grid(True)
#     ax[i // 3, i % 3].set_title(params[i] + " [" + units[i] + "]")


# fig.legend(ax[0][0].get_lines()[0:2], ["Error", r"$3\sigma$ Bounds"], loc=1)
# fig.supxlabel(r"Time ($t$) [sec]")
# fig.supylabel(r"Error ($e$) [plot-dependent]")


# fig, ax = plt.subplots(3, 1, layout="tight")
# fig.suptitle("Measurement Residuals Analysis\n" + case)
# params = [r"$x$", r"$y$", r"$z$"]
# units = [*["km"] * 3]
# for i in range(len(params)):
#     ax[i].plot(t, err[:, i])
#     ax[i].step(t, bars[:, i], "-r", lw=1, alpha=0.5)
#     ax[i].plot(t, -bars[:, i], "-r", lw=1, alpha=0.5)
#     ax[i].grid(True)
#     ax[i].set_title(params[i] + " [" + units[i] + "]")


# fig.legend(ax[-1].get_lines()[0:2], ["Residual", r"$3\sigma$ Bounds"], loc=1)
# fig.supxlabel(r"Time ($t$) [sec]")
# fig.supylabel(r"Measurement Residual ($e$) [plot-dependent]")


plt.show()
