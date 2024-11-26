import numpy as np
import scipy as sp
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
    I = np.identity(3)
    return sp.linalg.block_diag(10e-3**2 * I / rmag, 0.1e-3**2 * I / rmag)


G = np.block([np.identity(6), np.zeros((6, 1))]).T
R = np.diag([0.01] * 24) ** 2

# x0 = [6750, 0, 0, 0, 6, 8]
x0 = [6450, 0, 0, 0, 4, 6]
P0 = np.diag([*[0.01**2] * 3, *[0.01**2] * 3, 1e2**2])

case = r"EKF/Pos Measurement/$\mu$ Estimation"

dt = 60
propTime = 60 * 60 * 0.5


# %% setup
# continuous and discrete dt

mu = 3.9861e5  # km3/s2

t = np.arange(0, propTime, dt)  # sec

x0 = [*x0, mu]
nx = len(x0)
sensor = WalkerPseudorange(R, np.rad2deg(55), 24, 6, 20, 20180, planar=False)
propagator = KeplerMass(Q, G, planar=False)

xhat0 = np.random.multivariate_normal(x0, P0)

kf = UnscentedKalmanFilter(sensor, propagator, xhat0, P0)


# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# for i in range(24):
#     plt.plot(sensor.pts[i][0], sensor.pts[i][1], sensor.pts[i][2], "ko", markersize=4)

# plt.show()

# get truth and measurements
truth = propagator.get_truth(x0, t, disc_noise=True)
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
err[:, :6] *= 1e3
bars[:, :6] *= 1e3

epsilons = 1e3 * np.array(epsilons)
innovbars = 1e3 * 3 * np.sqrt(np.array([np.diag(W) for W in Ws]))

# %% analysis
plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

fig, ax = plt.subplots(3, 3, layout="tight")
fig.suptitle("Covariance Analysis\n" + case)
params = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\mu$"]
units = [*["m"] * 3, *["m/s"] * 3, r"km$^3$/s$^2$"]
for i in range(len(params)):
    ax[i // 3, i % 3].plot(t, err[:, i])
    ax[i // 3, i % 3].step(t, bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 3, i % 3].plot(t, -bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 3, i % 3].grid(True)
    ax[i // 3, i % 3].set_title(params[i] + " [" + units[i] + "]")


fig.legend(ax[0][0].get_lines()[0:2], ["Error", r"$3\sigma$ Bounds"], loc=1)
fig.supxlabel(r"Time ($t$) [sec]")
fig.supylabel(r"Error ($e$) [plot-dependent]")


fig, ax = plt.subplots(6, 4)
fig.suptitle("Pseudorange Residuals Analysis\n" + case)
params = [r"$\rho_{" + str(i + 1) + r"}$" for i in range(24)]
units = [*["m"] * 24]
for i in range(len(params)):
    ax[i // 4, i % 4].plot(t, epsilons[:, i])
    ax[i // 4, i % 4].step(t, innovbars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 4, i % 4].plot(t, -innovbars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 4, i % 4].grid(True)
    ax[i // 4, i % 4].set_title(params[i] + " [" + units[i] + "]")


fig.legend(ax.flatten()[0].get_lines()[0:2], ["Residual", r"$3\sigma$ Bounds"], loc=1)
fig.supxlabel(r"Time ($t$) [sec]")
fig.supylabel(r"Measurement Residual ($e$) [plot-dependent]")
plt.tight_layout(w_pad=-1, h_pad=-0.5, pad=0.25)


plt.show()
