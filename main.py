import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal as rand
from Filters import ExtendedKalmanFilter
from ForceModels import KeplerMass
from MeasModels import PosMeas
from tqdm import tqdm

np.random.seed(0)


# %% Configurable


def Q(t, x):  # mostly drag
    rmag = np.linalg.norm(x[:2] / 6500)
    return 0.1e-3**2 * np.identity(3) / rmag


G = np.block([np.zeros((3, 3)), np.identity(3), np.zeros((3, 1))]).T
R = np.diag([10e-3**2, 10e-3**2, 10e-3**2])

x0 = [6750, 0, 0, 0, 6, 8]
P0 = np.diag([*[0.1**2] * 3, *[0.1**2] * 3, 1e3**2])

case = r"EKF/Pos Measurement/$\mu$ Estimation"


# %% setup
# continuous and discrete dt

mu = 3.9861e5  # km3/s2

t = np.arange(0, 10 * 60 * 60, 60 * 5)  # sec
tcont = np.linspace(t[0], t[-1], 100 * len(t) - 1)  # "continuous" t values
pregenerated_rand = [rand(loc=0, scale=1, size=(len(x0) // 2)) for i in tcont]

x0 = [*x0, mu]
nx = len(x0)

sensor = PosMeas(R, planar=False)
propagator = KeplerMass(Q, G, planar=False)

xhat0 = np.random.multivariate_normal(x0, P0)

kf = ExtendedKalmanFilter(sensor, propagator, xhat0, P0)

# get truth and measurements
truth = propagator.get_truth(x0, t, (pregenerated_rand, tcont))
z = np.array([sensor.get_measurement(t[k], truth[k], True) for k in range(len(t))])

xhatm = []  # all prior state estiamtes
Pm = []  # all prior covariances
xhatp = [xhat0]  # all posterior state estiamtes
Pp = [P0]  # all posterior covariances

Ws = [np.full((3, 3), np.nan)]
epsilons = [np.full(3, np.nan)]

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
plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

fig, ax = plt.subplots(3, 3, layout="tight")
fig.suptitle("Covariance Analysis\n" + case)
params = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\mu$"]
units = [*["km"] * 3, *["km/s"] * 3, r"km$^3$/s$^2$"]
for i in range(len(params)):
    ax[i // 3, i % 3].plot(t, err[:, i])
    ax[i // 3, i % 3].step(t, bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 3, i % 3].plot(t, -bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 3, i % 3].grid(True)
    ax[i // 3, i % 3].set_title(params[i] + " [" + units[i] + "]")


fig.legend(ax[0][0].get_lines()[0:2], ["Error", r"$3\sigma$ Bounds"], loc=1)
fig.supxlabel(r"Time ($t$) [sec]")
fig.supylabel(r"Error ($e$) [plot-dependent]")


fig, ax = plt.subplots(3, 1, layout="tight")
fig.suptitle("Measurement Residuals Analysis\n" + case)
params = [r"$x$", r"$y$", r"$z$"]
units = [*["km"] * 3]
for i in range(len(params)):
    ax[i].plot(t, err[:, i])
    ax[i].step(t, bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i].plot(t, -bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i].grid(True)
    ax[i].set_title(params[i] + " [" + units[i] + "]")


fig.legend(ax[-1].get_lines()[0:2], ["Residual", r"$3\sigma$ Bounds"], loc=1)
fig.supxlabel(r"Time ($t$) [sec]")
fig.supylabel(r"Measurement Residual ($e$) [plot-dependent]")


plt.show()
