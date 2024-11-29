import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal as rand
from Utils.Filters import UnscentedKalmanFilter
from Utils.ForceModels import *
from Utils.MeasModels import *

from tqdm import tqdm

np.random.seed(0)


# %% Configurable

mu = 3.9861e5  # km3/s2
x0 = [6750, 0, 0, 0, 6, 8, mu]
P0 = np.diag([*[0.1] * 3, *[0.1] * 3, 1e3]) ** 2

case = r"Pos Measurement/CR3BP $\mu$ Estimation"

dt = 60 * 5
propTime = 60 * 60 * 24

accel_sigma = 1e-6


def Q(extradim):
    sigmav = accel_sigma * dt
    sigmap = accel_sigma * dt**2 / 2
    return np.diag([*[sigmap] * 3, *[sigmav] * 3, *[0] * extradim]) ** 2


propagator = KeplerMass(Q(1), np.identity(len(P0)))
params_x = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\mu$"]
units_x = [*["km"] * 3, *["km/s"] * 3, r"km$^3$/s$^2$"]

sensor = PosMeas(observer=np.array([6371, 0, 0]))
params_z = [r"$x$", r"$y$", r"$z$"]
units_z = [*["km"] * 3]

# %% setup


t = np.arange(0, propTime, dt)  # sec
nx = len(x0)


xhat0 = np.random.multivariate_normal(x0, P0)

kf = UnscentedKalmanFilter(sensor, propagator, xhat0, P0)

# get truth and measurements
truth = propagator.get_truth(x0, t, disc_noise=True)
z = np.array([sensor.get_measurement(t[k], truth[k], True) for k in range(len(t))])

xhatm = []  # all prior state estiamtes
Pm = []  # all prior covariances
xhatp = [xhat0]  # all posterior state estiamtes
Pp = [P0]  # all posterior covariances

nz = len(z[0])

Ws = [np.full((nz, nz), np.nan)]
epsilons = [np.full(nz, np.nan)]

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


fig = plt.figure()
gs = fig.add_gridspec(3, 3)

fig.suptitle("Covariance Analysis\n" + case)
for i in range(len(params_x)):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.step(t, err[:, i])
    ax.step(t, bars[:, i], "-r", lw=1, alpha=0.5)
    ax.step(t, -bars[:, i], "-r", lw=1, alpha=0.5)
    ax.grid(True)
    ax.set_title(params_x[i] + " [" + units_x[i] + "]")

fig.tight_layout()
fig.legend(ax.get_lines()[0:2], ["Error", r"$3\sigma$ Bounds"], loc=1)
fig.supxlabel(r"Time ($t$) [sec]")
fig.supylabel(r"Error ($e$) [plot-dependent]")


fig = plt.figure()
gs = fig.add_gridspec(3, 1)

fig.suptitle("Measurement Residuals Analysis\n" + case)
for i in range(len(params_z)):
    ax = fig.add_subplot(gs[i // 1, i % 1])
    ax.step(t, err[:, i])
    ax.step(t, bars[:, i], "-r", lw=1, alpha=0.5)
    ax.step(t, -bars[:, i], "-r", lw=1, alpha=0.5)
    ax.grid(True)
    ax.set_title(params_z[i] + " [" + units_z[i] + "]")

fig.tight_layout()
fig.legend(ax.get_lines()[0:2], ["Residual", r"$3\sigma$ Bounds"], loc=1)
fig.supxlabel(r"Time ($t$) [sec]")
fig.supylabel(r"Measurement Residual ($e$) [plot-dependent]")

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
plt.plot(truth[:, 0], truth[:, 1], truth[:, 2])
plt.plot(0, 0, 0, ms=100)
ax.set_xlabel(r"$x$ [km]")
ax.set_ylabel(r"$y$ [km]")
ax.set_zlabel(r"$z$ [km]")
ax.set_title("True Trajectory")
plt.grid(True)
plt.axis("Equal")

plt.show()
