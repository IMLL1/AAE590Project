import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal as rand
from Filters import UnscentedKalmanFilter
from ForceModels import KeplerMass
from MeasModels import PosMeas
from tqdm import tqdm

np.random.seed(0)


######################### problem-specific functions #########################


Q = np.diag([1e-3, 1e-3, 1e-4, 1e-4]) ** 2


G = np.vstack((np.identity(4), np.zeros(4)))
R = np.diag([1e-2**2, 1e-2**2])


######################### problem-specific functions #########################
# continuous and discrete dt

mu = 3.9861e5  # km3/s2
x0 = [6450, 0, 0, 7.5, mu]

nx = len(x0)

t = np.arange(0, 0.25 * 60 * 60, 10)  # sec

sensor = PosMeas(R, planar=True)
propagator = KeplerMass(Q, G, planar=True)

P0 = np.diag([0.05, 0.05, 1e-2, 1e-2, 1e3]) ** 2
xhat0 = np.random.multivariate_normal(x0, P0)

ekf = UnscentedKalmanFilter(sensor, propagator, xhat0, P0)
# ckf = CubatureKalmanFilter(sensor, propagator, xhat0, P0)

# get truth and measurements
truth = propagator.get_truth(x0, t, disc_noise=True)
z = np.array([sensor.get_measurement(t[k], truth[k], True) for k in range(len(t))])

xhatm = []  # all prior state estiamtes
Pm = []  # all prior covariances
xhatp = [xhat0]  # all posterior state estiamtes
Pp = [P0]  # all posterior covariances

Ws = [np.full((2, 2), np.nan)]
epsilons = [np.full(2, np.nan)]

dt = t[2] - t[1]
for k in tqdm(range(1, len(t))):
    ekf.propagate(t[k - 1], dt)
    xhatm.append(ekf.x)
    Pm.append(ekf.P)

    _, _, Wk, yk = ekf.update(z[k], t[k])

    xhatp.append(ekf.x)
    Pp.append(ekf.P)

    Ws.append(Wk)
    epsilons.append(yk)


xhat = np.reshape(xhatp, (-1, nx))
truth = np.array(truth)

# do the same thing to xcont to compare
err = xhat - truth
bars = 3 * np.sqrt(np.array([np.diag(P) for P in Pp]))
err[:, -1] /= mu
bars[:, -1] /= mu

plt.figure()
plt.plot(truth[:, 0], truth[:, 1], label="Truth", lw=1)
plt.plot(xhat[:, 0], xhat[:, 1], label="Estimate", lw=1)
plt.grid(True)
plt.xlabel("ECI x ($x$) [km]")
plt.ylabel("ECI y ($y$) [km]")
plt.axis("equal")
plt.legend()
plt.title("Cartesian State")

fig, ax = plt.subplots(3, 2, layout="constrained")
fig.suptitle("Covariance Analysis")
params = ["x", "y", "v_x", "v_y", "mu"]
symbols = params
units = ["km", "km", "km/s", "km/s", "km^3/s^2"]
for i in range(5):
    ax[i // 2, i % 2].step(
        t, bars[:, i], "-r", lw=1, alpha=0.5, label="$\\pm3\\sigma$ Bounds"
    )
    ax[i // 2, i % 2].plot(t, -bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 2, i % 2].scatter(t, err[:, i], s=1, label="$" + symbols[i] + "$ Error")
    ax[i // 2, i % 2].legend()
    ax[i // 2, i % 2].grid(True)
    ax[i // 2, i % 2].set_ylabel(
        "$"
        + params[i]
        + "$ Error ($\\hat{"
        + symbols[i]
        + "}-"
        + symbols[i]
        + "$) ["
        + units[i]
        + "]"
    )
    ax[i // 2, i % 2].set_xlabel("Time ($t$) [seconds]")


epsilons = np.array(epsilons)
innovbars = 3 * np.sqrt(np.array([np.diag(W) for W in Ws]))
fig, ax = plt.subplots(2, 1, layout="constrained")
fig.suptitle("Innovations Test")
params = ["x", "y"]
symbols = ["x", "y"]
units = ["km", "km"]
for i in range(2):
    ax[i].plot(t, innovbars[:, i], "-r", lw=1, alpha=0.5, label="$\\pm3\\sigma$ Bounds")
    ax[i].plot(t, -innovbars[:, i], "-r", lw=1, alpha=0.5)
    ax[i].scatter(t, epsilons[:, i], s=1, label="$" + symbols[i] + "$ Error")
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_ylabel(
        params[i]
        + " Error ($\\hat{"
        + symbols[i]
        + "}-"
        + symbols[i]
        + "$) ["
        + units[i]
        + "]"
    )
ax[1].set_xlabel("Time ($t$) [seconds]")


plt.show()
