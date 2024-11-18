import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.random import normal as rand
from Filters import UKF
from ForceModels import Kepler
from MeasModels import RangeAndRArate
from tqdm import tqdm

np.random.seed(0)


######################### problem-specific functions #########################
def calc_period(state, mu):
    r = state[:2]
    v = state[2:]
    energy = 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)
    a = -mu / (2 * energy)
    P = 2 * np.pi * np.sqrt(a**3 / mu)
    return P


G = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
Q = (
    lambda t, x: np.array([[1e-5**2, 0], [0, 1e-5**2]])
    / np.linalg.norm(x[:2] / 6500) ** 2
)
R = np.diag([1e-3, 1e-4])


######################### problem-specific functions #########################
# continuous and discrete dt

mu = 3.986004418e5  # km3/s2
x0 = np.array([6750, 0, 0, 10])

nx = len(x0)

t = np.arange(0, calc_period(x0, mu), 60)  # sec
tcont = np.linspace(t[0], t[-1], 10 * len(t) - 1)  # "continuous" t values
pregenerated_rand = [rand(loc=0, scale=1, size=(2)) for i in tcont]

sensor = RangeAndRArate(R, planar=True)
propagator = Kepler(Q, G, mu, planar=True)
P0 = np.diag([2**2, 2**2, 1e-2**2, 1e-2**2])
xhat0 = np.random.multivariate_normal(x0, P0)
ekf = UKF(sensor, propagator, xhat0, P0, alpha=1e-5, beta=1e4)

# get truth and measurements
truth = propagator.get_truth(x0, t, (pregenerated_rand, tcont))
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


xhat = np.reshape(xhatp, (-1, 4))
truth = np.array(truth)

# do the same thing to xcont to compare
err = xhat - truth
bars = 3 * np.sqrt(np.array([np.diag(P) for P in Pp]))

plt.figure()
plt.plot(truth[:, 0], truth[:, 1], label="Truth", lw=1)
plt.plot(xhat[:, 0], xhat[:, 1], label="Estimate", lw=1)
plt.grid(True)
plt.xlabel("ECI x ($x$) [km]")
plt.ylabel("ECI y ($y$) [km]")
plt.axis("equal")
plt.legend()
plt.title("Cartesian State")

fig, ax = plt.subplots(2, 2, layout="constrained")
fig.suptitle("Covariance Analysis")
params = ["x", "y", "v_x", "v_y"]
symbols = params
units = ["km", "km", "km/s", "km/s"]
for i in range(4):
    ax[i // 2, i % 2].step(
        t, bars[:, i], "-r", lw=1, alpha=0.5, label="$\\pm3\\sigma$ Bounds"
    )
    ax[i // 2, i % 2].plot(t, -bars[:, i], "-r", lw=1, alpha=0.5)
    ax[i // 2, i % 2].plot(t, err[:, i], label="$" + symbols[i] + "$ Error")
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
params = ["Range", "TA Rate"]
symbols = ["R", "\\dot{\\theta}"]
units = ["km", "rad/s"]
for i in range(2):
    ax[i].plot(t, innovbars[:, i], "-r", lw=1, alpha=0.5, label="$\\pm3\\sigma$ Bounds")
    ax[i].plot(t, -innovbars[:, i], "-r", lw=1, alpha=0.5)
    ax[i].plot(t, epsilons[:, i], label="$" + symbols[i] + "$ Error")
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
