import os
import sys

sys.path.append(os.path.abspath(os.curdir))

from utils.snippets import *
from functools import partial
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def lorenz(t, Y, sigma=10, beta=8 / 3, rho=28):
    x, y, z = Y
    YDot = np.zeros_like(Y)
    YDot[0] = sigma * (y - x)
    YDot[1] = x * (rho - z) - y
    YDot[2] = x * y - beta * z
    print(f"time={t:.6f}")
    return YDot


def lorenz_ddot(Y, YDot, sigma=10, beta=8 / 3, rho=28):
    x, y, z = Y
    xDot, yDot, zDot = YDot
    YDDot = np.zeros_like(Y)
    YDDot[0] = sigma * (yDot - xDot)
    YDDot[1] = xDot * (rho - z) - x * zDot - yDot
    YDDot[2] = xDot * y + x * yDot - beta * zDot
    return YDDot


def solve(t, X_0=np.ones(3), sigma=10, beta=8 / 3, rho=28):
    # solve
    x0, y0, z0 = X_0
    Y_0 = np.array([x0, y0, z0])
    func = partial(lorenz, sigma=sigma, beta=beta, rho=rho)
    solverparams = dict(method="Radau", t_eval=t, vectorized=True, max_step=t[1] - t[0])
    sol = solve_ivp(func, (t[0], t[-1]), Y_0, **solverparams)

    # save data
    y = sol.y  # [3, N]
    yDot = lorenz(0, sol.y)  # [3, N]
    yDDot = lorenz_ddot(y, yDot)  # [3, N]
    result = [y.T, yDot.T, yDDot.T]
    data = pd.DataFrame(
        np.concatenate([t.reshape(-1, 1)] + result, axis=1),
        columns=["t", "x", "y", "z", "xDot", "yDot", "zDot", "xDDot", "yDDot", "zDDot"],
    )
    os.makedirs('data', exist_ok=True)
    data.to_csv("data/train_lorenz.csv", index=False)

    # plot
    plot_template()
    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[0].plot(t, result[0][:, i], label="0th")
        ax[1].plot(t, result[1][:, i], label="1st")
        ax[2].plot(t, result[2][:, i], label="2nd")
    fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(12, 5))
    ax[0].plot(result[0][:, 0], result[0][:, 1], result[0][:, 2])
    ax[1].plot(result[1][:, 0], result[1][:, 1], result[1][:, 2])
    ax[2].plot(result[2][:, 0], result[2][:, 1], result[2][:, 2])
    # plt.show()
    os.makedirs("figures/lorenz", exist_ok=True)
    fig.savefig("figures/lorenz/data.png", dpi=200)

    fig, ax = plt.subplots(3, 1, figsize=(5, 5))
    for i in range(3):
        for j in range(3):
            ax[i].plot(t, result[i][:, j], c="grey")
    fig.tight_layout()
    fig.savefig("figures/lorenz/data_time.png", dpi=200)


if __name__ == "__main__":
    np.random.seed(0)
    t = np.linspace(0, 20, int(20 / 1e-3) + 1, endpoint=True)  # 20
    X_0 = np.random.rand(3) * 2 - 1
    solve(t, X_0)
