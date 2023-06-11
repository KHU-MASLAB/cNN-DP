import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import matplotlib.pyplot as plt
from utils.integrate import rkint
import pandas as pd
from utils.snippets import *
from functools import partial
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import shutil
import multiprocessing
import glob


def rf(t, Y, alpha=1.1, gamma=0.87, beta=0.1):
    x, y, z = Y[:3]
    YDot = np.zeros_like(Y)
    YDot[0] = y * (z - 1 + x**2) + gamma * x + beta
    YDot[1] = x * (3 * z + 1 - x**2) + gamma * y + beta
    YDot[2] = -2 * z * (alpha + x * y) + beta
    # print time
    if t % 1 < 1e-8:
        print(f"PID({os.getpid()}) time={t:.6f}\n")
    return YDot


def rf_ddot(Y, YDot, alpha=1.1, gamma=0.87, beta=0.1):
    x, y, z = Y
    xDot, yDot, zDot = YDot
    YDDot = np.zeros_like(Y)
    YDDot[0] = yDot * (z - 1 + x**2) + y * (zDot + 2 * x * xDot) + gamma * xDot
    YDDot[1] = (
        xDot * (3 * z + 1 - x**2) + x * (3 * zDot - 2 * x * xDot) + gamma * yDot
    )
    YDDot[2] = -2 * zDot*(alpha + x * y) - 2 * z * (xDot * y + x * yDot)
    return YDDot


def solve(
    t,
    Y_0=np.zeros(3),
    alpha=1.1,
    gamma=0.87,
    beta=0.1,
    plot=True,
    index=0,
    SUBDIR_NAME="train",
):
    # solve
    func = partial(rf, alpha=alpha, gamma=gamma, beta=beta)
    solverparams = dict(method="Radau", t_eval=t, vectorized=True, max_step=t[1] - t[0])
    sol = solve_ivp(func, (t[0], t[-1]), Y_0, **solverparams)

    # save data
    y = sol.y
    yDot = rf(0, y, alpha=alpha, gamma=gamma, beta=beta)
    yDDot = rf_ddot(y, yDot, alpha=alpha, gamma=gamma, beta=beta)
    result = [y.T, yDot.T, yDDot.T]
    if not sol.success:
        t = sol.t
    param_vecs = np.array([[alpha, gamma]]).repeat(len(t), axis=0)

    os.makedirs(f"data/rf/{SUBDIR_NAME}", exist_ok=True)
    data = pd.DataFrame(
        np.concatenate([t.reshape(-1, 1), param_vecs] + result, axis=1),
        columns=[
            "t",
            "alpha",
            "gamma",
            "x",
            "y",
            "z",
            "xDot",
            "yDot",
            "zDot",
            "xDDot",
            "yDDot",
            "zDDot",
        ],
    )
    csv_name = f"{SUBDIR_NAME}_{index:03d}"
    if not sol.success:
        data.to_csv(f"data/rf/{SUBDIR_NAME}/{csv_name+'_FAIL'}.csv", index=False)
    else:
        data.to_csv(f"data/rf/{SUBDIR_NAME}/{csv_name}.csv", index=False)

    # plot
    if plot:
        plot_template()
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[0].plot(
                t,
                result[0][:, i],
                label="0th",
            )
            ax[1].plot(
                t,
                result[1][:, i],
                label="1st",
            )
            ax[2].plot(
                t,
                result[2][:, i],
                label="2nd",
            )
        fig.tight_layout()

        fig2, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(12, 5))
        markersty = dict(marker="o", markerfacecolor="none", markersize=2)
        ax[0].plot(result[0][:, 0], result[0][:, 1], result[0][:, 2], **markersty)
        ax[1].plot(result[1][:, 0], result[1][:, 1], result[1][:, 2], **markersty)
        ax[2].plot(result[2][:, 0], result[1][:, 1], result[2][:, 2], **markersty)
        fig2.tight_layout()
        fig2.suptitle(f"alpha={alpha:.4f}, gamma={gamma:.4f}")

        os.makedirs(f"figures/rf/{SUBDIR_NAME}", exist_ok=True)

        # plt.show()
        img_name = f"{SUBDIR_NAME}_{index:03d}"
        if not sol.success:
            fig.savefig(
                f"figures/rf/{SUBDIR_NAME}/{img_name}_ts_FAIL.png", dpi=300
            )
            fig2.savefig(
                f"figures/rf/{SUBDIR_NAME}/{img_name}_3d_FAIL.png", dpi=300
            )
        else:
            fig.savefig(f"figures/rf/{SUBDIR_NAME}/{img_name}_ts.png", dpi=300)
            fig2.savefig(f"figures/rf/{SUBDIR_NAME}/{img_name}_3d.png", dpi=300)

        plt.close("all")
        # save_allfigs('lorenz')


def loop(param, SUBDIR_NAME="train"):
    end_time = 40
    index, alpha = param
    t = np.linspace(0, end_time, int(end_time / 1e-3) + 1, endpoint=True)
    X_0 = [0, 0, 0]  # [0, 0, 0, 0, 0.1, 0.1]
    solve(t, X_0, alpha=alpha, gamma=0.6, index=int(index), SUBDIR_NAME=SUBDIR_NAME)


def param_space(num_samples):
    # l_bounds = [0.1]
    # u_bounds = [1]
    l_bounds = [0.8]
    u_bounds = [1.2]
    sampler = qmc.LatinHypercube(d=len(l_bounds), seed=0)
    params = sampler.random(n=int(num_samples))
    params = qmc.scale(params, l_bounds, u_bounds)
    indices = np.arange(1, len(params) + 1).reshape(-1, 1)
    params = np.concatenate([indices, params], axis=1)
    return params.tolist()

def process_data():
    names = ["train", "valid", "test"]
    for n in names:
        files = glob.glob(f"data/rf/{n}/*.csv")
        dataset = []
        for f in files:
            dataset.append(pd.read_csv(f))
        dataset = pd.concat(dataset, axis=0).reset_index(drop=True)
        dataset.to_csv(f"data/{n}_rf.csv", index=False)


if __name__ == "__main__":
    NUM_THREADS = os.cpu_count()
    SUBDIR_NAME = "train"
    shutil.rmtree(f"figures/rf/{SUBDIR_NAME}", ignore_errors=True)
    shutil.rmtree(f"data/rf/{SUBDIR_NAME}", ignore_errors=True)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        param = np.linspace(0.8, 1.2, 40, endpoint=True).reshape(-1, 1)
        indices = np.arange(1, len(param) + 1).reshape(-1, 1)
        paramspace = np.concatenate([indices, param], axis=1).tolist()
        pool.map(partial(loop, SUBDIR_NAME=SUBDIR_NAME), paramspace)
        pool.close()
        pool.join()

    SUBDIR_NAME = "valid"
    shutil.rmtree(f"figures/rf/{SUBDIR_NAME}", ignore_errors=True)
    shutil.rmtree(f"data/rf/{SUBDIR_NAME}", ignore_errors=True)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        pool.map(partial(loop, SUBDIR_NAME=SUBDIR_NAME), param_space(10))
        pool.close()
        pool.join()

    SUBDIR_NAME = "test"
    shutil.rmtree(f"figures/rf/{SUBDIR_NAME}", ignore_errors=True)
    shutil.rmtree(f"data/rf/{SUBDIR_NAME}", ignore_errors=True)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        pool.map(partial(loop, SUBDIR_NAME=SUBDIR_NAME), param_space(40))
        pool.close()
        pool.join()
    
    process_data()
