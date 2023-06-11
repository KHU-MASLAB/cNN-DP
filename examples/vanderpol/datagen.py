import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import matplotlib.pyplot as plt
from utils.integrate import rkint
from utils.snippets import plot_template
from scipy.integrate import solve_ivp
from functools import partial
import pandas as pd
import shutil
from scipy.stats import qmc
import glob


class VDP:
    def __init__(self, mu=1, dset="train") -> None:
        self.__dict__.update(locals())

    def vdp(self, t, Y):
        x, xDot = Y
        YDot = np.zeros_like(Y)
        YDot[0] = xDot
        YDot[1] = self.mu * (1 - x**2) * xDot - x + np.sin(t)
        return YDot

    def solve(self, t, Y0=np.array([0, 0]), plot=True):
        func = self.vdp
        result = solve_ivp(
            func,
            (t[0], t[-1]),
            y0=Y0,
            method="Radau",
            t_eval=t,
            first_step=t[1] - t[0],
            max_step=t[1] - t[0],
        )
        yddot = self.vdp(0, result.y)[1].reshape(1, -1)
        result = np.concatenate([result.y, yddot], axis=0)
        if plot:
            plot_template()
            fig, ax = plt.subplots(3, 1)
            for i in range(3):
                # ax[i].plot(t / (2 * np.pi / self.omega), result[i])
                ax[i].plot(t, result[i])
            fig.tight_layout()
            fig.savefig(f"figures/vdp/{self.dset}/mu_{self.mu:.4f}.png", dpi=200)
            data = pd.DataFrame(
                np.concatenate(
                    [t.reshape(1, -1), np.full_like(t.reshape(1, -1), self.mu), result],
                    axis=0,
                ).T,
                columns=["t", "mu", "y", "yDot", "yDDot"],
            )
            data.to_csv(f"data/vdp/{self.dset}/mu_{self.mu:.4f}.csv", index=False)
            plt.close("all")
        return result

def process_data():
    names = ["train", "valid", "test"]
    for n in names:
        files = glob.glob(f"data/vdp/{n}/*.csv")
        dataset = []
        for f in files:
            dataset.append(pd.read_csv(f))
        dataset = pd.concat(dataset, axis=0).reset_index(drop=True)
        dataset.to_csv(f"data/{n}_vdp.csv", index=False)


if __name__ == "__main__":
    dsets = ["train", "valid", "test"]
    shutil.rmtree(f"figures/vdp", ignore_errors=True)
    shutil.rmtree(f"data/vdp", ignore_errors=True)
    for d in dsets:
        os.makedirs(f"figures/vdp/{d}", exist_ok=True)
        os.makedirs(f"data/vdp/{d}", exist_ok=True)

    t = np.linspace(0, 10, 1001)
    # training set
    for mu in np.linspace(0, 5, 11, endpoint=True):
        vdp = VDP(mu=mu, dset="train")
        vdp.solve(t)

    # validation set
    sampler = qmc.LatinHypercube(1, seed=1)
    vals = sampler.random(n=11) * 5
    for mu in vals.flatten():
        vdp = VDP(mu=mu, dset="valid")
        vdp.solve(t)

    # test set
    sampler = qmc.LatinHypercube(1, seed=0)
    vals = sampler.random(n=11) * 5
    for mu in vals.flatten():
        vdp = VDP(mu=mu, dset="test")
        vdp.solve(t)
        
    process_data()
