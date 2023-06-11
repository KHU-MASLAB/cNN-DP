import os, sys

sys.path.append(os.path.abspath(os.curdir))

import matplotlib
from architectures.interface import NetInterface
from utils.snippets import *
import pandas as pd
from utils.snippets import *
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import numpy as np

def plot_loss():
    models = [Nc, Nag, Ndp]
    exprs = [
        "$\mathcal{N}_{C}$",
        "$\mathcal{N}_{AG}$",
        "$\mathcal{N}_{DP}$",
    ]
    colors = [
        "black",
        "blue",
        "limegreen",
        "red",
    ]
    ylabels = ["$U$", "$\dot{U}$", "$\ddot{U}$"]
    d = np.arange(len(Nc.model["loss_history"]["train"])) + 1

    def adjust(fig):
        return fig.subplots_adjust(
            bottom=0.15, top=0.8, left=0.08, right=0.98, hspace=0, wspace=0.32
        )

    # loss function plot
    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    for i in range(3):  # diff orders
        for j in range(len(models)):  # models
            ax[i].plot(
                d,
                models[j].model["loss_history"]["train"][:, i],
                color=colors[j],
                linestyle="solid",
                lw=0.7,
                label=exprs[j] + " (Training)",
            )
            ax[i].plot(
                d,
                models[j].model["loss_history"]["valid"][:, i],
                color=colors[j],
                linestyle="dashed",
                lw=0.6,
                label=exprs[j] + " (Validation)",
            )
        ax[i].set(
            xlabel="Epochs",
            ylabel=f"Loss of {ylabels[i]}",
            yscale="log",
            ylim=(1e-3, 1e8),
        )
    increase_leglw(
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=len(models))
    )
    adjust(fig)
    fig.savefig("figures/rf/loss.png", dpi=300)

    # unscaled MSE plot
    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    for i in range(3):  # diff orders
        for j in range(len(models)):  # models
            ax[i].plot(
                d,
                models[j].model["error_history"]["train"][:, i],
                color=colors[j],
                linestyle="solid",
                lw=0.7,
                label=exprs[j] + " (Training)",
            )
            ax[i].plot(
                d,
                models[j].model["error_history"]["valid"][:, i],
                color=colors[j],
                linestyle="dashed",
                lw=0.6,
                label=exprs[j] + " (Validation)",
            )
        ax[i].set(
            xlabel="Epochs",
            ylabel=f"MSE of {ylabels[i]}",
            yscale="log",
            ylim=(1e-4, 1e4),
        )
    increase_leglw(
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=len(models))
    )
    adjust(fig)
    # fig.tight_layout()
    fig.savefig("figures/rf/error.png", dpi=300)


def plot_traj(model):
    if model.net_type == "n_ag":
        if model.model_info["loss_fn"] == "mse":
            color = "blue"
        else:
            color = "limegreen"
    elif model.net_type == "n_c":
        color = "black"
    elif model.net_type == "n_dp":
        color = "red"

    coords = [
        ["$x$", "$y$", "$z$"],
        ["$\dot{x}$", "$\dot{y}$", "$\dot{z}$"],
        ["$\ddot{x}$", "$\ddot{y}$", "$\ddot{z}$"],
    ]
    n_steps = 4001  # per simulation
    for i in range(30):
        s = testdata.iloc[n_steps * i : n_steps * (i + 1)]
        s_input = df2tensor(s[["t", "alpha"]])
        time = s_input[:, 0]

        py, pdy, pddy = model.predict(s_input)
        ly, ldy, lddy = [
            s[["x", "y", "z"]],
            s[["xDot", "yDot", "zDot"]],
            s[["xDDot", "yDDot", "zDDot"]],
        ]
        try:
            py = py
            pdy = pdy
        except:
            pass
        pddy = pddy
        ly = ly.to_numpy()
        ldy = ldy.to_numpy()
        lddy = lddy.to_numpy()

        # Time series plot
        fig, ax = plt.subplots(3, 3, figsize=(5, 7))
        sty_l = dict(c="grey", linestyle="solid", lw=0.7)
        sty_p = dict(c=color, linestyle="dashed", lw=0.9)
        for j in range(3):
            ax[0, j].plot(time, ly[:, j], **sty_l)
            ax[1, j].plot(time, ldy[:, j], **sty_l)
            ax[2, j].plot(time, lddy[:, j], **sty_l)
            try:
                ax[0, j].plot(time, py[:, j], **sty_p)
                ax[1, j].plot(time, pdy[:, j], **sty_p)
            except:
                pass
            ax[2, j].plot(time, pddy[:, j], **sty_p)
        for k in range(3):
            for l in range(3):
                ax[k, l].set_ylabel(f"{coords[k][l]}")
            ax[-1, k].set_xlabel("Time")
            # for q in range(3):
            # ax[k,q].set(ylim=lims[k])
        fig.tight_layout()
        os.makedirs("figures/rf", exist_ok=True)
        if model.model_info["loss_fn"] == "mse":
            fig.savefig(
                f"figures/rf/{model.net_type}_test_{i+1}_alpha_{s_input[0,1]:.4f}.png",
                dpi=200,
            )
        else:
            fig.savefig(
                f"figures/rf/{model.net_type}w_test_{i+1}_alpha_{s_input[0,1]:.4f}.png",
                dpi=200,
            )

        # 3D plot
        fig, ax = plt.subplots(3, 1, figsize=(3, 7), subplot_kw={"projection": "3d"})
        ax[0].plot(ly[:, 0], ly[:, 1], ly[:, 2], **sty_l)
        ax[1].plot(ldy[:, 0], ldy[:, 1], ldy[:, 2], **sty_l)
        ax[2].plot(lddy[:, 0], lddy[:, 1], lddy[:, 2], **sty_l)
        ax[2].plot(pddy[:, 0], pddy[:, 1], pddy[:, 2], **sty_p)
        try:
            ax[0].plot(py[:, 0], py[:, 1], py[:, 2], **sty_p)
            ax[1].plot(pdy[:, 0], pdy[:, 1], pdy[:, 2], **sty_p)
        except:
            pass
        for k in range(3):
            ax[k].set_xlabel(coords[k][0])
            ax[k].set_ylabel(coords[k][1])
            ax[k].set_zlabel(coords[k][2])
            ax[k].zaxis.labelpad = 0
            # ax[k].set(xlim=lims[k], ylim=lims[k], zlim=lims[k])
        fig.subplots_adjust(
            bottom=0.05, top=1, left=0, right=0.95, wspace=0, hspace=0.1
        )
        # remove grids and score r2
        labels = [ly, ldy, lddy]
        preds = [py, pdy, pddy]
        for m in range(3):
            ax[m].grid(False)
            try:
                add_textbox(ax[m], f"$R^2$={r2_score(labels[m],preds[m]):.4f}", loc=1)
            except:
                pass
        # save
        if model.model_info["loss_fn"] == "mse":
            fig.savefig(
                f"figures/rf/{model.net_type}_test_{i+1}_alpha_{s_input[0,1]:.4f}_3d.png",
                dpi=200,
            )
        else:
            fig.savefig(
                f"figures/rf/{model.net_type}w_test_{i+1}_alpha_{s_input[0,1]:.4f}_3d.png",
                dpi=200,
            )
        plt.close("all")
        print(f"figure {i+1} saved")



Nc = NetInterface("models/rf/n_c.pt")
Nag = NetInterface("models/rf/n_ag.pt")
Ndp = NetInterface("models/rf/n_dp.pt")
testdata = pd.read_csv("data/test_rf.csv")

# reduce sampling rate
t = np.linspace(0, 40, int(40 / 1e-2) + 1, endpoint=True).round(4)
testdata["t"] = testdata["t"].round(4)
testdata = testdata[testdata["t"].isin(t)]
# reduce sampling rate

matplotlib.use("Agg")
plot_template(11)
plot_traj(Nc)
plot_traj(Nag)
plot_traj(Ndp)
plot_loss()
