import os
import sys  # nopep8

sys.path.append(os.path.abspath(os.curdir))  # nopep8

import torch
import pandas as pd
import numpy as np
from architectures.interface import NetInterface
from utils.snippets import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import glob


def plot_prediction():
    # Plot
    style_label = dict(color="grey", linestyle="dashed", lw=0.8)
    style_c = dict(color="black", linestyle="solid", linewidth=1.3)
    style_ag = dict(color="blue", linestyle="solid", linewidth=1)
    style_ag_ = dict(color="limegreen", linestyle="dashed", linewidth=1.3)
    style_dp = dict(color="red", linestyle="solid", linewidth=1)

    # Trajectories
    style_label.update({"linestyle": "solid"})
    style_c.update({"linewidth": 1, "linestyle": "dashed"})
    style_ag.update({"linewidth": 1, "linestyle": "dashed"})
    style_ag_.update({"linewidth": 1, "linestyle": "dashed"})
    style_dp.update({"linewidth": 1, "linestyle": "dashed"})
    styles = [style_c, style_ag, style_ag_, style_dp]
    labels = [
        ["$x$", "$y$", "$z$"],
        ["$\dot{x}$", "$\dot{y}$", "$\dot{z}$"],
        ["$\ddot{x}$", "$\ddot{y}$", "$\ddot{z}$"],
    ]
    coords = ["x", "y", "z"]
    names = [coords, [f"{c}Dot" for c in coords], [f"{c}DDot" for c in coords]]

    models = [model_c, model_ag, model_dp]
    for k, m in enumerate(models):
        pred = m.predict(data[["t"]].to_numpy())

        # Trajectories N_C
        fig, axes = plt.subplots(3, 1, figsize=(4, 9), subplot_kw={"projection": "3d"})
        axes[0].plot(data["x"], data["y"], data["z"], **style_label)
        axes[1].plot(data["xDot"], data["yDot"], data["zDot"], **style_label)
        axes[2].plot(data["xDDot"], data["yDDot"], data["zDDot"], **style_label)

        for i in range(3):
            try:
                axes[i].plot(pred[i][:, 0], pred[i][:, 1], pred[i][:, 2], **styles[k])
            except:
                pass
            axes[i].set(xlabel=labels[i][0], ylabel=labels[i][1], zlabel=labels[i][2])
        try:
            add_textbox(
                axes[0],
                f"$R^2$={r2_score(data[['x','y','z']].to_numpy(), pred[0]):.4f}",
                loc=1,
                fontsize=12,
            )
            add_textbox(
                axes[1],
                f"$R^2$={r2_score(data[['xDot','yDot','zDot']].to_numpy(), pred[1]):.4f}",
                loc=1,
                fontsize=12,
            )
        except:
            pass
        add_textbox(
            axes[2],
            f"$R^2$={r2_score(data[['xDDot','yDDot','zDDot']].to_numpy(), pred[2]):.4f}",
            loc=1,
            fontsize=12,
        )
        # remove grid
        for i in range(3):
            axes[i].grid(False)
        fig.tight_layout()

        # Time series plot
        ylabels = ["$U$", "$\dot{U}$", "$\ddot{U}$"]
        # fig, axes=plt.subplots(3,3, figsize=(7,6))
        # for i in range(3):
        #     for j in range(3):
        #         axes[i,j].plot(data['t'],data[names[i][j]], **style_label)
        #         try:
        #             axes[i,j].plot(data['t'],pred[i][:,j], **styles[k])
        #         except:
        #             pass
        #         axes[i,j].set_ylabel(None)
        #         axes[i,j].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        # for i in range(3):
        #     axes[0,i].set_title(f"${coords[i]}$",fontsize=20)
        #     axes[i,0].set_ylabel(ylabels[i],fontsize=14)
        #     axes[-1,i].set_xlabel('$t$')
        fig, axes = plt.subplots(3, 1, figsize=(2.5, 9))
        for i in range(3):
            axes[i].plot(data["t"], data[names[i]], **style_label)
            try:
                axes[i].plot(data["t"], pred[i], **styles[k])
            except:
                pass
            axes[i].set_ylabel(ylabels[i], fontsize=14)
        axes[-1].set_xlabel("$t$")
        fig.tight_layout()


def plot_loss_hist():
    style_c = dict(
        color="black",
        linewidth=1.3,
    )
    style_ag = dict(
        color="blue",
        linewidth=1.3,
    )
    style_dp = dict(color="red", linewidth=1)

    types = [
        "train",
    ]
    # names = ['$x,y,z$', '$\dot{x},\dot{y},\dot{z}$',
    #          '$\ddot{x},\ddot{y},\ddot{z}$']
    names = ["$U$", "$\dot{U}$", "$\ddot{U}$"]
    lss = ["solid", "dashed"]
    labels = [
        "$\mathcal{N}_{C}$",
        "$\mathcal{N}_{AG}$",
        "$\mathcal{N}_{DP}$",
    ]
    types_ = [" (Training)", " (Validation)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    x = np.arange(len(model_dp.loss_history[t])) + 1
    for i in range(3):
        for j, t in enumerate(types):
            axes[i].plot(
                x,
                model_c.loss_history[t][:, i],
                ls=lss[j],
                label=labels[0] + types_[j],
                **style_c,
            )
            axes[i].plot(
                x,
                model_ag.loss_history[t][:, i],
                ls=lss[j],
                label=labels[1] + types_[j],
                **style_ag,
            )
            axes[i].plot(
                x,
                model_dp.loss_history[t][:, i],
                ls=lss[j],
                label=labels[3] + types_[j],
                **style_dp,
            )
        axes[i].set_yscale("log")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(f"Loss of {names[i]}")
        axes[i].set_ylim(1e-3, 1e8)  # lorenz
    increase_leglw(
        axes[1].legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=len(labels), fontsize=16
        )
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.12, top=0.8, hspace=0.3, wspace=0.4
    )
    fig.savefig("figures/lorenz/loss.png", dpi=200)


def plot_error_hist():
    style_c = dict(
        color="black",
        linewidth=1.3,
    )
    style_ag = dict(
        color="blue",
        linewidth=1.3,
    )
    style_dp = dict(color="red", linewidth=1)

    types = [
        "train",
    ]
    names = ["$x,y,z$", "$\dot{x},\dot{y},\dot{z}$", "$\ddot{x},\ddot{y},\ddot{z}$"]
    lss = ["solid", "dashed"]
    labels = [
        "$\mathcal{N}_{C}$",
        "$\mathcal{N}_{AG}$",  # '$\mathcal{N}_{AGw}$',
        "$\mathcal{N}_{DP}$",
    ]
    types_ = [" (Training)", " (Validation)"]

    x = np.arange(len(model_dp.loss_history[t])) + 1
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for i in range(3):
        for j, t in enumerate(types):
            axes[i].plot(
                x,
                model_c.error_history[t][:, i],
                ls=lss[j],
                label=labels[0] + types_[j],
                **style_c,
            )
            axes[i].plot(
                x,
                model_ag.error_history[t][:, i],
                ls=lss[j],
                label=labels[1] + types_[j],
                **style_ag,
            )
            axes[i].plot(
                x,
                model_dp.error_history[t][:, i],
                ls=lss[j],
                label=labels[2] + types_[j],
                **style_dp,
            )
        axes[i].set_yscale("log")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylim(1e-1, 1e7)  # lorenz
        axes[i].set_ylabel(f"MSE of unscaled {names[i]}")


if __name__ == "__main__":
    # Load data
    name_eqn = "lorenz"
    path_models = f"models/{name_eqn}"
    path_data = f"data/train_{name_eqn}.csv"
    data = pd.read_csv(path_data)

    model_c = NetInterface(f"{path_models}/n_c.pt")
    model_ag = NetInterface(f"{path_models}/n_ag.pt")
    model_dp = NetInterface(f"{path_models}/n_dp.pt")

    # Set templates and plot
    path_figures = "figures"
    plot_template()
    plot_prediction()
    plot_loss_hist()
    plot_error_hist()
    # plt.show()
    os.makedirs(f"figures/{name_eqn}", exist_ok=True)
    save_allfigs(f"{name_eqn}_result")
