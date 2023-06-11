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
        "red",
    ]
    ylabels = ["$x$", "$\dot{x}$", "$\ddot{x}$"]
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
            ylim=(1e-4, 1e4),
            xticks=np.linspace(1, 90, 6, endpoint=True, dtype=np.int8),
        )
    increase_leglw(
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=len(models))
    )
    adjust(fig)
    # fig.tight_layout()
    fig.savefig("figures/vdp/loss.png", dpi=300)

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
            print(models[j].model["error_history"]["train"][:, i].min())
            print(models[j].model["error_history"]["valid"][:, i].min())
        ax[i].set(
            xlabel="Epochs",
            ylabel=f"MSE of unscaled {ylabels[i]}",
            yscale="log",
            ylim=(1e-3, 1e3),
            xticks=np.linspace(1, 90, 6, endpoint=True, dtype=np.int8),
        )
    increase_leglw(
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=len(models))
    )
    adjust(fig)
    fig.tight_layout()
    fig.savefig("figures/vdp/error.png", dpi=300)


def plot_prediction(model):
    if model.net_type == "n_ag":
        if model.model_info["loss_fn"] == "mse":
            color = "blue"
        else:
            color = "limegreen"
    elif model.net_type == "n_c":
        color = "black"
    elif model.net_type == "n_dp":
        color = "red"

    coords = ["$x$", "$\dot{x}$", "$\ddot{x}$"]
    n_steps = 1001  # per simulation
    for i in range(11):
        s = testdata.iloc[n_steps * i : n_steps * (i + 1)]
        s_input = df2tensor(s[["t", "mu"]])
        time = s_input[:, 0]

        py, pdy, pddy = model.predict(s_input)
        ly, ldy, lddy = [
            s[
                [
                    "y",
                ]
            ],
            s[
                [
                    "yDot",
                ]
            ],
            s[
                [
                    "yDDot",
                ]
            ],
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
        fig, ax = plt.subplots(3, 1, figsize=(2.7, 5))
        sty_l = dict(c="grey", linestyle="solid", lw=1)
        sty_p = dict(c=color, linestyle="dashed", lw=1.5)

        ax[0].plot(time, ly.flatten(), **sty_l)
        ax[1].plot(time, ldy.flatten(), **sty_l)
        ax[2].plot(time, lddy.flatten(), **sty_l)
        try:
            ax[0].plot(time, py.flatten(), **sty_p)
            ax[1].plot(time, pdy.flatten(), **sty_p)
        except:
            pass
        ax[2].plot(time, pddy.flatten(), **sty_p)
        for k in range(3):
            ax[k].set_ylabel(f"{coords[k]}")
        ax[-1].set_xlabel("$t$")
        fig.tight_layout()
        os.makedirs("figures/vdp", exist_ok=True)
        if model.model_info["loss_fn"] == "mse":
            fig.savefig(
                f"figures/vdp/{model.net_type}_test_{i+1}_mu_{s_input[0,1]:.4f}.png",
                dpi=200,
            )
        else:
            fig.savefig(
                f"figures/vdp/{model.net_type}w_test_{i+1}_mu_{s_input[0,1]:.4f}.png",
                dpi=200,
            )
        plt.close("all")
        print(f"figure {i+1} saved")


def eval_r2():
    valid_data = pd.read_csv("data/test_vdp.csv")
    valid_input = df2tensor(valid_data[["t", "mu"]])
    valid_label = df2tensor(
        valid_data[
            [
                "yDDot",
            ]
        ]
    )
    p_c = Nc.predict(valid_input)
    p_ag = Nag.predict(valid_input)
    p_dp = Ndp.predict(valid_input)
    preds = [p_c, p_ag, p_dp]
    labels = ["y", "yDot", "yDDot"]
    colors = ["black", "blue", "red"]
    ylabels = ["$x^{ref}$", "$\dot{x}^{ref}$", "$\ddot{x}^{ref}$"]
    xlabels = [
        ["", "$x^{AG}$", "$x^{DP}$"],
        ["", "$\dot{x}^{AG}$", "$\dot{x}^{DP}$"],
        ["$\ddot{x}^{C}$", "$\ddot{x}^{AG}$", "$\ddot{x}^{DP}$"],
    ]
    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    for i in range(3):
        for j in range(3):
            try:
                ax[i, j].plot(
                    valid_data[labels[i]].to_numpy(),
                    preds[j][i].flatten(),
                    marker="o",
                    ms=0.1,
                    ls="none",
                    mfc=colors[j],
                    mec=colors[j],
                )
                add_textbox(
                    ax[i, j],
                    f"$R^2$={r2_score(valid_data[labels[i]].to_numpy(),preds[j][i].flatten()):.4f}",
                    loc=4,
                    fontsize=10,
                )
            except:
                pass
            ax[i, j].grid(False)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel(xlabels[i][j])
    for i in range(3):
        ax[i, 0].set_ylabel(ylabels[i])
    fig.tight_layout()
    fig.savefig("figures/vdp/r2.png", dpi=300)

    print(r2_score(valid_label.numpy(), p_c[-1], multioutput="raw_values"))
    print(r2_score(valid_label.numpy(), p_ag[-1], multioutput="raw_values"))
    print(r2_score(valid_label.numpy(), p_dp[-1], multioutput="raw_values"))
    pass



Nc = NetInterface("models/vdp/n_c.pt")
Nag = NetInterface("models/vdp/n_ag.pt")
Ndp = NetInterface("models/vdp/n_dp.pt")
testdata = pd.read_csv("data/test_vdp.csv")
t = np.linspace(0, 10, 1001, endpoint=True)

matplotlib.use("Agg")
plot_template(11)
plot_prediction(Nc)
plot_prediction(Nag)
plot_prediction(Ndp)
plot_loss()
eval_r2()
