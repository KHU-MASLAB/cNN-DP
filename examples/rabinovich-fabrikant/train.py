import os, sys

sys.path.append(os.path.abspath(os.curdir))

from architectures.n_c import Net_C
from architectures.n_ag import Net_AG
from architectures.n_dp import Net_DP
from utils.trainer import Trainer
import pandas as pd
import glob
import numpy as np


def train_n_c():
    net = Net_C(2, 700, 6, 3)
    trainer = Trainer(net)
    trainer.setup_dataloader(batchsize, data_train, data_valid, *col_args)
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        save_name=f"{name_eqn}/n_c",
    )


def train_n_ag():
    net = Net_AG(2, 700, 6, 3)
    trainer = Trainer(net)
    trainer.setup_dataloader(batchsize, data_train, data_valid, *col_args)
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        loss_fn="mse",
        save_name=f"{name_eqn}/n_ag",
    )


def train_n_dp():
    net = Net_DP(2, 400, 6, 3)
    trainer = Trainer(net)
    trainer.setup_dataloader(batchsize, data_train, data_valid, *col_args)
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        save_name=f"{name_eqn}/n_dp",
    )


if __name__ == "__main__":
    col_I = ["t", "alpha"]
    col_y = ["x", "y", "z"]
    col_yDot = ["xDot", "yDot", "zDot"]
    col_yDDot = ["xDDot", "yDDot", "zDDot"]
    col_args = [col_I, col_y, col_yDot, col_yDDot]

    name_eqn = "rf"
    data_train = pd.read_csv(f"data/train_{name_eqn}.csv")
    data_valid = pd.read_csv(f"data/valid_{name_eqn}.csv")

    # reduce sampling rate 1e-3 -> 1e-2
    t = np.linspace(0, 40, int(40 / 1e-2) + 1, endpoint=True).round(4)
    data_train["t"] = data_train["t"].round(4)
    data_train = data_train[data_train["t"].isin(t)]
    data_valid["t"] = data_valid["t"].round(4)
    data_valid = data_valid[data_valid["t"].isin(t)]
    # reduce sampling rate 1e-3 -> 1e-2

    batchsize = 256
    initial_lr = 1e-3
    epochs = 300
    lr_halflife = 100

    train_n_c()
    # train_n_dp()
    # train_n_ag()
