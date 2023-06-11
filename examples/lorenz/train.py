import os
import sys

sys.path.append(os.path.abspath(os.curdir))

from architectures.n_c import Net_C
from architectures.n_ag import Net_AG
from architectures.n_dp import Net_DP
from utils.trainer import Trainer
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np


def train_n_c():
    net = Net_C(1, 530, 8, 3)
    trainer = Trainer(net)
    trainer.setup_dataloader(
        batchsize,
        data_train,
        data_train,
        ["t"],
        ["x", "y", "z"],
        ["xDot", "yDot", "zDot"],
        ["xDDot", "yDDot", "zDDot"],
    )
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        save_name=f"{name_eqn}/n_c",
    )


def train_n_ag(loss_fn="mse"):
    net = Net_AG(1, 530, 8, 3)
    trainer = Trainer(net)
    trainer.setup_dataloader(
        batchsize,
        data_train,
        data_train,
        ["t"],
        ["x", "y", "z"],
        ["xDot", "yDot", "zDot"],
        ["xDDot", "yDDot", "zDDot"],
    )
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        loss_fn=loss_fn,
        save_name=f"{name_eqn}/n_ag",
    )

def train_n_dp():
    net = Net_DP(1, 300, 8, 3)
    trainer = Trainer(net)
    trainer.setup_dataloader(
        batchsize,
        data_train,
        data_train,
        ["t"],
        ["x", "y", "z"],
        ["xDot", "yDot", "zDot"],
        ["xDDot", "yDDot", "zDDot"],
    )
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        multioptim=True,
        save_name=f"{name_eqn}/n_dp",
    )


if __name__ == "__main__":
    col_I = [
        "t",
    ]
    col_y = ["x", "y", "z"]
    col_yDot = ["xDot", "yDot", "zDot"]
    col_yDDot = ["xDDot", "yDDot", "zDDot"]
    col_args = [col_I, col_y, col_yDot, col_yDDot]

    batchsize = 512
    initial_lr = 5e-4
    epochs = 500
    lr_halflife = 125

    name_eqn = f"lorenz"
    os.makedirs(f'models/{name_eqn}', exist_ok=True)
    path_data = f"data/train_{name_eqn}.csv"
    data_train = pd.read_csv(path_data)

    train_n_c()
    # train_n_dp()
    # train_n_ag()
    # train_n_ag(loss_fn='wmse_errorbased')