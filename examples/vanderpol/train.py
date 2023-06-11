import os
import sys

sys.path.append(os.path.abspath(os.curdir))

from architectures.n_c import Net_C
from architectures.n_ag import Net_AG
from architectures.n_dp import Net_DP
from utils.trainer import Trainer
import pandas as pd
import glob
import numpy as np
import os

def train_n_c():
    net = Net_C(2, 530, 4, 1)
    trainer = Trainer(net)
    trainer.setup_dataloader(batchsize, data_train, data_valid, *col_args)
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        save_name="vdp/n_c",
    )


def train_n_ag(loss_fn="mse"):
    net = Net_AG(2, 530, 4, 1)
    trainer = Trainer(net)
    trainer.setup_dataloader(batchsize, data_train, data_valid, *col_args)
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        loss_fn=loss_fn,
        save_name=f"vdp/n_ag_{loss_fn}",
    )


def train_n_dp():
    net = Net_DP(2, 300, 4, 1)
    trainer = Trainer(net)
    trainer.setup_dataloader(batchsize, data_train, data_valid, *col_args)
    trainer.fit(
        epochs=epochs,
        initial_lr=initial_lr,
        lr_halflife=lr_halflife,
        save_name="vdp/n_dp",
    )


if __name__ == "__main__":
    os.makedirs("models/vdp", exist_ok=True)
    col_I = ["t", "mu"]
    col_y = [
        "y",
    ]
    col_yDot = ["yDot"]
    col_yDDot = ["yDDot"]
    col_args = [col_I, col_y, col_yDot, col_yDDot]

    data_train = pd.read_csv("data/train_vdp.csv")
    data_valid = pd.read_csv("data/valid_vdp.csv")

    batchsize = 64
    initial_lr = 1e-3
    epochs = 90
    lr_halflife = 30

    # train_n_c()
    train_n_dp()
    # train_n_ag()
