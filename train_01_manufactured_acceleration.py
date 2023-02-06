from modules.n_c import Net_C
from modules.n_ag import Net_AG
from modules.n_dp import Net_DP
from modules.trainer import Trainer
import pandas as pd


def train_n_c():
    net = Net_C(1, 350, 8, 1).cuda()
    trainer = Trainer(net, 'n_c')
    trainer.setup_dataloader(batchsize, data_train, data_train, ['x'], ['y'], ['yDot'], ['yDDot'])
    trainer.fit(epochs=epochs, initial_lr=initial_lr, lr_halflife=lr_halflife)

def train_n_ag():
    net = Net_AG(1, 350, 8, 1).cuda()
    trainer = Trainer(net, 'n_ag')
    trainer.setup_dataloader(batchsize, data_train, data_train, ['x'], ['y'], ['yDot'], ['yDDot'])
    trainer.fit(epochs=epochs, initial_lr=initial_lr, lr_halflife=lr_halflife)

def train_n_dp():
    net = Net_DP(1, 200, 8, 1).cuda()
    trainer = Trainer(net, 'n_dp')
    trainer.setup_dataloader(batchsize, data_train, data_train, ['x'], ['y'], ['yDot'], ['yDDot'])
    trainer.fit(epochs=epochs, initial_lr=initial_lr, lr_halflife=lr_halflife)


if __name__ == '__main__':
    path_data = 'data/01_manufactured_acceleration.csv'
    data_train = pd.read_csv(path_data)
    
    epochs = 5000
    batchsize = 128
    initial_lr = 1e-3
    lr_halflife = 500
    
    train_n_c()
    train_n_ag()
    train_n_dp()
