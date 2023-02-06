import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def initializer(layer):
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    if isinstance(layer, (nn.Linear)):
        torch.nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5), nonlinearity='leaky_relu')
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

def mse(label, pred):
    return torch.mean(torch.square(label - pred))

def load_model(model: dict):
    from .n_c import Net_C
    from .n_ag import Net_AG
    from .n_dp import Net_DP
    
    net_type = model['net_type']
    input_dim = model['input_dim']
    width = model['width']
    depth = model['depth']
    output_dim = model['output_dim']
    if net_type == 'n_c':
        net = Net_C(input_dim, width, depth, output_dim)
    elif net_type == 'n_ag':
        net = Net_AG(input_dim, width, depth, output_dim)
    elif net_type == 'n_dp':
        net = Net_DP(input_dim, width, depth, output_dim)
    net.load_state_dict(model['state_dict'])
    return net.cuda()

def PlotTemplate(fontsize=15):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes3d.grid'] = True
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.labelsize'] = 1.5 * fontsize
    plt.rcParams['axes.titlesize'] = 2 * fontsize
    plt.rcParams['xtick.labelsize'] = 0.7 * fontsize
    plt.rcParams['ytick.labelsize'] = 0.7 * fontsize

def IncreaseLegendLinewidth(leg, linewidth: float = 3):
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth)

def AddTextBox(ax, string, loc: int = 3, fontsize: int = 12):
    artist = AnchoredText(string, loc=loc, prop={'fontsize': fontsize})
    ax.add_artist(artist)
