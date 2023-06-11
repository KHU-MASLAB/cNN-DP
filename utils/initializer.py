import torch
import torch.nn as nn
import math


def Initializer(layer):
    if isinstance(layer, (nn.Linear)):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # setting a=0 drives Ndp and Nag to diverge
        nn.init.kaiming_uniform_(
            layer.weight, a=math.sqrt(5), nonlinearity="leaky_relu"
        )
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
