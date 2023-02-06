import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import initializer

class Net_C(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(Net_C, self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.width = width
        self.output_dim = output_dim
        self.net_type = 'n_c'
        
        self.Linears = nn.ModuleDict()
        self.Linears['in'] = nn.Linear(self.input_dim, self.width)
        for i in range(depth - 1):
            self.Linears[f'mid{i}'] = nn.Linear(self.width, self.width)
        self.Linears['out'] = nn.Linear(self.width, self.output_dim)
        self.apply(initializer)
    
    
    def forward(self, x):
        assert len(x.shape) == 2
        # Dense net predicting yDDot
        x = self.Linears['in'](x)
        x = F.gelu(x)
        for i in range(self.depth - 1):
            x = self.Linears[f'mid{i}'](x)
            x = F.gelu(x)
        yDDot = self.Linears['out'](x)
        return yDDot
