import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from .utils import initializer

class Net_AG(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(Net_AG, self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.width = width
        self.output_dim = output_dim
        self.net_type = 'n_ag'
        
        self.Linears = nn.ModuleDict()
        self.Linears['in'] = nn.Linear(self.input_dim, self.width)
        for i in range(depth - 1):
            self.Linears[f'mid{i}'] = nn.Linear(self.width, self.width)
        self.Linears['out'] = nn.Linear(self.width, self.output_dim)
        self.apply(initializer)
    
    def forward(self, x):
        assert len(x.shape) == 2
        time = x[:, 0].unsqueeze(1).requires_grad_(True)  # To assign computational graph
        if self.input_dim > 1:
            params = x[:, 1:]
        x = torch.cat([time, params], dim=1) if 'params' in locals() else time
        
        # Dense net predicting y
        x = self.Linears['in'](x)
        x = F.gelu(x)
        for i in range(self.depth - 1):
            x = self.Linears[f'mid{i}'](x)
            x = F.gelu(x)
        y = self.Linears['out'](x)
        
        # Differential operators for yDot and yDDot
        yDot = torch.empty_like(y)
        yDDot = torch.empty_like(y)
        for i in range(y.shape[1]):
            yDot[:, i] = grad(y, time, grad_outputs=torch.ones_like(y[:, i].unsqueeze(1)), create_graph=True)[0].squeeze(1)
            yDDot[:, i] = grad(yDot, time, grad_outputs=torch.ones_like(yDot[:, i]).unsqueeze(1), create_graph=True)[0].squeeze(1)
        time.requires_grad_(False)
        
        return y, yDot, yDDot
