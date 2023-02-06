import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import load_model
from modules.utils import PlotTemplate
from modules.utils import IncreaseLegendLinewidth

# Load data
path_data = 'data/01_manufactured_acceleration.csv'
data = pd.read_csv(path_data)

# Load models
path_models = 'models'
model_c = torch.load(f'{path_models}/n_c.pt')
model_ag = torch.load(f'{path_models}/n_ag.pt')
model_dp = torch.load(f'{path_models}/n_dp.pt')
n_c = load_model(model_c)
n_ag = load_model(model_ag)
n_dp = load_model(model_dp)

# Load normalization params
mean_x = model_c['mean_x']
mean_y = model_c['mean_y']
mean_yDot = model_c['mean_yDot']
mean_yDDot = model_c['mean_yDDot']
std_x = model_c['std_x']
std_y = model_c['std_y']
std_yDot = model_c['std_yDot']
std_yDDot = model_c['std_yDDot']

# Normalize input
x = torch.FloatTensor(data['x'].to_numpy()).unsqueeze(1).cuda()  # time vector
x = (x - mean_x) / std_x

# Forward
with torch.no_grad():
    pred_c = n_c(x)
    pred_dp = torch.cat(n_dp(x), dim=1)
pred_ag = torch.cat(n_ag(x), dim=1).detach()

# Inverse normalize
x = x * std_x + mean_x
pred_c = pred_c * std_yDDot + mean_yDDot
pred_ag[:, 0] = pred_ag[:, 0] * std_y + mean_y
pred_ag[:, 1] = pred_ag[:, 1] * (std_y / std_x[0])
pred_ag[:, 2] = pred_ag[:, 2] * (std_y / std_x[0] ** 2)
pred_dp[:, 0] = pred_dp[:, 0] * std_y + mean_y
pred_dp[:, 1] = pred_dp[:, 1] * std_yDot + mean_yDot
pred_dp[:, 2] = pred_dp[:, 2] * std_yDDot + mean_yDDot

# Plot
path_figures = 'figures'
PlotTemplate()
style_label = dict(color='grey', linestyle='dashed', linewidth=1.5)
style_c = dict(color='black', linestyle='solid', linewidth=1.3)
style_ag = dict(color='blue', linestyle='solid', linewidth=1.3)
style_dp = dict(color='red', linestyle='solid', linewidth=1.3)

fig, axes = plt.subplots(3, 3, figsize=(17, 9))
for j in range(3):
    axes[0, j].plot(data['x'], data['y'], **style_label)
    axes[1, j].plot(data['x'], data['yDot'], **style_label)
    axes[2, j].plot(data['x'], data['yDDot'], **style_label)
axes[2, 0].plot(x.flatten().cpu(), pred_c.cpu(), **style_c)  # Net_C
for i in range(3):
    axes[i, 1].plot(x.flatten().cpu(), pred_ag[:, i].cpu(), **style_ag)  # Net_AG
    axes[i, 2].plot(x.flatten().cpu(), pred_dp[:, i].cpu(), **style_dp)  # Net_DP
axes[0, 0].set(ylabel='$y$')
axes[1, 0].set(ylabel='$\dot{y}$')
axes[2, 0].set(ylabel='$\ddot{y}$')
for i in range(3):
    axes[2, i].set(xlabel='$t$')
    for j in range(3):
        axes[i, j].set(xlim=(-2.5, 7.5), xticks=np.linspace(-2.5, 7.5, 5, endpoint=True))
fig.tight_layout()
fig.savefig(f"{path_figures}/01_manufactured_acceleration.png", dpi=300)
