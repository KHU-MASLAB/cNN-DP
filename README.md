# cNN-DP: Composite Neural Network with Differential Propagation

[![Paper](https://img.shields.io/badge/J.%20Comput.%20Phys.-496%20(2024)%20112578-red)](https://www.sciencedirect.com/science/article/pii/S0021999123006733)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jcp.2023.112578-informational)](https://doi.org/10.1016/j.jcp.2023.112578)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hits](https://hits.sh/github.com/leehyeonbeen/cNN-DP.svg?view=total&label=Hits&extraCount=1000&color=dfb317)](https://hits.sh/github.com/leehyeonbeen/cNN-DP/)

![github](https://github.com/KHU-MASLAB/cNN-DP/assets/78078652/b37e129f-4cef-4250-b958-12ada1e5e688)

This is our official PyTorch implementation of

> **cNN-DP: Composite neural network with differential propagation for impulsive nonlinear dynamics**<br>
> Hyeonbeen Lee, Seongji Han, Hee-Sun Choi, Jin-Gyun Kim<br>
> *Journal of Computational Physics*, Vol. 496, 112578, 2024

---

## TL;DR

Sequentially mapping from inputs to higher-order dynamics via a chain of neural subnetworks, termed **differential propagation**, enables accurate data-driven modeling of impulsive high-order nonlinear dynamics.

**Why impulsive nonlinear dynamics?**
Impulsive dynamics, such as earthquakes and rigid-body contacts, involve rapid and intense changes accompanied by high-frequency vibrations. Simulating them requires fine discretizations and is costly, which calls for fast data-driven surrogates. Yet conventional DNNs fail to capture such signals accurately, even with sufficient training data.

**Our contributions**
- A composite network that learns the simplest, lowest-order dynamics first and propagates it to higher-order derivatives, motivated by the fact that time integration alleviates high-frequency components
- Many orders of magnitude improvement in convergence rate and generalization accuracy over a conventional network and an autograd-based network (see [Baselines](#baselines))
- Validation on chaotic systems, earthquake measurements, rigid-body contact, and an industrial-level multibody vehicle
- No governing equations required, at a computational cost comparable to a conventional network

## Method

<img width="1108" alt="Network schematics" src="https://github.com/KHU-MASLAB/cNN-DP/assets/78078652/e640be65-35b1-4f9a-8095-7b755f0eaaf7">

Given an input $\mathcal{I} = [t, I_2, \dots, I_d]$ (time plus parameters such as initial conditions or geometry), cNN-DP stacks three MLP subnetworks as follows.

$$
\begin{aligned}
\mathcal{N}_{DP_0} &: \mathcal{I} \mapsto y^{DP} \\
\mathcal{N}_{DP_1} &: (\mathcal{I},\, y^{DP}) \mapsto \dot{y}^{DP} \\
\mathcal{N}_{DP_2} &: (\mathcal{I},\, y^{DP},\, \dot{y}^{DP}) \mapsto \ddot{y}^{DP}
\end{aligned}
$$

All three are trained at the same time, each with its own optimizer, on the sum of per-order MSEs.

$$
\mathcal{L}_{DP} = \lVert y^{DP} - y^{ref} \rVert^2 + \lVert \dot{y}^{DP} - \dot{y}^{ref} \rVert^2 + \lVert \ddot{y}^{DP} - \ddot{y}^{ref} \rVert^2
$$

The corresponding forward pass in [`architectures/n_dp.py`](architectures/n_dp.py) is shown below.

```python
y     = self.dp_0(x)
yDot  = self.dp_1(torch.cat([x, y], dim=1))
yDDot = self.dp_2(torch.cat([x, y, yDot], dim=1))
```

**Why it works.** We attribute the gain to *decoupling* the prediction of each derivative order. Learning multi-order derivatives without decoupling, i.e., in a single MLP whose higher-order outputs are obtained by automatic differentiation (the AG baseline below), forces all orders to share the same parameters. Moreover, the chain rule makes the loss terms of different orders differ by orders of magnitude, which biases training toward higher-order derivatives. In contrast, cNN-DP assigns each order its own subnetwork, so the loss terms remain on comparable scales.

We use three subnetworks, matching a second-order target such as acceleration. The idea works with any number of subnetworks, and the variable does not have to be time (we show spatial derivatives of a wave equation in Appendix A).

## Baselines

| Model | File | Output | Description |
|---|---|---|---|
| $\mathcal{N}_C$ | [`n_c.py`](architectures/n_c.py) | $\mathcal{I} \mapsto \ddot{y}$ | Conventional MLP that directly maps inputs to the target |
| $\mathcal{N}_{AG}$ | [`n_ag.py`](architectures/n_ag.py) | $\mathcal{I} \mapsto \lbrace y, \dot{y}, \ddot{y} \rbrace$ | Single MLP that predicts $y$, with $\dot{y}$ and $\ddot{y}$ obtained by `torch.autograd.grad` w.r.t. $t$. Trained on the **same** data as cNN-DP |
| $\mathcal{N}_{AGw}$ | `n_ag.py` + `loss_fn="wmse_errorbased"` | $\mathcal{I} \mapsto \lbrace y, \dot{y}, \ddot{y} \rbrace$ | $\mathcal{N}_{AG}$ with error-based loss weighting (ablation, Sec. 3.2.2) |
| $\mathcal{N}_{DP}$ | [`n_dp.py`](architectures/n_dp.py) | $\mathcal{I} \mapsto y \rightarrow \dot{y} \rightarrow \ddot{y}$ | **cNN-DP (proposed)**, three subnetworks that predict $y$, $\dot{y}$, and $\ddot{y}$ sequentially |

All models are sized to have roughly the same number of trainable parameters.

## Results

The table below lists $R^2$ scores of test predictions ($\mathcal{N}_C$ predicts only $\ddot{y}$).

| Problem | Order | $\mathcal{N}_C$ | $\mathcal{N}_{AG}$ | $\mathcal{N}_{DP}$ |
|---|---|---:|---:|---:|
| Rabinovich–Fabrikant ($\alpha=0.9764$) | $U$ / $\dot U$ / $\ddot U$ | – / – / 0.570 | −718.1 / 0.240 / 0.822 | **0.955 / 0.996 / 0.994** |
| Lorenz | $U$ / $\dot U$ / $\ddot U$ | – / – / 0.075 | −394.7 / 0.434 / 0.629 | **0.956 / 0.996 / 0.998** |
| Earthquake (Antelope Valley 2021) | $y$ / $\dot y$ / $\ddot y$ | – / – / 0.144 | −2.3e5 / −14.7 / 0.702 | **0.978 / 0.998 / 0.999** |

On the 204-DOF multibody trailer model, cNN-DP reaches a mean STFT-MSE of **2.55e-2**, compared with 8.89e-2 for $`\mathcal{N}_{C}`$ and 3.73e-2 for $`\mathcal{N}_{AG}`$.

**Computational cost.** The ratios below are relative to $\mathcal{N}_C$, normalized per parameter (RTX 3060 Ti, batch size 256).

| | $\mathcal{N}_C$ | $\mathcal{N}_{AG}$ | $\mathcal{N}_{DP}$ |
|---|---:|---:|---:|
| Memory for updates | 1 | 15.59 | **1.40** |
| Training time | 1 | 7.99 | **1.85** |
| Inference time | 1 | 14.06 | **1.63** |

## Repository structure

```
architectures/
  skeleton.py        # shared base class (model info, init args)
  n_c.py             # conventional MLP
  n_ag.py            # auto-gradient network
  n_dp.py            # cNN-DP
  interface.py       # NetInterface, loads a saved model and predicts (handles scaling)
utils/
  trainer.py         # Trainer, handles dataloaders, scaling, multi-optimizer training, checkpointing
  loss.py            # MSE / weighted MSE
  initializer.py     # He init
  integrate.py, snippets.py
examples/
  rabinovich-fabrikant/   # Sec. 3.1.1, 3.2
  lorenz/                 # Sec. 3.1.2
  vanderpol/              # Sec. 3.3 (smooth dynamics)
```

## Installation

The code was developed on Linux (Ubuntu 22.04) with PyTorch 2.0 and CUDA 11.8, and is tested with Python 3.10.

```bash
git clone https://github.com/leehyeonbeen/cNN-DP.git
```

```bash
pip install -r requirements.txt
```

## Usage

### Reproducing our examples

Each example has three scripts. **Run them from the repository root**, because paths such as `data/` and `models/` are relative to it.

```bash
python examples/lorenz/datagen.py
```

```bash
python examples/lorenz/train.py
```

```bash
python examples/lorenz/plot.py
```

| Script | What it does | Output |
|---|---|---|
| `datagen.py` | Solves the ODE (implicit RK, `scipy.integrate.solve_ivp`) and computes analytic derivatives | `data/*.csv` |
| `train.py` | Trains $`\mathcal{N}_C`$ / $`\mathcal{N}_{AG}`$ / $`\mathcal{N}_{DP}`$ | `models/<example>/*.pt` |
| `plot.py` | Draws loss curves, trajectories, and error plots | `figures/` |

`plot.py` compares all three models, so first train $`\mathcal{N}_C`$, $`\mathcal{N}_{AG}`$, and $`\mathcal{N}_{DP}`$ by uncommenting `train_n_c()`, `train_n_ag()`, and `train_n_dp()` under `__main__` in `train.py`.

The earthquake, double-pendulum, and trailer examples (Sec. 4) are not included in this repository.

### Training on your own data

Prepare a `pandas.DataFrame` with columns for the inputs and for each derivative order of the outputs, then train as below.

```python
from architectures.n_dp import Net_DP
from utils.trainer import Trainer

net = Net_DP(input_dim=2, width=300, depth=8, output_dim=3)
trainer = Trainer(net)
trainer.setup_dataloader(
    512, df_train, df_valid,
    input_cols=["t", "alpha"],
    y_cols=["x", "y", "z"],
    yDot_cols=["xDot", "yDot", "zDot"],
    yDDot_cols=["xDDot", "yDDot", "zDDot"],
)
trainer.fit(epochs=500, initial_lr=5e-4, lr_halflife=125,
            multioptim=True, save_name="my_problem/n_dp")   # -> models/my_problem/n_dp.pt
```

Inputs and outputs are standardized inside `Trainer`, and the scaling parameters are saved with the checkpoint.

### Inference

```python
from architectures.interface import NetInterface

model = NetInterface("models/my_problem/n_dp.pt")
y, yDot, yDDot = model.predict(x)   # x is an unscaled np.ndarray or torch.Tensor of shape (N, input_dim)
```

`predict` handles batching, device transfer, and scaling/unscaling, and returns unscaled `np.ndarray`s.

## Citation

```bibtex
@article{lee2024cnn,
  title     = {cNN-DP: Composite neural network with differential propagation for impulsive nonlinear dynamics},
  author    = {Lee, Hyeonbeen and Han, Seongji and Choi, Hee-Sun and Kim, Jin-Gyun},
  journal   = {Journal of Computational Physics},
  volume    = {496},
  pages     = {112578},
  year      = {2024},
  publisher = {Elsevier},
  doi       = {10.1016/j.jcp.2023.112578}
}
```

## License

[MIT](LICENSE)
