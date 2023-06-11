# cNN-DP: Composite neural network with differential propagation
Lee, Hyeonbeen and Han, Seongji and Choi, Hee-Sun and Kim, Jin-Gyun, *cNN-DP: Composite Neural Network with Differential Propagation for Impulsive Nonlinear Dynamics.* Available at SSRN: https://ssrn.com/abstract=4296911
![github](https://github.com/KHU-MASLAB/cNN-DP/assets/78078652/b37e129f-4cef-4250-b958-12ada1e5e688)
## What is this for?
> cNN-DP: Composite neural network with differential propagation

We propose a composite neural network for effective and efficient learning of highly impulsive / oscillatory time series data by utilizing low-order derivatives, which were rather wasted or unconsidered before.  
It is effective, specifically for:
* Modeling oscillatory solutions (derivatives) of differential equations
* Learning noisy and impulsive time series measurements

Examples we present in our paper are:

* Earthquake measurements
* Rigid body contact
* Vehicle simulations
* Solution derivatives of chaotic systems

## What's the advantage of using this?
* Highly precise generalization on oscillatory, impulsive, and chaotic systems
* Fast training and inference time
* Low VRAM usage
* Robust to data quality and hyperparameters
* Does not require any physical equations to use
* Expandable to any domains, such as functions of space, frequency or any arbitrary variables


## How does it work?
<img width="1108" alt="Screenshot 2023-06-11 at 20 33 29" src="https://github.com/KHU-MASLAB/cNN-DP/assets/78078652/e640be65-35b1-4f9a-8095-7b755f0eaaf7">
Dynamics tends to become more 'impulsive' in high-order derivatives. In reverse, it becomes 'simpler'. Then, why don't we let the neural network also refer to 'simple' when it's learning 'impulsive', rather than solely learning the 'impulsive'? 

We intend our neural network to learn the 'simple' and the 'impulsive' simultaneously by interconnecting multiple subnetworks with corresponding losses. The preceding subnets will predict the 'simple's, and their outputs are connected to inputs of subsequent subnets. This enables richer context information in learning of impulsive time series data.

Using ```torch``` psuedocode, the process can be expressed as:
```
n_dp0=MLP1()
n_dp1=MLP2()
n_dp2=MLP3()
def forward(x):
  y=n_dp0(x)
  yDot=n_dp1(x,y)
  yDDot=n_dp2(x,y,yDDot)
  return y,yDot,yDDot
```
Although we always use three subnets in our paper, the number of subnets in the cNN-DP is not strictly limited. Theroetically 2 to ```inf```.

## What is auto-gradient network?
Suppose we have data of multiple orders of derivatives and we target the highest derivative, just like the cNN-DP. The idea of auto-gradient network is to utilize automatic differentiation (```torch.autograd.grad```) to compute high-order predictions of neural network.

It will first output the lowest order prediction. Then, we repeatedly differentiate the network to time variable (which should be given in the input) to reach the highest-order prediction.

This approach can be a considerable substitute. However, it turns out to be extremely expensive and slow both for training and inference.

## How do I use the codes?
In the ```examples``` directory, we have three examples presented in the paper each including data generation, training, and visualizing codes.

```train.py``` will train models and save it to ```models``` directory. Inference of saved models can be easily obtained through ```architectures.interface.NetInterface``` class as follows:
```
n_dp = NetInterface(models/SAVED_MODEL.pt)
y,yDot,yDDot=n_dp.predict(input)
```
If you want to lookup the ```nn.Module``` classes of three models, please refer to ```architectures``` directory.

## Abstract
In mechanical engineering, abundant high-quality data from simulations and experimental observations can help develop practical and accurate data-driven models. However, when dynamics are complex and highly nonlinear, designing a suitable model and optimizing it accurately is challenging. In particular, when data comprise impulsive signals or high-frequency components, training a data-driven model becomes increasingly challenging. This study proposes a novel and robust composite neural network for impulsive time-transient dynamics by dividing the prediction of the dynamics into tasks for three sub-networks, one for approximating simplified dynamics and the other two for mapping lower-order derivatives to higher-order derivatives. The mapping serves as the temporal differential operator, hence, the name “composite neural network with differential propagation (cNN-DP)” for the suggested model. Furthermore, numerical investigations were conducted to compare cNN-DP with two baseline models, a conventional network and another employing the autogradient approach. Regarding the convergence rate of model optimizations and the generalization accuracy, the proposed network outperformed the baseline models by many orders of magnitude. In terms of computational efficiency, numerical tests showed that cNN-DP requires an acceptable and comparable computational load. Although the numerical studies and descriptions focus on accelerations, the proposed network can be easily extended to any other application involving impulsive data.

