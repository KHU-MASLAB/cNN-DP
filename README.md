# cNN-DP: Composite neural network with differential propagation
![github](https://github.com/KHU-MASLAB/cNN-DP/assets/78078652/b37e129f-4cef-4250-b958-12ada1e5e688)
Lee, Hyeonbeen and Han, Seongji and Choi, Hee-Sun and Kim, Jin-Gyun, *cNN-DP: Composite Neural Network with Differential Propagation for Impulsive Nonlinear Dynamics.* Available at SSRN: https://ssrn.com/abstract=4296911
## What is this for?
> cNN-DP: Composite neural network with differential propagation

We propose a novel composite neural network for effective learning of highly impulsive/oscillatory time series data. We assume our target as **high-order derivatives** and utilize **additional low-order derivatives**, which were *wasted* or *inactively treated* before.  
It is effective, specifically for:
* Modeling oscillatory solutions (derivatives) of differential equations
* Learning noisy and impulsive time series measurements

Examples we present in our paper include:

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
* Expandable to any domains, such as functions of space, frequency, or any arbitrary variables


## How does it work?
<img width="1108" alt="Screenshot 2023-06-11 at 20 33 29" src="https://github.com/KHU-MASLAB/cNN-DP/assets/78078652/e640be65-35b1-4f9a-8095-7b755f0eaaf7">
Dynamics tend to become more 'impulsive' in high-order derivatives. In reverse, it becomes 'simpler'. Then, why don't we let the neural network also refer to 'simple' when it's learning 'impulsive', rather than solely learning the 'impulsive'? 

We intend our neural network to learn the 'simple' and the 'impulsive' simultaneously by interconnecting multiple MLP subnetworks with corresponding losses. The preceding subnets predict the 'simple's, and their outputs are connected to inputs of subsequent subnets. This enables richer context information in the learning of impulsive or chaotic systems as functions of time and design variables.

Using ```torch``` pseudocode, the process can be expressed as:
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
Although we only use three subnets in our paper, the number of subnets in the cNN-DP is **not strictly limited**. Theoretically 2 to ```inf```.

## What is the auto-gradient network?
It is another competitor of the proposed cNN-DP. Suppose we have data of multiple orders of derivatives and we target the highest derivative, just like the cNN-DP. The idea of the auto-gradient network is to utilize automatic differentiation (```torch.autograd.grad```) to compute high-order predictions of neural networks.

It will first output the lowest-order prediction. Then, we repeatedly differentiate the network to the time variable (which should be given in the input) to reach the highest-order prediction.

This approach can be a considerable substitute. However, it turns out to be extremely expensive and slow both for training and inference.

## How do I use the codes?
In the ```examples``` directory, we have **three examples presented in the paper** each including **data generation, training, and visualizing** codes.

```train.py``` will train network models and save it to ```models``` directory. Inference of saved models can be easily obtained through ```architectures.interface.NetInterface``` class as following pseudocode:
```
n_dp = NetInterface(models/SAVED_MODEL.pt)
y,yDot,yDDot=n_dp.predict(input)
```

## Citing the paper (BibTeX)
```
@article{lee4296911cnn,
  title={cNN-DP: Composite Neural Network with Differential Propagation for Impulsive Nonlinear Dynamics},
  author={Lee, Hyeonbeen and Han, Seongji and Choi, Hee-Sun and Kim, Jin-Gyun},
  journal={Available at SSRN 4296911}
}
```

