# Distributed Learning Rate Tensors

We describe an alternative to training neural networks with stochastic 
gradient descent in which distributed learning rate signals are pointwise
multiplied by gradient signals to compute update steps. Learning rate tensors
are generated for each training example by learning rate tensors models that
simultaneously learn to minimize training loss by standard SGD. In particular,
we show that for a few synthetic mathematical datasets, a multilayer perceptron 
with linear learning rate tensor models converges to the minimum stable
test loss more quickly and more reliably than standard SGD.

Implementations of this optimization scheme for various model architectures can be found in models.py.

## Motivation

Most state-of-the-art machine learning systems are almost-everywhere differentiable functions with parameters trained by gradient optimization techniques such as stochastic gradient descent 
(SGD) or SGD with momentum. These gradient methods typically have 1 or several constant scalar hyperparameters such as a learning rate λ that dictate their optimization dynamics. The internal state of these optimizers are typically characterized by at most a handful of scalar variables.

While unbiased stochastic error gradients are theoretically motivated and efficiently
computable [1], they often diverge from the optimal update direction. For example,
in deep fully-connected networks, gradient signals tend to vanish to 0 or diverge to
infinity in early layers.[4] As a result, it is widely agreed that clipping gradient signals at
a maximum value increases the liklihood of SGD convergence. This suggests that, more generally, tailoring our optimizatation algorithm to each model parameter may improve learning.

This work is closely related to recent developments in meta-learning. Bengio et
al. (1992) proposes meta-learning of local update functions without references to
exact architecture or implementation.

Models that incorporate meta-learning typically divide optimization into an
inner training loop that optimizes model parameters and an outer training loop
that optimizes hyperparameters.[2][3] This work differs from standard meta-learning paradigms since optimization of model parameters and LRT models happen simultaneously. There is also a
constant learning rate hyperparameter learning rate used to optimize the LRT model parameters.

## LRT Optimization

Let $W \in \mathbb{R}^{n \times m}$ be a model parameter. In the LRT framework, the parameter $W$ has an associated model $\Lambda_W$ with parameter matrix $W^\Lambda \in \mathbb{R}^{j \times k}$ that outputs a learning rate tensor $\lambda_W \in \mathbb{R}^{n \times m}$ with a shape identical to $W$.

![alt text](https://raw.githubusercontent.com/henrysteinitz/lr_tensors/main/architecture.png)

At each timestep, the loss $l(M(x_t))$ is computed as a function of the forward model output. Note that this step does not include any computation from the LRT model $\Lambda_W$, so that the LRT model does not alter the original model's expressive capacity. First, the gradient is scaled pointwise by $\lambda_W$:

$$W_{t+ 1} = W_{t} - \lambda_W \odot \nabla_{W_t} l(M_t(x_t)).$$

Then, the loss $l(M_{t+1}(x_t))$ is recomputed with $W_{t + 1}$ in place of $W_{t}$. Note that this computation does include the parameters of the LRT model $\Lambda_W$. Finally, the gradient of the recomputed loss is taken with respect to LRT model parameter tensor $W^\Lambda$, and one step of standard stochastic gradient descent is taken with a constant metalearning rate of $\lambda'$:

$$W^\Lambda_{t+1} = W^\Lambda_{t} - \lambda' \nabla_{W^\Lambda_{t}}l(M_{t+1}(x_t))$$
In contrast to typical metalearning models, both the LRT model $\Lambda_W$ and forward model $M$ are optimized simultaneously.

## LRT Model Architecture

If we want LRT optimization to adapt to the current set of training examples, our LRT models must accept network state as input. We propose two styles of LRTM architectures for deep sequential networks: input-driven models and state-driven models. The models differ in their relationship to activations in sequential models, but are agnostic to their internal architecture. This paper considers only affine LRT models.

### Input-drive LRTMs

![alt text](https://raw.githubusercontent.com/henrysteinitz/lr_tensors/main/input_driven_architecture.png)

Let $W$ be a parameter in model $M$. Input driven learning rate tensor models $\lambda_W(x)$ are functions of the complete model input $x$


### State-driven LRTMs

![alt text](https://raw.githubusercontent.com/henrysteinitz/lr_tensors/main/state_driven_architecture.png)

Let $W$ be a parameter in a submodule $M'$ of $M$. State-driven learning rate tensor models $\lambda_W(h)$ are functions of the submodule input $h$. In sequential models, this is the networks hidden-state or activation coming from the previous layer. 

## References 

[1] Ozgur Ceyhan. Algorithmic complexities in backpropagation and tropical neural networks.
CoRR, abs/2101.00717, 2021.

[2] Julio Hurtado, Alain Raymond-Saez, and Alvaro Soto. Optimizing reusable knowledge for
continual learning via metalearning. CoRR, abs/2106.05390, 2021.

[3] Luke Metz, Niru Maheswaranathan, Brian Cheung, and Jascha Sohl-Dickstein. Learning un-
supervised learning rules. CoRR, abs/1804.00222, 2018.

[4] Razvan Pascanu, Tom ́as Mikolov, and Yoshua Bengio. Understanding the exploding gradient
problem. CoRR, abs/1211.5063, 2012.
