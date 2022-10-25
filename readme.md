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

## LRT Optimization

Let $W \in \mathbb{R}^{n \times m}$ be a model parameter. In the LRT framework, the parameter $W$ has an associated model $\Lambda_W$ with parameter matrix $W^\Lambda \in \mathbb{R}^{j \times k}$ that outputs a learning rate tensor $\lambda_W \in \mathbb{R}^{n \times m}$ with a shape identical to $W$.

![alt text](https://raw.githubusercontent.com/henrysteinitz/lr_tensors/main/architecture.png)

At each timestep, the loss $l(M(x_t))$ is computed as a function of the forward model output. Note that this step does not include any computation from the LRT model $\Lambda_W$, so that the LRT model does not alter the original model's expressive capacity. First, the gradient is scaled pointwise by $\lambda_W$:

$$W_{t+ 1} = W_{t} - \lambda_W \odot \nabla_{W_t} l(M_t(x_t)).$$

Then, the loss $l(M_{t+1}(x_t))$ is recomputed with $W_{t + 1}$ in place of $W_{t}$. Note that this computation does include the parameters of the LRT model $\Lambda_W$. Finally, the gradient of the recomputed loss is taken with respect to LRT model parameter tensor $W^\Lambda$, and one step of standard stochastic gradient descent is taken with a constant metalearning rate of $\lambda'$:

$$W^\Lambda_{t+1} = W^\Lambda_{t} - \lambda' \nabla_{W^\Lambda_{t}}l(M_{t+1}(x_t))$$
In contrast to typical metalearning models, both the LRT model $\Lambda_W$ and forward model $M$ are optimized simultaneously.