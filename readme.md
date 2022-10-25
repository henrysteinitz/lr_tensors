# Distributed Learning Rate Tensors

We describe an alternative to training neural networks with sto-
chastic gradient descent in which distributed learning rate signals are pointwise
multiplied by gradient signals to compute update steps. Learning rate tensors
are generated for each training example by learning rate tensors models that
simultaneously learn to minimize training loss by standard SGD. In particular,
we show that for a few synthetic mathematical datasets, a multilayer percep-
tron with linear learning rate tensor models converges to the minimum stable
test loss more quickly and more reliably than standard SGD.