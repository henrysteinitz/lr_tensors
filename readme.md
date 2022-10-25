# Distributed Learning Rate Tensors

We describe an alternative to training neural networks with stochastic 
gradient descent in which distributed learning rate signals are pointwise
multiplied by gradient signals to compute update steps. Learning rate tensors
are generated for each training example by learning rate tensors models that
simultaneously learn to minimize training loss by standard SGD. In particular,
we show that for a few synthetic mathematical datasets, a multilayer percep-
tron with linear learning rate tensor models converges to the minimum stable
test loss more quickly and more reliably than standard SGD.

## LRT Optimization

Let $W \in \mathbb{R}^{n \times m}$ be a model parameter. In the LRT framework, the parameter $W$ has an associated model $\Lambda_W$ with parameter matrix $W^\Lambda \in \mathbb{R}^{j \times k}$ that outputs a learning rate tensor $\lambda_W \in \mathbb{R}^{n \times m}$ with a shape identical to $W$.

\begin{center}
\begin{figure}
\begin{tikzpicture}[scale=1.3,every node/.style={minimum size=1cm},on grid]
    \draw[white] (-5,1.6) rectangle (-4,2.6);
    \draw[black] (-5,1) rectangle (-4,2);
    \draw[black] (-3,1) rectangle (-2,2);
    \draw[black] (0,1) rectangle (-1,2);
    \draw[black] (2.4,1) rectangle (1.4,2);
    
    \draw[black] (2.8,.2) rectangle (-5.3,-1.8);
    \draw[black] (-5,-1) rectangle (-4,0);
    \draw[black] (-3,-1) rectangle (-2,0);
    \draw[black] (0,-1) rectangle (-1,0);
    \draw[black] (2.4,-1) rectangle (1.4,0);
    \draw[white] (-5,-1.9) rectangle (-6,-1.6);
    \draw[black, thick, ->] (-4.5, 1) -> (-4.5, 0);
    \draw[black, thick, ->] (-2.5, 1) -> (-2.5, 0);
    \draw[black, thick, ->] (-0.5, 1) -> (-0.5, 0);
    \draw[black, thick, ->] (1.9, 1) -> (1.9, 0);
    
    \draw[black, thick, ->] (-5.5, -0.5) -> (-5.0, -0.5);
    \draw[black, thick, ->] (-4, -0.5) -> (-3, -0.5);
    \draw[black, thick, ->] (-2, -0.5) -> (-1, -0.5);
    \draw[black, thick, ->] (0, -0.5) -> (.3, -0.5);
    \draw[black, thick, ->] (1.0, -0.5) -> (1.4, -0.5);
    \draw[black, thick, ->] (2.4, -0.5) -> (3.0, -0.5);
    
    \node at (-4.5, 1.5){$\Lambda_{W_1, b_1}$};
    \node at (-2.5, 1.5){$\Lambda_{W_2, b_2}$};
    \node at (-0.5, 1.5){$\Lambda_{W_3, b_3}$};
    \node at (1.9, 1.5){$\Lambda_{W_N, b_N}$};
    \node at (-4.0, 0.5){$\lambda_{W_1, b_1}$};
    \node at (-2.0, 0.5){$\lambda_{W_2, b_2}$};
    \node at (0.0, 0.5){$\lambda_{W_3, b_3}$};
    \node at (2.4, 0.5){$\lambda_{W_N, b_N}$};
    \node at (-5.7, -0.5){$x$};
    \node at (-4.9, -1.4){$M$};
    \node at (-4.5, -0.5){$W_1, b_1$};
    \node at (-2.5, -0.5){$W_2, b_2$};
    \node at (-0.5, -0.5){$W_3, b_3$};
    \node at (1.9, -0.5){$W_N, b_N$};
    \node at (0.7, -0.5){$\dots$};
    \node at (3.2, -0.5){$y$};
\end{tikzpicture}
\end{figure}
\end{center}