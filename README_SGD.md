# README - Moving from continuous functions to improving stochastic gradient descent

View a data-generating function as $ y ~ f(\theta^*)$ or equivalently $ y = f(\theta^*, x)$ where $x$ is the (unknowable)
complete data distribution and y are the accompanying targets/labels.  

In practice, we see some set of observations $\hat{x}$ and may take mini-batches $\hat{x}_i$, which we will equivalently
denote as simply $x_i$ for convenience. 

We do not know the true functional form of $f$ but can approximate it with a generalized function $g(\hat{\theta})$ where 
for the ideal parameter estimate and an adequately parameterized approximating function we can say
$g(\hat{\theta}^*) ~ f(\theta^*)$

The goal of an optimizer is thus to find optimal values of $\hat{\theta}$ to minimize the loss between y=f(x) and 
$\hat{y} = g(\hat{\theta}, \hat{x})$ based on observed values of $\hat{x}$ and some optimization algorithm.

In practice, this could mean that for our data generating functions {rastrigin, griewank, ...} we seek to find some 
optimal parameters which allow the estimated function to map to the original function

Our test functions are of the form:
$$
\begin{align*}
z &= A^T(x - b) \\
y &= f(z) \\
f &\in \text{rastrigin}(z), \text{griewank}(z), \text{rosenbroeck}(z), \ldots 
\end{align*}
$$

So the problem then becomes:

$$
\begin{align*}
y &= f(A^{*T}(x - b^*)) \\
\hat{y} &= g(\hat{\theta}, x) \\
\hat{\theta} &= \{ \hat{A}, \hat{b} \} \\
\hat{y} &= f(\hat{A}^T(x - \hat{b})) \\
l &= \text{criterion}(y, \hat{y}), \qquad \text{e.g.} \\
l &= \text{MSE}(\hat{y}, y)\\
\end{align*}
$$

The loss thus should be significantly nonconvex in $\hat{\theta} = \{ \hat{A}, \hat{b} \}$, which makes this approach
promising as the parameter space will scale as $n^2 + n$ for problem dimension n. 

## Algorithmic improvement space

This approach should allow us full flexibility of implementing minibatch-SGD style optimization. This means that we can 
consider improvements in any aspect of the optimizer loop (batching, learning rate, parameter update)

### Evaluation

Ultimately, we care about the speed with which we can reach acceptable loss on a real machine learning problem (e.g. 
NanoGPT speedrun). This will be dependent on many factors, but will include:
- Number of observations (or epochs) to meet loss target
- GPU kernel parallelism
- Adaptability to a variety of minibatch sizes (GPU memory limitation)
- GPU rack parallelism
