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

# MVP

For simplicity, let's consider just a teacher-student problem where the teacher has randomly initialized weights,
and the student has the same architecture as the teacher. Changing the model size/architecture allows us to make this
arbitrarily complex.

What counts as "solved"? 
- **algorithm hits a loss threshold**: Good if we can know the loss form ahead of time or can create a lower bound on the loss. **If problems are drawn from different scales, this will require normalizing**.
- Loss plateau: Could use stopping criteria based on the loss becoming flat, e.g. through a minimum value on a learning rate scheduler

What is a good experimentation format?
- Create the teacher problem and sample data: on-the-fly data creation can be expensive
- Define the batch size, learning rate scheduler, and stopping criteria
- Run the problem until it stops
- Report the loss and total compute time.  Ideally, also report the memory, and measure GPU parallelism in the proposed model.

evaluator.py:benchmark_optimizer ->  
- make_optuna_objective(optimizer, func_optima_tuple, ...)
  - univariate_model_runner(**kwargs)
    - log_rel_error, mean_time_elapsed = multivariate_model_runner(**kwargs)
    - 
- multivariate_model_runner(optimizer, test_functions, ...)
  - if n_jobs==1: single_thread_multivariate_model_runner(minimizer, func_optima_tuples, **kwargs)
  - else: _evaluate_single_function(args=[test_func_with_minimizer, ])
    - test_func(optimum)
    - x_hat = minimizer(test_func)
    - numerator = test_func(x_hat)

we're going from 
```
optimum = wrapped_func.optimum_x
test_func = wrapped_func.func_z
minimizer = wrapped_func.minimizer

x_hat = minimizer(test_func)
y_hat = test_func(x_hat)
loss = y_hat - test_func(optimum)
```

to something like:
```
x_train = wrapped_func.x_train
y_train = wrapped_func.y_train
optimizer = wrapped_func.optimizer
x_test = wrapped_func.x_test
y_test = wrapped_func.y_test

model = fit(optimizer, x_train, y_train)
y_hat = model(x_test)
loss = y_hat - y_test
```

Approach: Move all of this into a new method of wrapped_func, something like fit_and_report_loss()


Fundamentally, the shift from functions which can easily be evaluated like func(x) to something that is evaluated like 


For simplicity, let's