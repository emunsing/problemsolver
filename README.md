# Problemsolver AI tool

Goal: Develop novel optimization methods by using an LLM to smash up ideas from existing algorithms.

API / signatures:

Unconstrained optimization:
- minimize(fun: Callable[[np.ndarray], float], initial_guess: np.ndarray, **kwargs) -> np.ndarray
  - Note: kwargs should all be float-valued, 
- function(x: np.ndarray) -> float

Instructions:

Rules:
- The goal is to advance the pareto frontier of computation time and log relative error in the optimal solution.  As a result, novel optimization algorithms should equally strive for accuracy and speed.
- All algorithms must be able to handle arbitrary dimensionality, and should be able to handle at least 10 dimensions.
- We will evaluate the performance of the optimization algorithms on a set of randomly generated benchmark problems. Because the problems are randomly generated, the algorithms will not be able to overfit to the training set.
- We will run hyperparameter optimization on the `minimize` function over a set of randomly generated benchmark problems to find the best set of hyperparameters.- Those hyperparameters will then be used when evaluating performance of the optimization algorithms on the holdout set of benchmark problems.  Again, these will be randomly generated to prevent overiftting.
- Hyperparameter selection and range for optimization will be identified by any function parameters which have a typing Annotated Interval.
- A fixed hyperparameter optimization iteration budget will be used to find the best hyperparameters for each algorithm. As a result, the most succcessful algorithms will likely have good guesses for hyperparameter ranges defined in the Interval, or a small number of hyperparameters. Models with a large number of hyperparameters or large ranges will likely underperform their peers.
- Any hyperparameters which do not have a type Annotated with an Interval will not be considered for hyperparameter optimization, and will use their default value.
- Any static hyperparameters (e.g. random seed, etc.) should either have a default value without Annotated Interval, or should be set as variables internal to the `minimize` function and not used as a hyperparameter.


Model design:
- Models must be made with pure-python and numpy. Scipy or other advanced problem-solving packages are not allowed.

Function signatures:
- All functions should be named `minimize`
- All functions should take a callable `fun` that takes a numpy array and returns a float.
- All functions should take an `initial_guess` numpy array. The value of this is not important, but it is used to identify the appropriate number of dimensions.
- Function kwargs should either be hyperparameters which need to be tuned, or default values which will not be changed:
  - Hyperparameters will be noted with a type Annotated which includes an `Interval`, e.g.  `learning_rate: Annotated[float, Interval(1e-5, 1e-1, log=True)]` or `n_layers: Annotated[int, Interval(1, 5)]`
  - Non-tunable function parameters should not have an Annotated type, e.g. `max_iterations: int = 1000`



Optimizer builder:
- Have both a train set of problems, and a holdout set of problems
- Give the "Experimenter" process a budget of *n* experiments for each functional idea.  This should help prevent overfitting.
- Interested in multiple metrics:
  - Total number of function evaluations
  - Number of iterations
  - Accuracy of solution at stopping

- For each experiment,
  - Gather a set of existing optimization algorithms
  - Draw a random card of inspiration from a deck of emergent systems
  - LLM: Come up with a new algorithm by combining the existing algorithms and the inspiration card
  - Iteratively improve the new algorithm on a set of benchmark problems:
    - For i in range(n_iterations):
      - Run the algorithm on the train set of benchmark problems
      - LLM: Debug the algorithm if it fails to produce valid output 
      - Evaluate the performance of the new algorithm against existing algorithms
      - LLM: Improve the code of the algorithm based on the results of the test
  - If the new optimization algorithm is better than the parent algorithms on the holdout set, then add it to the set of existing algorithms.


Steps:
1. Unconstrained
1. Constrained with min/max values on each variable
1. Constrained with affine constraints
1. Constrained with mixed integer variables
1. Constrained with complementarity constraints <- *this can represent combinatorial problems*

# problemsolver

See [list of metaheuristic algorithms](https://en.wikipedia.org/wiki/Table_of_metaheuristics)

Solvers:
- [x] L-BFGS
- [x] stochastic gradient descent
- [x] Adam
- [x] adamw
- [x] Simulated Annealing
- [x] Differential Evolution
- [x] Ant Colony Optimization 
- [x] Bees algorithm
- [x] Artificial Bee Colony
- [x] Particle Swarm Optimization
- [x] Genetic Algorithm
- Whale Optimization Algorithm
- Firefly Algorithm

Combinatoric search algorithms:
- Cuckoo Search
- Tabu search

[Test functions of n dimensions](https://en.wikipedia.org/wiki/Test_functions_for_optimization):
- Sphere
- Rosenbrock
- Rastrigin
- Griewank
- Styblinski–Tang function
- Keane's bump function

Main function:

- 