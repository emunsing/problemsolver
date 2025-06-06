# Problemsolver AI tool

Goal: Develop novel optimization methods by using an LLM to smash up ideas from existing algorithms.

API / signatures:
- minimize(function, initial_guess: List[float], constraints: Optional[List[list[float], list[float]]])
- function(x: List[float]) -> float

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
- L-BFGS
- stochastic gradient descent
- Adam
- adamw
- Simulated Annealing
- Differential Evolution
- Ant Colony Optimization 
- Bees algorithm
- Artificial Bee Colony
- Particle Swarm Optimization
- Genetic Algorithm
- Whale Optimization Algorithm
- Firefly Algorithm

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