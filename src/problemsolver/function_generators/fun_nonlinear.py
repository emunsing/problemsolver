import numpy as np
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def generate_affine_transformation(n_dims: int):
    rng = np.random.default_rng()
    scale_range = (0.5, 2.0)
    shift_range = (-5.0, 5.0)
    shift = rng.uniform(*shift_range, size=n_dims)
    Q, _ = np.linalg.qr(rng.normal(size=(n_dims, n_dims)))
    scales = rng.uniform(*scale_range, size=n_dims)
    A_mat = Q @ np.diag(scales)
    return A_mat, shift


def generate_transformed_function(func_z: Callable[[np.ndarray], float], optimum_z: np.ndarray):
    n_dims = len(optimum_z)
    A_mat, shift = generate_affine_transformation(n_dims)
    optimum_x = np.linalg.solve(A_mat, optimum_z) + shift

    def transformed_func(x: list[float]) -> float:
        x = np.asarray(x)
        z = A_mat @ (x - shift)
        return func_z(z)

    return transformed_func, optimum_x


def rastrigin(x):
    A = 10.0
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def sphere(x):
    return np.sum(x ** 2)


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def griewank(z: np.ndarray) -> float:
    n_dims = len(z)
    sum_sq = np.sum(z**2)
    prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, n_dims + 1))))
    return 1 + sum_sq / 4000 - prod_cos


def styblinski_tang(z: np.ndarray) -> float:
    return 0.5 * np.sum(z**4 - 16 * z**2 + 5 * z)


def keane_bump(z: np.ndarray) -> float:
    # The keane bump function is symmetric; this modifies it to be asymmetric with a preference for the negative dimension
    n_dims = len(z)
    sum_cos4 = np.sum(np.cos(z)**4)
    prod_cos2 = np.prod(np.cos(z)**2)
    sum_i_x2 = np.sum((np.arange(1, n_dims + 1)) * z**2)
    numerator = sum_cos4 - 2 * prod_cos2
    denominator = np.sqrt(sum_i_x2)
    bump_val = -abs(numerator / denominator)

    # Symmetry-breaking bias: slight slanted paraboloid
    weights = 1.0 / (np.arange(1, n_dims + 1))  # Decreasing weights for stability
    bias = 1e-3 * np.sum(weights * (0.1 * z + z**2))
    return bump_val + bias



FUNCTIONS_AND_OPTIMA = {
    # "rastrigin": (rastrigin, lambda n_dims: np.zeros(n_dims)),
    # "sphere": (sphere, lambda n_dims: np.zeros(n_dims)),
    # "rosenbrock": (rosenbrock, lambda n_dims: np.ones(n_dims)),
    # "griewank": (griewank, lambda n_dims: np.zeros(n_dims)),
    # "styblinski_tang": (styblinski_tang, lambda n_dims: np.ones(n_dims) * -2.903534),
    "keane_bump": (keane_bump, lambda n_dims: np.array([-np.sqrt(2)] + [0.0] * (n_dims - 1))),
}


def get_function_and_optimum(func_name: str, n_dims: int):
    func_z, optimum_gen = FUNCTIONS_AND_OPTIMA[func_name]
    optimum_z = optimum_gen(n_dims)
    func_x, optimum_x = generate_transformed_function(func_z, optimum_z)
    return func_x, optimum_x


def visualize_function(func_x: Callable, optimum: np.ndarray = None, title: str = "Function Visualization"):
    plot_range = (-5.0, 5.0)
    x1_range = np.linspace(plot_range[0], plot_range[1], 100)
    x2_range = np.linspace(plot_range[0], plot_range[1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Y = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Y[i, j] = func_x(np.array([X1[i, j], X2[i, j]]))

    plt.figure(figsize=(10, 10))
    plt.contour(X1, X2, Y, levels=100)

    if optimum is not None:
        plt.scatter(optimum[0], optimum[1], color="red")

    plt.colorbar()
    plt.title(title)
    plt.show()
    


if __name__ == "__main__":
    for func_name in FUNCTIONS_AND_OPTIMA.keys():
        func_z, opt_gen = FUNCTIONS_AND_OPTIMA[func_name]
        optimum = opt_gen(2)  # Assuming 2D for visualization

        print(f"Visualizing {func_name}")
        print(f"Optimum value : {func_z(optimum): .3f} at {optimum}")
        visualize_function(func_z, optimum=optimum, title=f"{func_name} Function")



