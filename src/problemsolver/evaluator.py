from inspect import signature
from typing import Annotated, Callable, get_origin, get_args
from problemsolver.utils import Interval
from problemsolver.function_generators import fun_nonlinear as fun_generator
from problemsolver.optimizers.bioinspired.pso import minimize
import optuna
import time
import numpy as np

from optimizers.bioinspired.pso import minimize

def generate_test_functions(n_samples, n_dims, function_names = None) -> list[tuple[Callable, np.ndarray]]:
    # Generate a list of [function, optimum] pairs
    function_names = function_names or fun_generator.FUNCTIONS_AND_OPTIMA.keys()
    output_functions_and_optima = []
    for func_name in function_names:
        for i in range(n_samples):
            # Generate a function and its optimum
            func, optimum_x = fun_generator.get_function_and_optimum(func_name, n_dims=n_dims)
            output_functions_and_optima.append((func, optimum_x))
    return output_functions_and_optima

N_DIMS_TUNE = 2
N_DIMS_TEST = 2
TUNE_FUNCTIONS = generate_test_functions(n_samples=2, n_dims=N_DIMS_TUNE)
TEST_FUNCTIONS = generate_test_functions(n_samples=2, n_dims=N_DIMS_TEST)

def model_runner(**kwargs):
    """
    Kwargs are Optuna trial.suggest_* parameters.
    Full loss should be computed within this. Return the loss value for Optuna to minimize.
    return: float
    """

    log_rel_errors = []
    time_start = time.time()

    for test_func, optimum in TUNE_FUNCTIONS:
        assert np.abs(test_func(optimum)) > 1e-3, "Optimal value should not be near-zero"
        x_hat = minimize(fun=test_func, initial_guess=np.zeros(N_DIMS_TUNE), **kwargs)
        log_rel_errors.append(np.log10(np.abs(test_func(x_hat) - test_func(optimum)) / np.abs(test_func(optimum))))

    time_elapsed = time.time() - time_start
    print(f"Trial with params {kwargs} took {time_elapsed:.2f}s, mean log rel errors: {np.mean(log_rel_errors):.3f}")

    total_loss = np.mean(log_rel_errors) + time_elapsed
    return total_loss


def make_optuna_objective(minimizer_to_test: Callable) -> Callable:
    sig = signature(minimizer_to_test)

    # The term "trial" is magic used by Optuna
    def optuna_loss(trial):
        kwargs = {}
        for name, param in sig.parameters.items():
            if name in ['fun', 'initial_guess']:
                continue
            anno = param.annotation
            if get_origin(anno) is Annotated:
                base_type, meta = get_args(anno)
                if isinstance(meta, Interval):
                    if base_type is int:
                        step = meta.step if meta.step is not None else 1
                        kwargs[name] = trial.suggest_int(name, meta.low, meta.high,
                                                         step=step, log=meta.log)
                    else:
                        step = meta.step if meta.step is not None else (meta.high - meta.low) / 100
                        kwargs[name] = trial.suggest_float(name, meta.low, meta.high,
                                                           step=step, log=meta.log)
                elif isinstance(meta, list) and base_type is str:
                    kwargs[name] = trial.suggest_categorical(name, meta)
                else:
                    raise ValueError(f"Unsupported metadata for {name}: {meta}")
            else:
                kwargs[name] = param.default

        return model_runner(**kwargs)

    return optuna_loss

# Create and run the study
study = optuna.create_study(direction="minimize")
study.optimize(make_optuna_objective(minimize), n_trials=50)
print("Best params:", study.best_params)