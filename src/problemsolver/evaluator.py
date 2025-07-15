from inspect import signature
from typing import Annotated, Callable, get_origin, get_args
from problemsolver.utils import Interval
from problemsolver.function_generators import fun_nonlinear as fun_generator
from problemsolver.optimizers.bioinspired.pso import minimize
import optuna
import time
import numpy as np
import click


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


def multivariate_model_runner(minimizer: Callable, func_optima_tuples: list[tuple[Callable, np.ndarray]], **kwargs) -> tuple[float, float]:
    """
    reliant on the `minimize` function imported into this context.
    TODO: Make the `minimize` function a parameter to this function.
    TODO: Make the function generator TUNE_FUNCTIONS a parameter to this function.

    Return a univariate metric for performance of the minimizer.  In this case, we use the log of the relative error,
    plus the mean time taken to run the minimization across a set of test functions.

    Kwargs are Optuna trial.suggest_* parameters.
    Full loss should be computed within this. Return the loss value for Optuna to minimize.
    return: float
    """

    log_rel_errors = []
    time_start = time.time()

    for test_func, optimum in func_optima_tuples:
        assert np.abs(test_func(optimum)) > 1e-3, "Optimal value should not be near-zero"
        x_hat = minimizer(fun=test_func, initial_guess=np.zeros(N_DIMS_TUNE), **kwargs)
        log_rel_errors.append(np.log10(np.abs(test_func(x_hat) - test_func(optimum)) / np.abs(test_func(optimum))))

    time_elapsed = time.time() - time_start
    print(f"Trial with params {kwargs} took {time_elapsed:.2f}s, mean log rel errors: {np.mean(log_rel_errors):.3f}")

    return np.mean(log_rel_errors), time_elapsed


def univariate_model_runner(**kwargs):
    log_rel_error, time_elapsed = multivariate_model_runner(**kwargs)
    total_loss = np.mean(log_rel_error) + time_elapsed
    return total_loss


def make_optuna_objective(minimizer_to_test: Callable,
                          func_optima_tuples: list[tuple[Callable, np.ndarray]]) -> Callable:
    sig = signature(minimizer_to_test)

    # The term "trial" is magic used by Optuna
    def optuna_loss(trial):
        kwargs = {'minimizer': minimizer_to_test, 'func_optima_tuples': func_optima_tuples}
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

        return univariate_model_runner(**kwargs)

    return optuna_loss


def tune_minimizer(minimizer_to_test: Callable, n_trials: int = 50):
    """
    Tune the minimizer using Optuna.

    :param minimizer_to_test: The minimizer function to tune.
    :param n_trials: Number of trials for tuning.
    :return: The best parameters found by Optuna.
    """
    objective = make_optuna_objective(minimizer_to_test, func_optima_tuples=TUNE_FUNCTIONS)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def test_minimizer(minimizer_to_test: Callable, n_tuning_trials: int = 50):
    """
    Test the minimizer with a set of test functions.

    :return: None
    """
    best_params = tune_minimizer(minimizer_to_test=minimizer_to_test, n_trials=n_tuning_trials)
    print("Best parameters found:", best_params)
    log_rel_errors, time_elapsed = multivariate_model_runner(minimizer=minimizer_to_test,
                                                             func_optima_tuples=TEST_FUNCTIONS,
                                                             **best_params)
    print(f"Test results: mean log rel errors = time elapsed = {time_elapsed:.2f}s, mean log rel errors {log_rel_errors:.3f}")



@click.group()
def cli():
    pass



@cli.command()
@click.option('--n-trials', default=50, help='Number of trials for hyperparameter tuning')
def tune(n_trials):
    """Tune hyperparameters for a numeric minimizer."""
    best_params = tune_minimizer(minimizer_to_test=minimize, n_trials=n_trials)
    
    click.echo("Best parameters found:")
    for param, value in best_params.items():
        click.echo(f"  {param}: {value}")



@cli.command()
def test():
    """Test the minimizer with tuned parameters."""
    click.echo("Testing minimizer...")
    test_minimizer(minimizer_to_test=minimize, n_tuning_trials=50)


if __name__ == '__main__':
    cli()
