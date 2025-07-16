from inspect import signature
from typing import Annotated, Callable, get_origin, get_args
from problemsolver.utils import Interval
from problemsolver.function_generators import fun_nonlinear as fun_generator
import optuna
import time
import numpy as np
import click
import matplotlib.pyplot as plt
from problemsolver.optimizers import OPTIMIZERS  # Import the mapping

def generate_test_functions(n_samples, n_dims, function_names = None) -> list[tuple[Callable, np.ndarray]]:
    # Generate a list of [function, optimum] pairs
    function_names = function_names or fun_generator.FUNCTIONS_AND_OPTIMA.keys()
    output_functions_and_optima = []
    for func_name in function_names:
        n_func_samples = 0
        while n_func_samples < n_samples:
            # Generate a function and its optimum
            func, optimum_x = fun_generator.get_function_and_optimum(func_name, n_dims=n_dims)
            if np.abs(func(optimum_x)) < 1e-6:
                print(f"Skipping {func_name} because its optimal value is near-zero")
                # Skip to avoid functions with near-zero optimal values which will create log errors
                continue
            else:
                output_functions_and_optima.append((func, optimum_x))
                n_func_samples += 1
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
        denominator = np.abs(test_func(optimum))
        assert denominator > 1e-3, "Optimal value should not be near-zero"
        x_hat = minimizer(fun=test_func, initial_guess=np.zeros(len(optimum)), **kwargs)
        numerator = np.abs(test_func(x_hat) - test_func(optimum))
        rel_error = numerator / denominator
        if rel_error <= 1e-12:
            log_rel_errors.append(-12)  # Avoid log-zero issues when very small numbers
        else:
            log_rel_errors.append(np.log10(rel_error))

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
                        if meta.log:
                            step = None
                        else:
                            step = meta.step if meta.step is not None else 1
                        kwargs[name] = trial.suggest_int(name, meta.low, meta.high,
                                                         step=step, log=meta.log)
                    else:
                        if meta.log:
                            step = None
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


def benchmark_all_optimizers(n_tune_functions: int = 2, n_test_functions: int = 2, 
                           n_tuning_trials: int = 10, n_dims: int = 2, save_path: str | None = None,
                           optimizer_names: list[str] | None = None,
                             seed: int | None = None):
    """
    Benchmark optimizers and create a scatter plot.
    
    Args:
        n_tune_functions: Number of functions to use for tuning
        n_test_functions: Number of functions to use for testing
        n_tuning_trials: Number of trials for hyperparameter tuning
        n_dims: Number of dimensions for the test functions
        save_path: Path to save the plot
        optimizer_names: List of optimizer names to test. If None, test all optimizers.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate test functions
    tune_functions = generate_test_functions(n_samples=n_tune_functions, n_dims=n_dims)
    test_functions = generate_test_functions(n_samples=n_test_functions, n_dims=n_dims)

    # Get optimizers to test
    if optimizer_names is None:
        # Test all optimizers
        optimizer_names = list(OPTIMIZERS.keys())
        optimizer_functions = list(OPTIMIZERS.values())
    else:
        # Test only specified optimizers
        optimizer_functions = []
        valid_names = []
        for name in optimizer_names:
            if name in OPTIMIZERS:
                optimizer_functions.append(OPTIMIZERS[name])
                valid_names.append(name)
            else:
                print(f"Warning: Optimizer '{name}' not found, skipping...")
        optimizer_names = valid_names
    
    print(f"Benchmarking {len(optimizer_names)} optimizers...")
    print(f"Tune functions: {n_tune_functions}, Test functions: {n_test_functions}")
    print(f"Tuning trials: {n_tuning_trials}, Dimensions: {n_dims}")
    print("-" * 60)
    
    results = []
    
    for i, (name, optimizer) in enumerate(zip(optimizer_names, optimizer_functions)):
        print(f"[{i+1}/{len(optimizer_names)}] Testing {name}...")
        
        try:
            # Tune the optimizer
            objective = make_optuna_objective(optimizer, func_optima_tuples=tune_functions)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_tuning_trials)
            best_params = study.best_params
            
            # Test with tuned parameters
            log_rel_error, time_elapsed = multivariate_model_runner(
                minimizer=optimizer,
                func_optima_tuples=test_functions,
                **best_params
            )
            
            results.append({
                'name': name,
                'log_rel_error': log_rel_error,
                'time_elapsed': time_elapsed,
                'best_params': best_params
            })
            
            print(f"  ✓ {name}: log_rel_error={log_rel_error:.3f}, time={time_elapsed:.2f}s")
            
        except Exception as e:
            print(f"  ✗ {name}: Failed - {str(e)}")
            continue
    
    # Create scatter plot
    if results:
        create_benchmark_plot(results, save_path=save_path)
        
        # Print summary
        print("BENCHMARK SUMMARY")
        for result in sorted(results, key=lambda x: x['log_rel_error']):
            print(f"{result['name']:25} | log_rel_error: {result['log_rel_error']:8.3f} | time: {result['time_elapsed']:6.2f}s")
    
    return results


def create_benchmark_plot(results, save_path: str | None = None):
    """Create a scatter plot of optimizer performance."""
    names = [r['name'] for r in results]
    log_errors = [r['log_rel_error'] for r in results]
    times = [r['time_elapsed'] for r in results]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(times, log_errors, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, name in enumerate(names):
        plt.annotate(name.replace('minimize_', ''), 
                    (times[i], log_errors[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.xlabel('Time Elapsed (seconds)')
    plt.ylabel('Log Relative Error')
    plt.title('Optimizer Performance Comparison\n(Lower and Left is Better)')
    plt.grid(True, alpha=0.3)
    
    # Add Pareto frontier
    pareto_points = []
    for i, (time, error) in enumerate(zip(times, log_errors)):
        is_pareto = True
        for j, (other_time, other_error) in enumerate(zip(times, log_errors)):
            if i != j and other_time <= time and other_error <= error:
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append((time, error))
    
    if pareto_points:
        pareto_times, pareto_errors = zip(*sorted(pareto_points))
        plt.plot(pareto_times, pareto_errors, 'r--', alpha=0.7, label='Pareto Frontier')
        plt.legend()
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    plt.show()


@click.group()
def cli():
    pass



@cli.command()
@click.option('--n-trials', default=50, help='Number of trials for hyperparameter tuning')
@click.option('--optimizer', type=click.Choice(list(OPTIMIZERS.keys())), 
              default='minimize_pso', help='Which optimizer to tune')
def tune(n_trials, optimizer):
    """Tune hyperparameters for a specific optimizer."""
    minimizer_func = OPTIMIZERS[optimizer]
    best_params = tune_minimizer(minimizer_to_test=minimizer_func, n_trials=n_trials)
    
    click.echo(f"Best parameters found for {optimizer}:")
    for param, value in best_params.items():
        click.echo(f"  {param}: {value}")



@cli.command()
@click.option('--optimizer', type=click.Choice(list(OPTIMIZERS.keys())), 
              default='minimize_pso', help='Which optimizer to test')
@click.option('--n-tuning-trials', default=50, help='Number of trials for hyperparameter tuning')
def test(optimizer, n_tuning_trials):
    """Test a specific optimizer with tuned parameters."""
    minimizer_func = OPTIMIZERS[optimizer]
    click.echo(f"Testing {optimizer}...")
    test_minimizer(minimizer_to_test=minimizer_func, n_tuning_trials=n_tuning_trials)


@cli.command()
def list_optimizers():
    """List all available optimizers."""
    click.echo("Available optimizers:")
    click.echo("-" * 40)
    for i, name in enumerate(sorted(OPTIMIZERS.keys()), 1):
        # Extract the algorithm name from the function name
        algo_name = name.replace('minimize_', '').replace('_', ' ').title()
        click.echo(f"{i:2d}. {name:25} ({algo_name})")
    click.echo(f"\nTotal: {len(OPTIMIZERS)} optimizers")


@cli.command()
@click.option('--n-tune-functions', default=3, help='Number of functions to use for tuning')
@click.option('--n-test-functions', default=3, help='Number of functions to use for testing')
@click.option('--n-tuning-trials', default=20, help='Number of trials for hyperparameter tuning')
@click.option('--save-path', default=None, help='Path to save the plot')
@click.option('--n-dims', default=2, help='Number of dimensions for the test functions')
@click.option('--seed', default=None, type=int, help='Random seed for reproducibility')
@click.option('--optimizers', multiple=True, type=click.Choice(list(OPTIMIZERS.keys())), 
              help='Specific optimizers to test (can specify multiple times). If not specified, test all optimizers.')
def benchmark(n_tune_functions, n_test_functions, n_tuning_trials, save_path, n_dims, seed, optimizers):
    """Benchmark optimizers and create a scatter plot."""
    # Convert tuple to list, or None if empty
    optimizer_list = list(optimizers) if optimizers else None
    
    benchmark_all_optimizers(n_tune_functions=n_tune_functions,
                             n_test_functions=n_test_functions,
                             n_tuning_trials=n_tuning_trials,
                             n_dims=n_dims,
                             save_path=save_path,
                             seed=seed,
                             optimizer_names=optimizer_list)


if __name__ == '__main__':
    cli()
