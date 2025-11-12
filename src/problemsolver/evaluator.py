from inspect import signature
from typing import Annotated, Callable, get_origin, get_args
import json
from functools import partial
from problemsolver.utils import Interval, _evaluate_single_function, setup_logging
from problemsolver.function_generators import ProblemFunction
from problemsolver.function_generators.fun_torch import generate_mlp_test_models
from problemsolver.function_generators.fun_nonlinear import generate_nonconvex_test_functions
import optuna
import multiprocessing as mp
import time
import numpy as np
import click
import matplotlib.pyplot as plt
import csv
import os
from problemsolver.optimizers import OPTIMIZERS  # Import the mapping

ROLLING_WINDOW_SIZE = 3

DEFAULT_SAVE_PATH = "src/problemsolver/data/output/optimizer_performance.csv"

FUNCTION_GENERATORS_AND_TIMEOUTS = {'nonconvex': (generate_nonconvex_test_functions, 5.0, 2.0),  # MAX_ALLOWED_PROBLEM_TIME, MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME
                                    'mlp': (generate_mlp_test_models, 20.0, 10.0)
                                    }
_, MAX_ALLOWED_PROBLEM_TIME, MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME = FUNCTION_GENERATORS_AND_TIMEOUTS['nonconvex']



def single_thread_multivariate_model_runner(minimizer: Callable,
                                            func_optima_tuples: list[tuple[ProblemFunction, np.ndarray]],
                                            max_time_per_function: float = 30.0,
                                            max_rolling_average_function_time: float = 20.0,
                                            **kwargs) -> tuple[float, float]:
    """
    Return a univariate metric for performance of the minimizer.  In this case, we use the log of the relative error,
    plus the mean time taken to run the minimization across a set of test functions.

    Kwargs are Optuna trial.suggest_* parameters.
    Full loss should be computed within this. Return the loss value for Optuna to minimize.
    return: float
    """
    losses = []
    problem_times =  [0.1 * max_rolling_average_function_time] * 3  # Keep a rolling average of the last 3 problem times, prepopulate with something safe
    time_start = time.time()

    for wrapped_func, optimum in func_optima_tuples:
        wrapped_func.optimizer = minimizer
        problem_start_time = time.time()
        loss = wrapped_func.fit_and_report_loss(**kwargs)
        problem_elapsed_time = time.time() - problem_start_time
        if problem_elapsed_time > max_time_per_function:
            raise TimeoutError(f"Problem took too long: {problem_elapsed_time:.2f}s")
        problem_times.append(problem_elapsed_time)
        losses.append(loss)
        problem_times = problem_times[1:]  # Keep only the last 3 times for averaging
        if np.mean(problem_times) > max_rolling_average_function_time:
            raise TimeoutError(f"Rolling average evaluation time too long: {np.mean(problem_times):.2f}s")

    time_elapsed = time.time() - time_start
    mean_time = time_elapsed / len(func_optima_tuples)
    print(f"Trial with params {kwargs} took total {time_elapsed:.03f}s, mean time {mean_time:.3f}s, mean log rel errors: {np.mean(losses):.3f}")

    return np.mean(losses), mean_time


def multivariate_model_runner(minimizer: Callable,
                              func_optima_tuples: list[tuple[Callable, np.ndarray]], 
                              n_jobs: int = None,
                              max_allowed_time_per_function: float = 30.0,
                              max_allowed_rolling_average_function_time: float = 20.0,
                              **kwargs) -> tuple[float, float]:
    """
    Parallel version of the multivariate model runner with proper timeout handling.
    
    Args:
        minimizer: The minimization function to test
        func_optima_tuples: List of (function, optimum) pairs
        max_allowed_time_per_function: Maximum time allowed for any single function evaluation
        max_allowed_rolling_average_function_time: Maximum allowed rolling average time
        **kwargs: Additional parameters for the minimizer
    
    Returns:
        Tuple of (mean_log_rel_error, mean_time)
    """
    n_jobs = n_jobs or max(mp.cpu_count() - 2, 1)
    if n_jobs == 1:
        return single_thread_multivariate_model_runner(minimizer=minimizer,
                                                       func_optima_tuples=func_optima_tuples,
                                                       max_time_per_function=max_allowed_time_per_function,
                                                       max_rolling_average_function_time=max_allowed_rolling_average_function_time,
                                                       **kwargs)

    log_rel_errors = []
    problem_times = [0.1 * max_allowed_rolling_average_function_time] * ROLLING_WINDOW_SIZE  # Rolling average prepopulation
    time_start = time.time()
    
    # Use multiprocessing with timeout protection
    pool = mp.Pool(processes=min(n_jobs, len(func_optima_tuples)))

    args_list = []
    for test_func, _ in func_optima_tuples:
        test_func.optimizer = minimizer  # Attach the minimizer to the function for use in the worker
        args_list.append((test_func, max_allowed_time_per_function))

    try:
        # Submit all tasks asynchronously
        results = [pool.apply_async(_evaluate_single_function, args=args, kwds=kwargs) for args in args_list]
        
        # Collect results with timeout monitoring
        for i, result in enumerate(results):
            try:
                # Wait for result with timeout
                log_rel_error, problem_elapsed_time = result.get(timeout=max_allowed_time_per_function)
                
                log_rel_errors.append(log_rel_error)
                problem_times.append(problem_elapsed_time)
                problem_times = problem_times[1:]  # Keep only the last 3 times
                
                # Check rolling average timeout
                current_rolling_avg = np.mean(problem_times)
                if current_rolling_avg > max_allowed_rolling_average_function_time:
                    # Kill all remaining processes using proper pool termination
                    pool.terminate()
                    pool.join()
                    raise TimeoutError(f"Rolling average time {current_rolling_avg:.2f}s exceeded limit {max_allowed_rolling_average_function_time}s")
                    
            except (mp.TimeoutError, TimeoutError, Exception) as e:
                # Catch both multiprocessing timeouts and worker exceptions
                # Kill all remaining processes using proper pool termination
                pool.terminate()
                pool.join()
                
                if isinstance(e, mp.TimeoutError):
                    raise TimeoutError(f"Function evaluation {i} exceeded {max_allowed_time_per_function}s")
                elif isinstance(e, TimeoutError):
                    raise e  # Re-raise worker timeout errors
                else:
                    raise TimeoutError(f"Function evaluation {i} failed with error: {e}")
                    
    finally:
        # Ensure pool is always properly cleaned up
        try:
            pool.terminate()
            pool.join()
        except (OSError, BrokenPipeError, ConnectionResetError):
            # Ignore common multiprocessing cleanup errors
            pass
        except Exception as e:
            # Log other errors but don't let them propagate
            print(f"Warning: Error during pool cleanup: {e}")
            pass
    
    time_elapsed = time.time() - time_start
    mean_time = time_elapsed / len(func_optima_tuples)
    
    print(f"Trial with params {kwargs} took total {time_elapsed:.2f}s, mean time {mean_time:.3f}s, mean log rel errors: {np.mean(log_rel_errors):.3f}")
    
    return np.mean(log_rel_errors), mean_time


def univariate_model_runner(**kwargs):
    log_rel_error, mean_time_elapsed = multivariate_model_runner(**kwargs)
    total_loss = np.mean(log_rel_error) + 100 * mean_time_elapsed
    return total_loss


def make_optuna_objective(minimizer_to_test: Callable,
                          func_optima_tuples: list[tuple[Callable, np.ndarray]],
                          n_jobs: int = None,
                          max_allowed_time_per_function: float = MAX_ALLOWED_PROBLEM_TIME,
                          max_allowed_rolling_average_function_time: float = MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME,
                          ) -> Callable:
    sig = signature(minimizer_to_test)

    # The term "trial" is magic used by Optuna
    def optuna_loss(trial):
        kwargs = {'minimizer': minimizer_to_test, 'func_optima_tuples': func_optima_tuples, 'n_jobs': n_jobs, 'max_allowed_time_per_function': max_allowed_time_per_function, 'max_allowed_rolling_average_function_time': max_allowed_rolling_average_function_time}
        for name, param in sig.parameters.items():
            if name in ['fun', 'initial_guess', 'params']:
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


def tune_minimizer(minimizer_to_test: Callable, tune_functions,
                   max_allowed_time_per_function: float = 30.0,
                   max_allowed_rolling_average_function_time: float = 20.0,
                   n_jobs=1, n_trials: int = 50):
    """
    Tune the minimizer using Optuna.

    :param minimizer_to_test: The minimizer function to tune.
    :param tune_functions: List of (function, optimum) pairs for tuning.
    :param n_trials: Number of trials for tuning.
    :param n_jobs: Number of parallel jobs to use.
    :return: The best parameters found by Optuna.
    """
    objective = make_optuna_objective(minimizer_to_test,func_optima_tuples=tune_functions,
                                      max_allowed_time_per_function=max_allowed_time_per_function,
                                      max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time,
                                      n_jobs=n_jobs)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def test_minimizer(minimizer_to_test: Callable, test_functions,
                   max_allowed_time_per_function: float = 30.0,
                   max_allowed_rolling_average_function_time: float = 20.0,
                   n_jobs=1):
    """
    Use default params

    :return: None
    """
    log_rel_errors, time_elapsed = multivariate_model_runner(minimizer=minimizer_to_test,
                             func_optima_tuples=test_functions,
                             max_allowed_time_per_function=max_allowed_time_per_function,
                             max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time,
                             n_jobs=n_jobs)
    print(f"Test results: time elapsed = {time_elapsed:.3f}s, mean log rel errors {log_rel_errors:.3f}")


def tune_test_minimizer(minimizer_to_test: Callable, tune_functions, test_functions,
                        max_allowed_time_per_function: float = 30.0,
                        max_allowed_rolling_average_function_time: float = 20.0,
                        n_jobs=1, n_tuning_trials: int = 50):
    """
    Test the minimizer with a set of test functions.

    :return: None
    """
    best_params = tune_minimizer(minimizer_to_test=minimizer_to_test, tune_functions=tune_functions, n_jobs=n_jobs, n_trials=n_tuning_trials)
    print("Best parameters found:", best_params)
    log_rel_errors, time_elapsed = multivariate_model_runner(minimizer=minimizer_to_test,
                                                             func_optima_tuples=test_functions,
                                                             max_allowed_time_per_function=max_allowed_time_per_function,
                                                             max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time,
                                                             n_jobs=n_jobs,
                                                             **best_params)
    print(f"Test results: time elapsed = {time_elapsed:.3f}s, mean log rel errors {log_rel_errors:.3f}")



def benchmark_optimizer(optimizer: Callable, 
                       test_functions, 
                       tune_functions, 
                       n_tuning_trials: int = 50,
                       n_jobs: int = None,
                       max_allowed_time_per_function: float = MAX_ALLOWED_PROBLEM_TIME,
                       max_allowed_rolling_average_function_time: float = MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME):
    # Tune the optimizer
    objective = make_optuna_objective(optimizer,
                                      func_optima_tuples=tune_functions,
                                      n_jobs=n_jobs,
                                      max_allowed_time_per_function=max_allowed_time_per_function,
                                      max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time)
    study = optuna.create_study(direction="minimize")
    start_time = time.time()
    study.optimize(objective, n_trials=n_tuning_trials)
    print(f"Profiling optuna {time.time() - start_time:.3f}")
    best_params = study.best_params
    
    # Test with tuned parameters
    start_time = time.time()
    log_rel_error, time_elapsed = multivariate_model_runner(
        minimizer=optimizer,
        func_optima_tuples=test_functions,
        n_jobs=n_jobs,
        max_allowed_time_per_function=max_allowed_time_per_function,
        max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time,
        **best_params
    )
    print(f"Profiling model runner {time.time() - start_time:.3f}")
    return log_rel_error, time_elapsed, best_params



def benchmark_all_optimizers(n_tune_functions: int = 2,
                             n_test_functions: int = 2,
                             n_tuning_trials: int = 10,
                             wrapped_function_generator: Callable | None = None,
                             n_dims: int = 2,
                             n_jobs: int = 1,
                             save_fig: str | None = None,
                             save_csv: str | None = None,
                             optimizer_names: list[str] | None = None,
                             max_allowed_time_per_function: float = MAX_ALLOWED_PROBLEM_TIME,
                             max_allowed_rolling_average_function_time: float = MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME,
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
    wrapped_function_generator = wrapped_function_generator or generate_nonconvex_test_functions
    tune_functions = wrapped_function_generator(n_samples=n_tune_functions, n_dims=n_dims)
    test_functions = wrapped_function_generator(n_samples=n_test_functions, n_dims=n_dims)

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
            log_rel_error, time_elapsed, best_params = benchmark_optimizer(optimizer=optimizer,
                                                                           test_functions=test_functions,
                                                                           tune_functions=tune_functions,
                                                                           n_tuning_trials=n_tuning_trials,
                                                                           n_jobs=n_jobs,
                                                                           max_allowed_time_per_function=max_allowed_time_per_function,
                                                                           max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time)
            
            results.append({
                'name': name,
                'log_rel_error': log_rel_error,
                'time_elapsed': time_elapsed,
                'best_params': best_params
            })
            
            print(f"  ✓ {name}: log_rel_error={log_rel_error:.3f}, time={time_elapsed:.3f}s")
            
        except Exception as e:
            print(f"  ✗ {name}: Failed - {str(e)}")
            continue
    
    # Create scatter plot and save results
    if results:
        create_benchmark_plot(results, save_fig=save_fig)
        save_benchmark_results_to_csv(results, save_csv=save_csv)
        
        # Print summary
        print("BENCHMARK SUMMARY")
        for result in sorted(results, key=lambda x: x['log_rel_error']):
            print(f"{result['name']:25} | log_rel_error: {result['log_rel_error']:8.3f} | time: {result['time_elapsed']:6.2f}s")
    
    return results


def save_benchmark_results_to_csv(results, save_csv: str | None = None):
    """Save benchmark results to CSV file."""
    if save_csv is None:
        save_csv = DEFAULT_SAVE_PATH
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    
    # Write results to CSV
    with open(save_csv, 'w', newline='') as csvfile:
        fieldnames = ['method_name', 'log_rel_error', 'time_elapsed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'method_name': result['name'],
                'log_rel_error': result['log_rel_error'],
                'time_elapsed': result['time_elapsed']
            })
    
    print(f"Results saved to '{save_csv}'")


def create_benchmark_plot(results, save_fig: str | None = None):
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
    if save_fig is not None:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_fig}'")
    plt.show()


@click.group()
def cli():
    pass



@cli.command()
@click.option('--n-tuning-trials', default=50, help='Number of trials for hyperparameter tuning')
@click.option('--optimizer', type=click.Choice(list(OPTIMIZERS.keys())), 
              default='minimize_pso', help='Which optimizer to tune')
@click.option('--n-tune-functions', default=2, help='Number of functions to use for tuning')
@click.option('--generator', default='nonconvex', type=click.Choice(FUNCTION_GENERATORS_AND_TIMEOUTS.keys()))
@click.option('--generator-kwargs', default='{}', type=str)
@click.option('--n-dims', default=2, help='Number of dimensions for the test functions')
@click.option('--n-jobs', default=1, help='Number of parallel jobs to use')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def tune(n_tuning_trials, optimizer, n_tune_functions, n_dims, n_jobs, generator, generator_kwargs, log_level):
    """Tune hyperparameters for a specific optimizer."""
    setup_logging(log_level)
    minimizer_func = OPTIMIZERS[optimizer]

    test_generator, max_problem_time, max_rolling_problem_time = FUNCTION_GENERATORS_AND_TIMEOUTS[generator]
    generator_kwargs = json.loads(generator_kwargs)
    test_generator_wrapped = partial(test_generator, **generator_kwargs)
    tune_functions = test_generator_wrapped(n_samples=n_tune_functions, n_dims=n_dims)
    best_params = tune_minimizer(minimizer_to_test=minimizer_func, n_jobs=n_jobs, tune_functions=tune_functions,
                                 max_allowed_time_per_function=max_problem_time,
                                    max_allowed_rolling_average_function_time=max_rolling_problem_time,
                                 n_trials=n_tuning_trials)
    
    click.echo(f"Best parameters found for {optimizer}:")
    for param, value in best_params.items():
        click.echo(f"  {param}: {value}")



@cli.command()
@click.option('--optimizer', type=click.Choice(list(OPTIMIZERS.keys())), 
              default='minimize_pso', help='Which optimizer to test')
@click.option('--n-dims', default=2, help='Number of dimensions for the test functions')                      
@click.option('--n-tuning-trials', default=50, help='Number of trials for hyperparameter tuning')
@click.option('--n-tune-functions', default=2, help='Number of functions to use for tuning')
@click.option('--n-test-functions', default=2, help='Number of functions to use for testing')
@click.option('--generator', default='nonconvex', type=click.Choice(FUNCTION_GENERATORS_AND_TIMEOUTS.keys()))
@click.option('--generator-kwargs', default='{}', type=str)
@click.option('--n-jobs', default=1, help='Number of parallel jobs to use')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def tune_test(optimizer, n_tuning_trials, n_tune_functions, n_test_functions, n_dims, n_jobs, generator, generator_kwargs, log_level):
    """Test a specific optimizer with tuned parameters."""
    setup_logging(log_level)
    minimizer_func = OPTIMIZERS[optimizer]

    test_generator, max_problem_time, max_rolling_problem_time = FUNCTION_GENERATORS_AND_TIMEOUTS[generator]
    generator_kwargs = json.loads(generator_kwargs)
    test_generator_wrapped = partial(test_generator, **generator_kwargs)
    tune_functions = test_generator_wrapped(n_samples=n_tune_functions, n_dims=n_dims)
    test_functions = test_generator_wrapped(n_samples=n_test_functions, n_dims=n_dims)
    click.echo(f"Testing {optimizer}...")
    tune_test_minimizer(minimizer_to_test=minimizer_func, tune_functions=tune_functions, test_functions=test_functions,
                        max_allowed_time_per_function=max_problem_time,
                        max_allowed_rolling_average_function_time=max_rolling_problem_time,
                        n_tuning_trials=n_tuning_trials, n_jobs=n_jobs)

@cli.command()
@click.option('--optimizer', type=click.Choice(list(OPTIMIZERS.keys())), 
              default='minimize_pso', help='Which optimizer to test')
@click.option('--n-dims', default=2, help='Number of dimensions for the test functions')                      
@click.option('--n-test-functions', default=2, help='Number of functions to use for testing')
@click.option('--n-jobs', default=1, help='Number of parallel jobs to use')
@click.option('--generator', default='nonconvex', type=click.Choice(FUNCTION_GENERATORS_AND_TIMEOUTS.keys()))
@click.option('--generator-kwargs', default='{}', type=str)
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def test(optimizer, n_test_functions, n_dims, n_jobs, generator, generator_kwargs, log_level):
    """Test a specific optimizer with tuned parameters."""
    setup_logging(log_level)
    minimizer_func = OPTIMIZERS[optimizer]

    test_generator, max_problem_time, max_rolling_problem_time = FUNCTION_GENERATORS_AND_TIMEOUTS[generator]
    generator_kwargs = json.loads(generator_kwargs)
    test_generator_wrapped = partial(test_generator, **generator_kwargs)
    test_functions = test_generator_wrapped(n_samples=n_test_functions, n_dims=n_dims)
    click.echo(f"Testing {optimizer}...")
    test_minimizer(minimizer_to_test=minimizer_func, test_functions=test_functions, n_jobs=n_jobs,
                   max_allowed_time_per_function=max_problem_time,
                   max_allowed_rolling_average_function_time=max_rolling_problem_time,
                   )



@cli.command()
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def list_optimizers(log_level):
    """List all available optimizers."""
    setup_logging(log_level)
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
@click.option('--save-fig', default=None, help='Path to save the plot')
@click.option('--save-csv', default=None, help='Path to save the CSV file')
@click.option('--n-dims', default=2, help='Number of dimensions for the test functions')
@click.option('--n-jobs', default=1, help='Number of parallel jobs to use')
@click.option('--generator', default='nonconvex', type=click.Choice(FUNCTION_GENERATORS_AND_TIMEOUTS.keys()))
@click.option('--generator-kwargs', default='{}', type=str)
@click.option('--seed', default=None, type=int, help='Random seed for reproducibility')
@click.option('--optimizers', multiple=True, type=click.Choice(list(OPTIMIZERS.keys())), 
              help='Specific optimizers to test (can specify multiple times). If not specified, test all optimizers.')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def benchmark(n_tune_functions, n_test_functions, n_tuning_trials, save_fig, save_csv, n_dims, n_jobs, seed, optimizers,
              generator, generator_kwargs, log_level):
    """Benchmark optimizers and create a scatter plot."""
    setup_logging(log_level)
    # Convert tuple to list, or None if empty
    optimizer_list = list(optimizers) if optimizers else None

    test_generator, max_problem_time, max_rolling_problem_time = FUNCTION_GENERATORS_AND_TIMEOUTS[generator]
    generator_kwargs = json.loads(generator_kwargs)
    test_generator_wrapped = partial(test_generator, **generator_kwargs)
    
    benchmark_all_optimizers(n_tune_functions=n_tune_functions,
                             n_test_functions=n_test_functions,
                             n_tuning_trials=n_tuning_trials,
                             wrapped_function_generator=test_generator_wrapped,
                             n_dims=n_dims,
                             n_jobs=n_jobs,
                             save_fig=save_fig,
                             save_csv=save_csv,
                             seed=seed,
                             optimizer_names=optimizer_list,
                             max_allowed_time_per_function=max_problem_time,
                             max_allowed_rolling_average_function_time=max_rolling_problem_time,
                             )


if __name__ == '__main__':
    cli()
