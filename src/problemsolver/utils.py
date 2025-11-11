import numpy as np
from typing import Callable, Annotated, Optional, Dict, List, get_origin, get_args
import attrs
from problemsolver.function_generators import fun_nonlinear as fun_generator, ProblemFunction
import re
import unicodedata
import logging
import multiprocessing as mp
import signal
import time
from functools import wraps

logger = logging.getLogger(__name__)

@attrs.define
class Performance:
    """Performance data for an optimizer, including benchmarking results and failure analysis."""
    
    # Core performance metrics
    name: str
    log_rel_error: Optional[float] = None
    time_elapsed: Optional[float] = None
    best_params: Dict = attrs.field(factory=dict)
    
    # Failure analysis fields
    failure_mode: Optional[str] = None
    error: Optional[str] = None
    closest_breakthrough_distance: Optional[float] = None
    error_breakthrough_gap: Optional[float] = None
    time_breakthrough_gap: Optional[float] = None
    best_error: Optional[float] = None
    best_time: Optional[float] = None
    needs_error_improvement: Optional[bool] = None
    needs_time_improvement: Optional[bool] = None
    closest_breakthrough_dimension: Optional[str] = None
    
    def is_successful(self) -> bool:
        """Check if the optimizer was successful (no failure mode)."""
        return self.failure_mode is None
    
    def to_csv_row(self) -> Dict[str, str]:
        """Convert to dictionary for CSV writing with proper string conversion."""
        return {
            'method_name': self.name,
            'log_rel_error': str(self.log_rel_error) if self.log_rel_error is not None else '',
            'time_elapsed': str(self.time_elapsed) if self.time_elapsed is not None else '',
            'best_params': str(self.best_params) if self.best_params else '',
            'failure_mode': self.failure_mode or '',
            'error': self.error or '',
            'closest_breakthrough_distance': str(self.closest_breakthrough_distance) if self.closest_breakthrough_distance is not None else '',
            'error_breakthrough_gap': str(self.error_breakthrough_gap) if self.error_breakthrough_gap is not None else '',
            'time_breakthrough_gap': str(self.time_breakthrough_gap) if self.time_breakthrough_gap is not None else '',
            'best_error': str(self.best_error) if self.best_error is not None else '',
            'best_time': str(self.best_time) if self.best_time is not None else '',
            'needs_error_improvement': str(self.needs_error_improvement) if self.needs_error_improvement is not None else '',
            'needs_time_improvement': str(self.needs_time_improvement) if self.needs_time_improvement is not None else '',
            'closest_breakthrough_dimension': self.closest_breakthrough_dimension or ''
        }
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'Performance':
        """Create Performance object from CSV row with proper type conversion."""
        def safe_float(value: str) -> Optional[float]:
            if not value or value.strip() == '':
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        def safe_bool(value: str) -> Optional[bool]:
            if not value or value.strip() == '':
                return None
            return value.lower() == 'true'
        
        def safe_eval_dict(value: str) -> Dict:
            if not value or value.strip() == '':
                return {}
            try:
                return eval(value) if value else {}
            except:
                return {}
        
        return cls(
            name=row.get('method_name', ''),
            log_rel_error=safe_float(row.get('log_rel_error', '')),
            time_elapsed=safe_float(row.get('time_elapsed', '')),
            best_params=safe_eval_dict(row.get('best_params', '')),
            failure_mode=row.get('failure_mode') or None,
            error=row.get('error') or None,
            closest_breakthrough_distance=safe_float(row.get('closest_breakthrough_distance', '')),
            error_breakthrough_gap=safe_float(row.get('error_breakthrough_gap', '')),
            time_breakthrough_gap=safe_float(row.get('time_breakthrough_gap', '')),
            best_error=safe_float(row.get('best_error', '')),
            best_time=safe_float(row.get('best_time', '')),
            needs_error_improvement=safe_bool(row.get('needs_error_improvement', '')),
            needs_time_improvement=safe_bool(row.get('needs_time_improvement', '')),
            closest_breakthrough_dimension=row.get('closest_breakthrough_dimension') or None
        )
    
    @classmethod
    def get_csv_fieldnames(cls) -> List[str]:
        """Get the fieldnames for CSV writing."""
        return [
            'method_name', 'log_rel_error', 'time_elapsed', 'best_params',
            'failure_mode', 'error', 'closest_breakthrough_distance',
            'error_breakthrough_gap', 'time_breakthrough_gap', 'best_error',
            'best_time', 'needs_error_improvement', 'needs_time_improvement',
            'closest_breakthrough_dimension'
        ]


def to_camel_case(text: str) -> str:
    # Convert text like "convert THIS_to–camelCASE!" to "ConvertThisToCamelCase"
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 2. Replace any sequence of non-alphanumeric characters with a single space
    text = re.sub(r'[^0-9A-Za-z]+', ' ', text)
    # 3. Split on whitespace, capitalize each word, and join
    parts = text.strip().split()
    return ''.join(word.capitalize() for word in parts)


def check_optimizer_annotations(optimizer: Callable):
    import inspect
    sig = inspect.signature(optimizer)

    has_annotated_param = False
    for param_name, param in sig.parameters.items():
        if param_name in ['fun', 'initial_guess']:
            continue

        anno = param.annotation
        if get_origin(anno) is Annotated:
            args = get_args(anno)
            if len(args) >= 2 and isinstance(args[1], Interval):
                has_annotated_param = True
                break

    if not has_annotated_param:
        raise ValueError(f"No Annotated parameters with Interval")

#
# def _optimizer_worker(test_func, initial_guess):
#     """
#     Worker function to run an optimizer with timeout protection.
#     This is a top-level function that can be pickled for multiprocessing.
#
#     Args:
#         test_func: The TransformedFunction object with optimizer assigned
#         initial_guess: Initial guess for optimization
#
#     Returns:
#         The optimization result
#     """
#     # Set up timeout handler for this process
#     def timeout_handler(signum, frame):
#         raise TimeoutError(f"Optimizer execution exceeded timeout")
#
#     # Set signal handler for timeout
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(30)  # Set a reasonable default timeout
#
#     try:
#         # The optimizer is accessible via test_func.optimizer
#         result = test_func.optimizer(fun=test_func.evaluate_at_x, initial_guess=initial_guess)
#         return result
#     finally:
#         # Always cancel the alarm, even if an exception occurred
#         try:
#             signal.alarm(0)
#         except Exception:
#             # Ignore any errors from signal.alarm during cleanup
#             pass


def _evaluate_single_function(wrapped_func: ProblemFunction,
                              max_allowed_time_per_function: float,
                              **kwargs) -> tuple[float, float]:
    """
    Worker function to evaluate a single test function.
    This is intended to be called from a Multiprocessing thread, which requires all arguments be pickled.
    The wrapped_func.optimizer in ProblemFunction allows us to make the optimizer (natively a callable) into a

    Args:
        args_tuple: (test_func, optimum, minimizer, kwargs, max_allowed_time_per_function, max_allowed_rolling_average_function_time)

    Returns:
        Tuple of (log_rel_error, problem_time)
    """

    # Set up timeout handler for this process
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function evaluation exceeded {max_allowed_time_per_function}s")

    # Set signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(max_allowed_time_per_function))

    try:
        problem_start_time = time.time()
        loss = wrapped_func.fit_and_report_loss(**kwargs)
        problem_elapsed_time = time.time() - problem_start_time

        return loss, problem_elapsed_time

    finally:
        # Always cancel the alarm, even if an exception occurred
        try:
            signal.alarm(0)
        except Exception:
            # Ignore any errors from signal.alarm during cleanup
            pass


def _run_optimizer_with_timeout(optimizer, test_func, initial_guess, timeout_seconds=10):
    """
    Run an optimizer function in a separate process with timeout.
    Returns the result or raises TimeoutError if timeout is exceeded.
    
    Args:
        optimizer: The optimizer function to run
        test_func: The TransformedFunction object (will have optimizer assigned to it)
        initial_guess: Initial guess for optimization
        timeout_seconds: Maximum time to allow for execution
    """
    # Assign the optimizer to the test_func object
    test_func.optimizer = optimizer
    
    # Use multiprocessing Pool with timeout protection (same pattern as multivariate_model_runner)
    pool = mp.Pool(processes=1)
    
    try:
        # Submit the task asynchronously
        result = pool.apply_async(_evaluate_single_function, args=(test_func, timeout_seconds))
        
        # Wait for result with timeout
        return result.get(timeout=timeout_seconds)
        
    except mp.TimeoutError:
        # Kill the pool and raise timeout error
        pool.terminate()
        pool.join()
        raise TimeoutError(f"Optimizer execution timed out after {timeout_seconds} seconds")
    except Exception as e:
        # Kill the pool and re-raise the error
        pool.terminate()
        pool.join()
        raise e
    finally:
        # Ensure pool is always properly cleaned up
        try:
            pool.terminate()
            pool.join()
        except (OSError, BrokenPipeError, ConnectionResetError):
            # Ignore common multiprocessing cleanup errors
            pass
        except Exception:
            # Ignore other cleanup errors
            pass


def check_optimizer_function(optimizer: Callable, timeout_seconds: float = 5.0):
    """
    Check if an optimizer function works correctly with timeout protection.
    
    Args:
        optimizer: The optimizer function to test
        timeout_seconds: Maximum time to allow for optimizer execution (default: 5.0)
    
    Raises:
        TimeoutError: If the optimizer takes longer than timeout_seconds
        Various assertion errors: If the optimizer doesn't meet requirements
    """
    func_name = 'rastrigin'
    n_dims = 10
    test_func, optimum_x = fun_generator.get_function_and_optimum(func_name, n_dims=n_dims)
    
    try:
        # Run optimizer with timeout protection
        # TODO: Return a standard result class to allow for better error-checking
        loss, _ = _run_optimizer_with_timeout(
            optimizer=optimizer, 
            test_func=test_func, 
            initial_guess=np.zeros(n_dims),
            timeout_seconds=timeout_seconds
        )
    except TimeoutError:
        raise TimeoutError(f"Optimizer execution timed out after {timeout_seconds} seconds")

    assert not np.isinf(loss), f"Returned inf values in loss estimate"
    assert not np.isnan(loss), f"Returned NaN values in loss estimate"


class Interval:
    """
    Optuna metadata class for use with parameter annotations using typing.Annotated
    Low and high are required, and must be numeric.
    Step is optional, and should be None if log=True.
    """
    def __init__(self, low: int | float, high: int | float, step: int | float | None=None, log: bool=False):
        self.low = low
        self.high = high
        self.step = step
        self.log = log


def setup_logging(log_level: str = "INFO"):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=numeric_level,
        force=True
    )
