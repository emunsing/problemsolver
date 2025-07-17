import numpy as np
from typing import Callable, Annotated, get_origin, get_args
from problemsolver.function_generators import fun_nonlinear as fun_generator

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


def check_optimizer_function(optimizer: Callable):
    func_name = 'rastrigin'
    n_dims = 10
    test_func, optimum_x = fun_generator.get_function_and_optimum(func_name, n_dims=n_dims)
    result_x = optimizer(fun=test_func, initial_guess=np.zeros(n_dims))
    result_f = test_func(result_x)
    assert result_x is not None, f"Returned None"
    assert isinstance(result_x, np.ndarray), f"Didn't return numpy array"
    assert result_x.shape == (n_dims,), f"Returned wrong shape"

    # Check for inf values in result
    assert not np.any(np.isinf(result_x)), f"Returned inf values in x estimate"
    assert not np.any(np.isnan(result_x)), f"Returned NaN values in x estimate"

    # Check function value at result
    assert not np.isinf(result_f), f"Produced solution with inf function value"
    assert not np.isnan(result_f), f"Produced solution with NaN function value"


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