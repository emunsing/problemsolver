#!/usr/bin/env python3
"""
Test suite for all optimizers in the problemsolver package.
"""

import numpy as np
import sys
import os
from typing import Callable, Annotated, get_origin, get_args

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from problemsolver.optimizers import OPTIMIZERS
from problemsolver.function_generators import fun_nonlinear as fun_generator
from problemsolver.utils import Interval
import inspect


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


def test_all_optimizers():
    """Test all optimizers on sample problems from each function type."""
    print("Testing optimizers on sample problems...")
    optimizer_errors = []

    # Test each optimizer on each function
    for optimizer_name, optimizer in OPTIMIZERS.items():
        print(f"  Testing {optimizer_name}...")

        try:
            check_optimizer_annotations(optimizer)
            check_optimizer_function(optimizer)

        except Exception as e:
            print(f"    ✗ {optimizer_name}: {str(e)}")
            optimizer_errors.append((optimizer_name, str(e)))
            # Don't raise here - just log the error and continue

    assert not optimizer_errors, f"Errors in optimizers: {optimizer_errors}"
