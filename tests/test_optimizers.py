#!/usr/bin/env python3
"""
Test suite for all optimizers in the problemsolver package.
"""

import sys
import os
from problemsolver.optimizers import OPTIMIZERS
from problemsolver.utils import check_optimizer_annotations, check_optimizer_function

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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
