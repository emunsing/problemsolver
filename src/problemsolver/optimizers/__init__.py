# Import all optimizers from subdirectories
from .bioinspired import *
from .numeric import *
from .sgd import *

# Combine all __all__ lists from subdirectories
__all__ = []

# Add bioinspired optimizers
from problemsolver.optimizers.bioinspired import __all__ as bioinspired_all
__all__.extend(bioinspired_all)

# # Add numeric optimizers
# from problemsolver.optimizers.numeric import __all__ as numeric_all
# __all__.extend(numeric_all)
#
# # Add SGD optimizers
# from problemsolver.optimizers.sgd import __all__ as sgd_all
# __all__.extend(sgd_all)

# Create a mapping of optimizer names to functions
OPTIMIZERS = {}

# Build the mapping from the imported functions
for name in __all__:
    if name.startswith('minimize_'):
        OPTIMIZERS[name] = globals()[name]

# Now you can import any optimizer like:
# from problemsolver.optimizers import minimize_pso, minimize_lbfgs, minimize_spsa
# Or access the mapping: from problemsolver.optimizers import OPTIMIZERS
