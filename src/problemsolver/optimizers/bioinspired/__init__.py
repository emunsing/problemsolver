from .pso import minimize as minimize_pso
from .simulated_annealing import minimize as minimize_simulated_annealing
from .genetic import minimize as minimize_ga
# from .firefly import minimize as minimize_firefly
from .aco import minimize as minimize_aco
from .differential_evolution import minimize as minimize_de
from .bees import minimize as minimize_bees
from .whale import minimize as minimize_wao

__all__ = [
    'minimize_pso',
    'minimize_simulated_annealing',
    'minimize_ga',
    # 'minimize_firefly',
    'minimize_aco',
    'minimize_de',
    'minimize_bees',
    'minimize_wao'
]
