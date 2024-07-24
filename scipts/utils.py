# Imports, as always
from os import makedirs, path
import numpy as np
from qiskit_algorithms.utils import algorithm_globals


# RNG.
def reset_seed(seed):
    if seed is None:
        return

    np.random.seed(seed)
    algorithm_globals.random_seed = seed
