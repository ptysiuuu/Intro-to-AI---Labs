from copy import deepcopy
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class SGDParamiters:
    function: Callable
    starting_point: np.object_
    beta: float
    epsilon: float
    iterations: int
    max_bound: float


def gradient_func(f, x, epsilon):
    H = epsilon
    y = f(x)
    dimensions = len(x[0])
    gradient_value = np.zeros(dimensions)
    for n in range(dimensions):
        xn = deepcopy(x)
        xn[0][n] = xn[0][n] + H
        derivative = (f(xn) - y) / H
        gradient_value[n] = derivative
    return gradient_value


def solver(params: SGDParamiters):
    '''
    SGD solver without stop conditions in order to compare it to ES solver.
    '''
    values = []
    x = params.starting_point
    for _ in range(1, params.iterations + 1):
        gradient = gradient_func(params.function, x, params.epsilon)
        y = params.function(x)
        values.append(y)
        x = x - gradient * params.beta
        x = np.clip(x, -params.max_bound, params.max_bound)
    return x, params.iterations, values
