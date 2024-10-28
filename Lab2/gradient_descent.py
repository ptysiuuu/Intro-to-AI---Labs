from copy import deepcopy
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class GradientDescentParamiters:
    function: Callable
    starting_point: np.array
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


def solver(params: GradientDescentParamiters):
    values = []
    x = params.starting_point
    for n in range(1, params.iterations + 1):
        previous_x = x
        gradient = gradient_func(params.function, x, params.epsilon)
        y = params.function(x)
        values.append(y)
        x = x - gradient * params.beta
        x = np.clip(x, -params.max_bound, params.max_bound)
        if (np.linalg.norm(gradient) <= params.epsilon) or (
            np.linalg.norm(x - previous_x) <= params.epsilon
        ):
            return x, n, values
    return x, params.iterations, values
