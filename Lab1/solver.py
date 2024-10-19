import numpy as np
from time import time
from copy import deepcopy


def gradient_func(f, x, epsilon):
    H = epsilon
    y  = f(x)
    dimensions = len(x[0])
    gradient_value = np.zeros(dimensions)
    for n in range(dimensions):
        xn = deepcopy(x)
        xn[0][n] = xn[0][n] + H
        derivative = (f(xn) - y) / H
        gradient_value[n] = derivative
    return gradient_value



def solver(
        f, x, beta, epsilon, iterations, max_bound
):
    values = []
    start = time()
    for n in range(1, iterations + 1):
        previous_x = x
        gradient = gradient_func(f, x, epsilon)
        y = f(x)
        values.append(y)
        x = x - gradient * beta
        x = np.clip(x, -max_bound, max_bound)
        if (
            (np.linalg.norm(gradient) <= epsilon)
            or
            (np.linalg.norm(x - previous_x) <= epsilon)
        ):
            end = time()
            print(f'Solution found in {n} iterations.')
            return x, n, values, end - start
    print("Solution not found.")
    end = time()
    return x, iterations, values, end - start
