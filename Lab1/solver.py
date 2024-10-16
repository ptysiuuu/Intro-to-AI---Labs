from autograd import grad
import autograd.numpy as np
from time import time


def solver(
        f, x, step_size, epsilon, iterations, max_bound, plot_2d=False
):
    gradient_function = grad(f)
    values = []
    start = time()
    arrows = []
    for n in range(iterations):
        previous_x = x
        gradient = gradient_function(x)
        y = f(x)
        values.append(y)
        x = x - gradient * step_size
        x = np.clip(x, -max_bound, max_bound)
        if plot_2d:
            arrows.append([previous_x, x])
        if (
            (np.linalg.norm(gradient) <= epsilon)
        ):
            end = time()
            print(f'Solution found in {n} iterations.')
            return x, n + 1, values, end - start, arrows
    print("Solution not found.")
    end = time()
    return x, iterations, values, end - start, arrows
