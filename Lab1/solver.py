from autograd import grad
import autograd.numpy as np


def solver(f, x, step_size, epsilon, iterations, max_bound):
    gradient_function = grad(f)
    values = []
    for n in range(iterations):
        values.append(f(x))
        gradient = gradient_function(x)
        x = x - gradient * step_size
        x = np.clip(x, -max_bound, max_bound)
        if np.linalg.norm(gradient) <= epsilon:
            return x, n + 1, values
    return x, iterations, values
