from autograd import grad
import autograd.numpy as np
from matplotlib import pyplot as plt
from time import time


def solver(f, x, step_size, epsilon, iterations, max_bound):
    gradient_function = grad(f)
    values = []
    start = time()
    for n in range(iterations):
        previous_x = x
        y = f(x)
        values.append(y)
        gradient = gradient_function(x)
        x = x - gradient * step_size
        x = np.clip(x, -max_bound, max_bound)
        plt.arrow(previous_x[0], previous_x[1], x[0] - previous_x[0], x[1] - previous_x[1], head_width=1, head_length=1, fc='k', ec='k')
        if (
            (np.linalg.norm(gradient) <= epsilon)
        ):
            end = time()
            return x, n + 1, values, end - start
    print("Solution not found.")
    end = time()
    return x, iterations, values, end - start
