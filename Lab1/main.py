from solver import solver
from cec2017.functions import f3, f12
import autograd.numpy as np
from matplotlib import pyplot as plt


def init_plot(function):
    MAX_X = 100
    PLOT_STEP = 0.1

    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    q = function
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.zeros(10)
            point[0] = X[i, j]
            point[1] = Y[i, j]
            Z[i, j] = q(point)

    plt.contour(X, Y, Z, 20)


def quadriatic(x):
    return np.sum(x ** 2)


def main(function):

    init_plot(function)
    x0 = np.random.uniform(-100, 100, 10)
    result, iterations, values, time = solver(function, x0, 0.1, 1e-4, 1000, 100)
    plt.show()
    plt.clf()
    print(np.round(result, decimals=2))
    print(np.round(function(result), decimals=2))
    print(f'Time taken: {time:.2f}')
    plt.plot(range(iterations), values)
    plt.ylabel('q(xt) - wartość funkcji celu dla iteracji')
    plt.xlabel('t - numer iteracji')
    plt.show()


if __name__ == "__main__":
    main(quadriatic)
