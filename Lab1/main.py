from solver import solver
from cec2017.functions import f3, f12
import autograd.numpy as np
from matplotlib import pyplot as plt


def init_plot(function):
    MAX_BOUND = 100
    PLOT_STEP = 0.1

    x_arr = np.arange(-MAX_BOUND, MAX_BOUND, PLOT_STEP)
    y_arr = np.arange(-MAX_BOUND, MAX_BOUND, PLOT_STEP)
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


def plot_results(values, iterations, arrows=None, function=None):
    if arrows and function:
        init_plot(function)
        for arrow in arrows:
            plt.arrow(
                arrow[0][0],  # x
                arrow[0][1],  # y
                arrow[1][0] - arrow[0][0],  # dx
                arrow[1][1] - arrow[0][1],  # dy
                head_width=1, head_length=1, fc='k', ec='k'
                )
        plt.show()
        plt.clf()
    plt.plot(range(iterations), values)
    plt.ylabel('q(xt) - wartość funkcji celu dla iteracji')
    plt.xlabel('t - numer iteracji')
    plt.show()


def main():
    STEP_SIZE = 1e-8
    EPSILON = 1e-4
    MAX_ITERATIONS = 1000
    MAX_BOUND = 100
    FUNCTION = f3
    x0 = np.random.uniform(-MAX_BOUND, MAX_BOUND, 10)
    result, iterations, values, time, arrows = solver(
        FUNCTION, x0, STEP_SIZE, EPSILON, MAX_ITERATIONS, MAX_BOUND
        )

    print(np.round(result, decimals=2))
    print(np.round(FUNCTION(result), decimals=2))
    print(f'Czas działania: {np.round(time, decimals=5)} sekund.')

    plot_results(values, iterations, arrows, FUNCTION)


if __name__ == "__main__":
    main()
