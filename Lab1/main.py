from solver import solver
from cec2017.functions import f3, f12
import numpy as np
from matplotlib import pyplot as plt


def my_f3(params):
    return f3(params)[0]


def my_f12(params):
    return f12(params)[0]


def quadriatic(x):
    return np.sum(x[0] ** 2)


def plot_results(values, iterations, semilogy=True):
    for n in range(len(values)):
        if semilogy:
            plt.semilogy(range(iterations[n]), values[n])
        else:
            plt.plot(range(iterations[n]), values[n])
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)


def main():
    FUNCTION_NAME = {quadriatic: "Quadriatic Function", my_f12: "F12", my_f3: "F3"}
    BETA = 0.1
    EPSILON = 1e-6
    MAX_ITERATIONS = 1500
    MAX_BOUND = 100
    FUNCTION = quadriatic
    TRIALS = 3

    all_values = []
    all_iterations = []

    for n in range(TRIALS):
        x0 = np.random.uniform(-MAX_BOUND, MAX_BOUND, size=(1, 10))
        result, iterations, values, time = solver(
            FUNCTION, x0, BETA, EPSILON, MAX_ITERATIONS, MAX_BOUND
        )
        print(f"Trial number {n + 1}.")
        print(f"Final point: X = {np.round(result, decimals=2)[0]}")
        print(f"q(X) = {np.round(FUNCTION(result), decimals=2)}")
        print(f"Time taken: {np.round(time, decimals=5)} s")
        all_values.append(values)
        all_iterations.append(iterations)
    plt.title(f"{FUNCTION_NAME[FUNCTION]}. Beta = {BETA}")
    plot_results(all_values, all_iterations, False)
    # plt.savefig(f'graphs/{FUNCTION_NAME[FUNCTION]}_{BETA}.png')
    plt.show()


if __name__ == "__main__":
    main()
