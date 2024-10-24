from evolution_algorithm import solver as EA
from cec2017.functions import f3, f7
import numpy as np
from matplotlib import pyplot as plt


def my_f3(params):
    return f3(params)[0]


def my_f7(params):
    return f7(params)[0]


def quadriatic(x):
    return np.sum(x ** 2)


def main():
    FUNCTION_NAMES = {quadriatic: "Quadriatic function", my_f3: "F3", my_f7: "F7"}
    MUTATION_STRENGTH = 1.5
    MAX_BOUND = 100
    ADAPTAION_INTERVAL = 5
    MAX_ITERATIONS = 1000
    FUNCTION = my_f3

    x0 = np.random.uniform(-MAX_BOUND, MAX_BOUND, size=(1, 10))
    result, score, all_scores = EA(FUNCTION, x0, MUTATION_STRENGTH, ADAPTAION_INTERVAL, MAX_ITERATIONS)
    print(f'Result point X = {np.round(result[0], 8)}')
    print(f'Score: q(X) = {np.round(score, 8)}')
    plt.scatter(range(MAX_ITERATIONS + 1), all_scores, s=2, c='orange')
    plt.yscale('log')
    plt.title(
        f'{FUNCTION_NAMES[FUNCTION]} Sigma = {MUTATION_STRENGTH} Adaptation interval = {ADAPTAION_INTERVAL}',
        fontweight='bold'
        )
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)
    plt.show()


if __name__ == "__main__":
    main()
