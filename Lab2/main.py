from evolution_algorithm import solver as EA
from cec2017.functions import f3, f7
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Callable

@dataclass
class AlgorithmParamiters:
    function: Callable
    starting_point: np.array
    mutation_strength: float
    adaptaion_interval: int
    max_iterations: int
    function_names: dict


def my_f3(params):
    return f3(params)[0]


def my_f7(params):
    return f7(params)[0]


def quadriatic(x):
    return np.sum(x ** 2)


def plot_results(algorithm_paramiters):
    result, score, all_evals = EA(
        algorithm_paramiters.function, algorithm_paramiters.starting_point,
        algorithm_paramiters.mutation_strength, algorithm_paramiters.adaptaion_interval,
        algorithm_paramiters.max_iterations
        )
    print(f'Result point X = {np.round(result[0], 8)}')
    print(f'Score: q(X) = {np.round(score, 8)}')
    plt.scatter(range(algorithm_paramiters.max_iterations), all_evals, s=2, c='orange')
    plt.yscale('log')
    function_name = algorithm_paramiters.function_names[algorithm_paramiters.function]
    mutation_strength = algorithm_paramiters.mutation_strength
    adaptaion_interval = algorithm_paramiters.adaptaion_interval
    plt.title(
        f'{function_name} Sigma = {mutation_strength} Adaptation interval = {adaptaion_interval}',
        fontweight='bold'
        )
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)


def plot_mean_results(algorithm_paramiters, trials):
    all_trials = np.zeros((trials, algorithm_paramiters.max_iterations))
    for trial in range(trials):
        _, _, scores = EA(
        algorithm_paramiters.function, algorithm_paramiters.starting_point,
        algorithm_paramiters.mutation_strength, algorithm_paramiters.adaptaion_interval,
        algorithm_paramiters.max_iterations
        )
        all_trials[trial, :] = scores
    mean = np.mean(all_trials, axis=0)
    function_name = algorithm_paramiters.function_names[algorithm_paramiters.function]
    mutation_strength = algorithm_paramiters.mutation_strength
    adaptaion_interval = algorithm_paramiters.adaptaion_interval
    plt.title(
        f'{function_name} Sigma = {mutation_strength} Adaptation interval = {adaptaion_interval} Trials = {trials}',
        fontweight='bold'
        )
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)
    plt.scatter(
        range(algorithm_paramiters.max_iterations), mean,
        label='Mean value for target function', s=2, c='orange')


def main():
    MAX_BOUND = 100
    algorithm_paramiters = AlgorithmParamiters(
        my_f3, np.random.uniform(-MAX_BOUND, MAX_BOUND, size=(1, 10)),
        1.5, 5, 1000, {quadriatic: "Quadriatic function", my_f3: "F3", my_f7: "F7"}
    )
    plot_mean_results(algorithm_paramiters, 50)
    plt.show()


if __name__ == "__main__":
    main()
