from evolution_algorithm import solver as EA
from cec2017.functions import f3, f7
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Callable

@dataclass
class AlgorithmParamiters:
    function_param: Callable
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


def calculate_and_plot_deviations(trials, algorithm_paramiters):
    all_scores = []
    for _ in range(50):
        _, score, _ = EA(
        algorithm_paramiters.function_param, algorithm_paramiters.starting_point,
        algorithm_paramiters.mutation_strength, algorithm_paramiters.adaptaion_interval,
        algorithm_paramiters.max_iterations
        )
        all_scores.append(score)
    mean = np.mean(all_scores)
    std_deviation = np.std(all_scores)

    plt.hist(all_scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Średnia = {mean:.2f}')
    plt.axvline(mean - std_deviation, color='orange', linestyle='dotted', linewidth=1, label=f'-1σ = {mean - std_deviation:.2f}')
    plt.axvline(mean + std_deviation, color='orange', linestyle='dotted', linewidth=1, label=f'+1σ = {mean - std_deviation:.2f}')
    plt.xlabel('Algorithm scores', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.legend()
    plt.show()
    plt.clf()


def plot_results(algorithm_paramiters):
    result, score, all_evals = EA(
        algorithm_paramiters.function_param, algorithm_paramiters.starting_point,
        algorithm_paramiters.mutation_strength, algorithm_paramiters.adaptaion_interval,
        algorithm_paramiters.max_iterations
        )
    print(f'Result point X = {np.round(result[0], 8)}')
    print(f'Score: q(X) = {np.round(score, 8)}')
    plt.scatter(range(algorithm_paramiters.max_iterations + 1), all_evals, s=2, c='orange')
    plt.yscale('log')
    function_name = algorithm_paramiters.function_names[algorithm_paramiters.function_param]
    mutation_strength = algorithm_paramiters.mutation_strength
    adaptaion_interval = algorithm_paramiters.adaptaion_interval
    plt.title(
        f'{function_name} Sigma = {mutation_strength} Adaptation interval = {adaptaion_interval}',
        fontweight='bold'
        )
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)
    plt.show()
    plt.clf()


def main():
    MAX_BOUND = 100
    algorithm_paramiters = AlgorithmParamiters(
        my_f3, np.random.uniform(-MAX_BOUND, MAX_BOUND, size=(1, 10)),
        1.5, 5, 1000, {quadriatic: "Quadriatic function", my_f3: "F3", my_f7: "F7"}
    )
    plot_results(algorithm_paramiters)
    calculate_and_plot_deviations(50, algorithm_paramiters)


if __name__ == "__main__":
    main()
