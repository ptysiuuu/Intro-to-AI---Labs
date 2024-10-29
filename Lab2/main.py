from evolution_algorithm import solver as es, EsStrategyParamiters
from cec2017.functions import f3, f7
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from gradient_descent import solver as sgd, SGDParamiters
from scipy.stats import wilcoxon


def my_f3(params: np.array):
    return f3(params)[0]


def my_f7(params: np.array):
    return f7(params)[0]


def quadriatic(x: np.array):
    return np.sum(x ** 2)


def plot_results(params: EsStrategyParamiters):
    result, score, all_evals = es(params)
    print(f'Result point X = {np.round(result[0], 8)}')
    print(f'Score: q(X) = {np.round(score, 8)}')
    plt.scatter(range(params.max_iterations), all_evals, s=2, c='orange')
    if params.function != quadriatic:
        plt.yscale('log')
    function_name = params.function_names[params.function]
    mutation_strength = params.mutation_strength
    adaptaion_interval = params.adaptaion_interval
    plt.title(
        f'{function_name} Sigma = {mutation_strength} Adaptation interval = {adaptaion_interval}',
        fontweight='bold'
        )
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)


def plot_mean_results(params: EsStrategyParamiters, trials: int):
    all_trials = np.zeros((trials, params.max_iterations))
    for trial in range(trials):
        _, _, scores = es(params)
        all_trials[trial, :] = scores
    mean = np.mean(all_trials, axis=0)
    function_name = params.function_names[params.function]
    mutation_strength = params.mutation_strength
    adaptaion_interval = params.adaptaion_interval
    if params.function != quadriatic:
        plt.yscale('log')
    plt.title(
        f'{function_name}, sigma = {mutation_strength} a = {adaptaion_interval}',
        fontweight='bold'
        )
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)
    plt.scatter(
        range(params.max_iterations), mean,
        label=f'ES(1+1) Mean value from {trials} trials', s=0.5, c='orange')


def test_wilcoxon(trials: int, es_params: EsStrategyParamiters, grad_params: SGDParamiters):
    results_sgd = []
    results_es = []
    for _ in range(trials):
        _, _, all_values = sgd(grad_params)
        results_sgd.append(all_values[-1])
        _, result_es, _= es(es_params)
        results_es.append(result_es)
        starting_point = np.random.uniform(grad_params.max_bound, grad_params.max_bound, size=(1, 10))
        es_params.starting_point = starting_point
        grad_params.starting_point = starting_point
    statistic, p_value = wilcoxon(results_es, results_sgd)
    return statistic, p_value


def main():
    MAX_BOUND = 100
    starting_point = np.random.uniform(-MAX_BOUND, MAX_BOUND, size=(1, 10))
    es_params = EsStrategyParamiters(
        my_f7, starting_point,
        1.5, 10, 1000, {quadriatic: "Quadriatic function", my_f3: "F3", my_f7: "F7"}
    )
    grad_params = SGDParamiters(
        my_f7, starting_point, 1e-8, 1e-6, 1000, MAX_BOUND
    )
    plot_mean_results(es_params, 50)
    plt.show()


if __name__ == "__main__":
    main()
