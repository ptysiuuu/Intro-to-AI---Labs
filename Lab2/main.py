from evolution_algorithm import solver as es, EsStrategyParamiters
from cec2017.functions import f3, f7
import numpy as np
from matplotlib import pyplot as plt
from gradient_descent import solver as sgd, SGDParamiters
from typing import Callable
from scipy.stats import wilcoxon


def my_f3(params: np.array):
    return f3(params)[0]


def my_f7(params: np.array):
    return f7(params)[0]


def quadriatic(x: np.array):
    return np.sum(x**2)


def plot_results(params: EsStrategyParamiters | SGDParamiters):
    result, score, all_evals = es(params)
    print(f"Result point X = {np.round(result[0], 8)}")
    print(f"Score: q(X) = {np.round(score, 8)}")
    plt.scatter(range(params.max_iterations), all_evals, s=2, c="orange")
    if params.function != quadriatic:
        plt.yscale("log")
    plt.title(params.get_desc(), fontweight="bold")
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)


def plot_mean_results(
    params: EsStrategyParamiters | SGDParamiters, trials: int, algorithm: Callable
):
    """
    Plots mean values of points visited by ES(1+1) 1/5 or SGD strategy over given amount of trials.
    """
    all_trials = np.zeros((trials, params.max_iterations))
    for trial in range(trials):
        _, _, scores = algorithm(params)
        all_trials[trial, :] = scores
    mean = np.mean(all_trials, axis=0)
    if params.function != quadriatic:
        plt.yscale("log")
    plt.title(params.get_desc(), fontweight="bold")
    plt.ylabel("q(xt) - function value for iteration", fontweight="bold", fontsize=16.0)
    plt.xlabel("t - iteration number", fontweight="bold", fontsize=16.0)
    if params.get_label() == "ES(1+1)":
        color = "orange"
    else:
        color = "blue"
    plt.scatter(
        range(params.max_iterations),
        mean,
        label=f"{params.get_label()}, {trials} trials mean.",
        s=0.5,
        c=color,
    )


def test_wilcoxon(
    trials: int, es_params: EsStrategyParamiters, grad_params: SGDParamiters
):
    """
    Computes the wilcoxon test between the results of function optimalization of SGD and ES.
    Trials represents the amount of samples taken to consideration.
    """
    results_sgd = []
    results_es = []
    for _ in range(trials):
        _, _, all_values = sgd(grad_params)
        results_sgd.append(all_values[-1])
        _, result_es, _ = es(es_params)
        results_es.append(result_es)
        starting_point = np.random.uniform(
            grad_params.max_bound, grad_params.max_bound, size=(1, 10)
        )
        es_params.starting_point = starting_point
        grad_params.starting_point = starting_point
    statistic, p_value = wilcoxon(results_es, results_sgd)
    return statistic, p_value


def plot_es_vs_sgd(
    trials: int, es_params: EsStrategyParamiters, grad_params: SGDParamiters
):
    plot_mean_results(es_params, trials, es)
    plot_mean_results(grad_params, trials, sgd)
    func_name = es_params.function_names[es_params.function]
    plt.title(f"ES(1+1) vs SGD, {func_name}", fontweight="bold")
    plt.legend(prop={"weight": "bold"})


def main():
    MAX_BOUND = 100
    starting_point = np.random.uniform(-MAX_BOUND, MAX_BOUND, size=(1, 10))
    es_params = EsStrategyParamiters(
        my_f3,
        starting_point,
        1,
        5,
        1000,
        {quadriatic: "Quadriatic function", my_f3: "F3", my_f7: "F7"},
    )
    grad_params = SGDParamiters(
        my_f3,
        starting_point,
        1e-8,
        1e-6,
        3000,
        MAX_BOUND,
        {quadriatic: "Quadriatic function", my_f3: "F3", my_f7: "F7"},
    )
    stats, p = test_wilcoxon(20, es_params, grad_params)
    print('Statistics =', stats)
    print('P value =', p)


if __name__ == "__main__":
    main()
