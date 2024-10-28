import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class EsStrategyParamiters:
    function: Callable
    starting_point: np.array
    mutation_strength: float
    adaptaion_interval: int
    max_iterations: int
    function_names: dict


def solver(params: EsStrategyParamiters):
    t = 1
    x = params.starting_point
    success_counter = 0
    current_score = params.function(x)
    scores = []
    while t <= params.max_iterations:
        mutant = x + params.mutation_strength * np.random.normal(0, 1, len(x[0]))
        mutant_score = params.function(mutant)
        if mutant_score <= current_score:
            success_counter += 1
            current_score = mutant_score
            x = mutant
        if t % params.adaptaion_interval == 0:
            if success_counter / params.adaptaion_interval > 1 / 5:
                params.mutation_strength *= 1.22
            if success_counter / params.adaptaion_interval < 1 / 5:
                params.mutation_strength *= 0.82
            success_counter = 0
        t += 1
        scores.append(current_score)
    return x, current_score, scores
