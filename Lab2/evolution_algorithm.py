import numpy as np
from typing import Callable


def solver(
        q: Callable, x: np.array, mutation_strength: float,
        adaptaion_interval: int, max_iterations: int
        ):
    t = 1
    success_counter = 0
    current_score = q(x)
    scores = []
    while t <= max_iterations:
        mutant = x + mutation_strength * np.random.normal(0, 1, len(x[0]))
        mutant_score = q(mutant)
        if mutant_score <= current_score:
            success_counter += 1
            current_score = mutant_score
            x = mutant
        if t % adaptaion_interval == 0:
            if success_counter / adaptaion_interval > 1 / 5:
                mutation_strength *= 1.22
            if success_counter / adaptaion_interval < 1 / 5:
                mutation_strength *= 0.82
            success_counter = 0
        t += 1
        scores.append(current_score)
    return x, current_score, scores
