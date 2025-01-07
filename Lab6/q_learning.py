import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Protocol
from gym import Env


class ExplorationStrategy(Protocol):
    def __call__(self, q_table: NDArray[np.float64], state: int) -> int:
        """
        Should return an action choosen using the strategy.
        """
        pass


@dataclass
class AlgorithmParams:
    learning_rate: float
    discount_factor: float
    epochs: int
    q_table: NDArray[np.float64]
    explo_strat: ExplorationStrategy


def q_learning(env: Env, params: AlgorithmParams):
    state = env.reset()[0]
    total_rewards = 0
    q_table = params.q_table
    for epoch in range(params.epochs):

        action = params.explo_strat(q_table, state)

        new_state, reward, done, _, _ = env.step(action)
        total_rewards += reward
        factor = (reward + params.discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
        q_table[state, action] = q_table[state, action] + params.learning_rate * factor

        state = new_state

        if done:
            return q_table, total_rewards
    return q_table, total_rewards
