import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Protocol, Tuple
from gym import Env


class Strategy(Protocol):
    def select_action(self, q_table: NDArray[np.float64], state: int) -> int:
        """
        Should return an action choosen using the strategy.
        """
        pass


class Environment(Protocol):
    def reset(self) -> int:
        """
        Should return a state after resetting the environment.
        """
        pass

    def make_step(self, action: int) -> Tuple[int, int, bool]:
        """
        Should apply the given action to the environment and return:
        new state
        reward for action
        information (boolean) if the state is final
        """
        pass


@dataclass
class AlgorithmParams:
    learning_rate: float
    discount_factor: float
    epochs: int
    q_table: NDArray[np.float64]
    explo_strat: Strategy
    env: Environment


def q_learning(params: AlgorithmParams):
    state = params.env.reset()[0]
    total_rewards = 0
    q_table = params.q_table
    for epoch in range(params.epochs):

        action = params.explo_strat.select_action(q_table, state)

        new_state, reward, done, _, _ = params.env.make_step(action)
        total_rewards += reward
        factor = (
            reward
            + params.discount_factor * np.max(q_table[new_state, :])
            - q_table[state, action]
        )
        q_table[state, action] = q_table[state, action] + params.learning_rate * factor

        state = new_state

        if done:
            return q_table, total_rewards
    return q_table, total_rewards
