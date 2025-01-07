import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from gym import Env


@dataclass
class AlgorithmParams:
    learning_rate: float
    discount_factor: float
    exploration_prob: float
    epochs: int
    q_table: NDArray[np.float64]


def q_learning(env: Env, params: AlgorithmParams):
    state = env.reset()[0]
    total_rewards = 0
    q_table = params.q_table
    for epoch in range(params.epochs):
        if params.exploration_prob > np.random.uniform(0, 1):
           action = np.argmax(q_table[state,:])
        else:
           action = env.action_space.sample()
        new_state, reward, done, _, _ = env.step(action)
        total_rewards += reward
        factor = (reward + params.discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
        q_table[state, action] = q_table[state, action] + params.learning_rate * factor

        state = new_state

        if done:
            return q_table, total_rewards
    return q_table, total_rewards
