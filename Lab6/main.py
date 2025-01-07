from typing import Tuple
import gym
import numpy as np
from matplotlib import pyplot as plt
from q_learning import AlgorithmParams, q_learning, Strategy, Environment


class GymEnv(Environment):
    def __init__(self, env_name) -> None:
        self.env = gym.make(env_name)
    
    def reset(self):
        return self.env.reset()
    
    def make_step(self, action: int) -> Tuple[int, int, bool]:
        return self.env.step(action)


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, epsilon: float, action_space_size: int):
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def select_action(self, q_table: np.ndarray, state: int) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(q_table[state, :])

    def __str__(self) -> str:
        return f"Epsilon-Greedy (epsilon={self.epsilon})"


class BoltzmannStrategy(Strategy):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def select_action(self, q_table: np.ndarray, state: int) -> int:
        q_values = q_table[state, :]
        exp_values = np.exp(q_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(q_values), p=probabilities)

    def __str__(self) -> str:
        return f"Boltzmann (temperature={self.temperature})"


class CountBasedStrategy:
    def __init__(self, action_space_size: int):
        self.visits = {}
        self.action_space_size = action_space_size

    def select_action(self, q_table: np.ndarray, state: int) -> int:
        if state not in self.visits:
            self.visits[state] = np.zeros(self.action_space_size, dtype=int)

        visit_counts = self.visits[state]
        bonus = 1 / (1 + visit_counts)

        q_values = q_table[state, :] + bonus

        action = np.argmax(q_values)

        self.visits[state][action] += 1

        return action

    def __str__(self) -> str:
        return "Counter Strategy"


# Comparing strategies
def compare_strategies(env_name="Taxi-v3", episodes=500, trials=5):
    env = GymEnv(env_name)

    strategies = [
        EpsilonGreedyStrategy(epsilon=0.1, action_space_size=env.env.action_space.n),
        BoltzmannStrategy(temperature=1.0),
        CountBasedStrategy(action_space_size=env.env.action_space.n)
    ]

    plt.figure(figsize=(12, 6))

    for strategy in strategies:
        average_rewards = np.zeros(episodes)

        for trial in range(trials):
            q_table = np.zeros((env.env.observation_space.n, env.env.action_space.n))
            params = AlgorithmParams(
                learning_rate=0.9,
                discount_factor=0.99,
                epochs=episodes,
                q_table=q_table,
                explo_strat=strategy,
                env=env
            )

            rewards = []
            for _ in range(episodes):
                _, total_reward = q_learning(params)
                rewards.append(float(total_reward))

            average_rewards += np.array(rewards, dtype=float)

        average_rewards /= trials
        plt.plot(average_rewards, label=str(strategy))

    plt.xlabel("Episodes", fontweight="bold")
    plt.ylabel("Average Total Reward", fontweight="bold")
    plt.title(f"Comparison of Q-Learning Strategies ({trials} trials)", fontweight="bold")
    plt.legend()
    plt.show()


def compare_learning_rates(env_name="Taxi-v3", episodes=500, trials=3, learning_rates=None):
    if learning_rates is None:
        learning_rates = [0.01, 0.1, 0.5, 0.9]

    env = GymEnv(env_name)
    strategies = {
        "Counter Strategy": CountBasedStrategy(action_space_size=env.env.action_space.n),
        "Epsilon-Greedy": EpsilonGreedyStrategy(epsilon=0.1, action_space_size=env.env.action_space.n),
        "Boltzmann Strategy": BoltzmannStrategy(temperature=1.0)
    }

    for strategy_name, strategy in strategies.items():
        plt.figure(figsize=(12, 6))

        for lr in learning_rates:
            average_rewards = np.zeros(episodes)

            for trial in range(trials):
                q_table = np.zeros((env.env.observation_space.n, env.env.action_space.n))
                params = AlgorithmParams(
                    learning_rate=lr,
                    discount_factor=0.99,
                    epochs=episodes,
                    q_table=q_table,
                    explo_strat=strategy,
                    env=env
                )

                rewards = []
                for _ in range(episodes):
                    _, total_reward = q_learning(params)
                    rewards.append(float(total_reward))

                rewards_array = np.array(rewards, dtype=float)
                average_rewards += rewards_array

            average_rewards /= trials
            plt.plot(average_rewards, label=f"Learning Rate: {lr}")

        plt.xlabel("Episodes", fontweight="bold")
        plt.ylabel("Average Total Reward", fontweight="bold")
        plt.title(f"{strategy_name} - Impact of Learning Rate", fontweight="bold")
        plt.legend(fontsize=10, loc="best", title_fontsize="13", title="Learning Rates")
        plt.savefig(fname=f'{str(strategy)}_lr.png')


if __name__ == "__main__":
    compare_strategies()
    # compare_learning_rates()
