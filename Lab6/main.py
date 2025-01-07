import gym
from q_learning import *
from matplotlib import pyplot as plt


class EpsilonGreedyStrategy:
    def __init__(self, epsilon: float, action_space_size: int):
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def __call__(self, q_table: NDArray[np.float64], state: int) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(q_table[state, :])


class BoltzmannStrategy:
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, q_table: NDArray[np.float64], state: int) -> int:
        q_values = q_table[state, :]
        exp_values = np.exp(q_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)

        return np.random.choice(len(q_values), p=probabilities)


class CountBasedStrategy:
    def __init__(self, action_space_size: int):
        self.visits = {}
        self.action_space_size = action_space_size

    def __call__(self, q_table: NDArray[np.float64], state: int) -> int:
        if state not in self.visits:
            self.visits[state] = np.zeros(self.action_space_size, dtype=int)

        visit_counts = self.visits[state]
        bonus = 1 / (1 + visit_counts)

        q_values = q_table[state, :] + bonus

        action = np.argmax(q_values)

        self.visits[state][action] += 1

        return action


if __name__ == "__main__":
    LEARNING_RATE = 0.7
    DISCOUNT_FACTOR = 0.62
    EPOCHS = 100
    TRAIN_EPISODES = 2000
    TEST_EPISODES = 100

    env = gym.make("Taxi-v3", render_mode="ansi").env
    env.reset()
    # env.render()

    epsilon_strat = EpsilonGreedyStrategy(1.0, env.action_space.n)
    boltzman_strat = BoltzmannStrategy(1.0)
    count_strat = CountBasedStrategy(env.action_space.n)

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    params = AlgorithmParams(LEARNING_RATE, DISCOUNT_FACTOR, EPOCHS, q_table, boltzman_strat)
    total_rewards = []
    for episode in range(TRAIN_EPISODES):
        q_table, episode_rewards = q_learning(env, params)
        total_rewards.append(episode_rewards)
        plt.plot(range(TEST_EPISODES), total_rewards)
