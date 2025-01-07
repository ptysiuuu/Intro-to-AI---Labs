import gym
from q_learning import *
from matplotlib import pyplot as plt


if __name__ == "__main__":
    LEARNING_RATE = 0.7
    DISCOUNT_FACTOR = 0.62
    EXPLORATION_PROB = 1
    EPOCHS = 100
    TRAIN_EPISODES = 2000
    TEST_EPISODES = 100

    env = gym.make("Taxi-v3", render_mode='human').env
    env.reset()
    env.render()

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    params = AlgorithmParams(LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_PROB, EPOCHS, q_table)
    total_rewards = []
    for episode in range(TRAIN_EPISODES):
        q_table, episode_rewards = q_learning(env, params)
        total_rewards.append(episode_rewards)
        plt.plot(range(TEST_EPISODES), total_rewards)
