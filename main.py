import experiment
import dynaq
import dynaqplus
import time
import datetime
import gym
import gym_environments
import matplotlib.pyplot as plt
import csv
import numpy as np

if __name__ == "__main__":
    start = time.time()
    environment = "Blocks-v0"
    episodes = 1000
    alpha = 1.0
    gamma = 0.95
    epsilon = 0.1
    kappa = 0.0001

    env1, env2 = gym.make(environment), gym.make(environment)

    dynagent = dynaq.DYNAQ(
        env1.observation_space.n, env1.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon
    )
    dynagentplus = dynaqplus.DYNAQPlus(
        env1.observation_space.n, env1.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon, kappa=kappa
    )

    # Train
    steps_episodes, _, _ = experiment.run(
        env1, dynagent, "epsilon-greedy", episodes)
    env1.close()

    steps_episodes_plus, _, _ = experiment.run(
        env1, dynagentplus, "epsilon-greedy", episodes)
    env2.close()

    # Play
    env1 = gym.make(environment)
    _, terminated, _ = experiment.run(env1, dynagent, "greedy", 2)
    dynagent.render()
    env1.close()

    env2 = gym.make(environment)
    _, terminated_plus, _ = experiment.run(env1, dynagentplus, "greedy", 2)
    dynagentplus.render()
    env2.close()

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')

    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['agent', 'episode', 'steps'])
        for i in range(episodes):
            writer.writerow(['dynaq', i, steps_episodes[i]])
            writer.writerow(['dynaq+', i, steps_episodes_plus[i]])

    # Plot
    x = np.array([i for i in range(episodes)])
    # z1 = np.polyfit(x, np.array(steps_episodes), 6)
    # z2 = np.polyfit(x, np.array(steps_episodes_plus), 6)
    plt.plot(x, steps_episodes, label=f"DYNA-Q (Goal? {terminated})")
    plt.plot(x, steps_episodes_plus,
             label=f"DYNA-QPlus (Goal? {terminated_plus})")

    plt.xlabel("Episodes")
    plt.ylabel("Steps Per Episode")
    plt.title(f"Steps Per Episode vs Episodes\nAlpha{alpha} Epsilon{epsilon} Gamma{gamma} Kappa{kappa}")
    plt.legend()
    plt.show()
