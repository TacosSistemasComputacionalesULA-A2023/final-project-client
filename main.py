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
    episodes = 2000
    alpha = 1.0
    gamma = 0.95
    epsilon = 0.1
    kappa = 0.0001
    experiments = 10

    steps_episodes = np.zeros(episodes)
    steps_episodes_plus = np.zeros(episodes)
    for i in range(experiments):
        env1, env2 = gym.make(environment), gym.make(environment)

        dynagent = dynaq.DYNAQ(
            env1.observation_space.n, env1.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon
        )
        dynagentplus = dynaqplus.DYNAQPlus(
            env2.observation_space.n, env2.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon, kappa=kappa
        )

        # Train
        episodes_step, _, _ = experiment.run(
            env1, dynagent, "epsilon-greedy", episodes)
        env1.close()

        episodes_plus, _, _ = experiment.run(
            env2, dynagentplus, "epsilon-greedy", episodes)
        env2.close()

        steps_episodes = np.add(steps_episodes, episodes_step)
        steps_episodes_plus = np.add(steps_episodes_plus, episodes_plus)

        print(f'Experiment: {i}/{experiments}')
        
    steps_episodes /= experiments
    steps_episodes_plus /= experiments

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')

    dynaqfile = open('dynaq.csv', 'w', newline='')
    dynaqplusfile = open('dynaqplus.csv', 'w', newline='')
    with dynaqplusfile and dynaqfile:
        dynaqwriter = csv.writer(dynaqfile)
        dynaqpluswriter = csv.writer(dynaqplusfile)
        dynaqwriter.writerow(['episode', 'steps'])
        dynaqpluswriter.writerow(['episode', 'steps'])
        for i in range(episodes):
            dynaqwriter.writerow([i, steps_episodes[i]])
            dynaqpluswriter.writerow([i, steps_episodes_plus[i]])

    # Plot
    x = np.array([i for i in range(episodes)])
    plt.plot(x, steps_episodes, label=f"DYNA-Q (Goal? {terminated})")
    plt.plot(x, steps_episodes_plus,
             label=f"DYNA-QPlus (Goal? {terminated_plus})")

    plt.xlabel("Episodes")
    plt.ylabel("Steps Per Episode")
    plt.title(
        f"Steps Per Episode vs Episodes\nAlpha{alpha} Epsilon{epsilon} Gamma{gamma} Kappa{kappa}")
    plt.legend()
    plt.show()