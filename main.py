import experiment
import dynaq
import dynaqplus
import time
import datetime
import gym
import gym_taco_environments
import matplotlib.pyplot as plt
import csv
import numpy as np

if __name__ == "__main__":
    start = time.time()
    environment = "BlockyRocks-v0"
    episodes = 100
    alpha = 1.0
    gamma = 0.95
    epsilon = 0.1
    kappa = 0.0001
    experiments = 10
    terminated_count = 0
    terminated_plus_count = 0
    truncated_count = 0
    truncated_plus_count = 0

    steps_episodes = np.zeros(episodes)
    steps_episodes_plus = np.zeros(episodes)
    for i in range(experiments):
        env1, env2 = gym.make(environment, render_mode='human', delay=0.0005), gym.make(environment, render_mode='human', delay=0.0005)

        dynagent = dynaq.DYNAQ(
            env1.observation_space.n, env1.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon
        )
        dynagentplus = dynaqplus.DYNAQPlus(
            env2.observation_space.n, env2.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon, kappa=kappa
        )

        # Train
        episodes_step, terminated, truncated = experiment.run(
            env1, dynagent, "epsilon-greedy", episodes)
        env1.close()

        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1

        episodes_plus, terminated_plus, truncated_plus = experiment.run(
            env2, dynagentplus, "epsilon-greedy", episodes)
        env2.close()

        if terminated_plus:
            terminated_plus_count += 1
        if truncated_plus:
            truncated_plus_count += 1

        steps_episodes = np.add(steps_episodes, episodes_step)
        steps_episodes_plus = np.add(steps_episodes_plus, episodes_plus)

        print(f'Experiment: {i}/{experiments}')
        
    steps_episodes /= experiments
    steps_episodes_plus /= experiments

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')
    print(f'Succes: dynaq - {terminated_count / episodes, truncated_count / episodes}, dynaqplus - {terminated_plus_count / episodes, truncated_plus_count / episodes}')

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