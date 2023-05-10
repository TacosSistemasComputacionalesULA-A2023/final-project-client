from src import experiment
from src.agents import dynaq
from src.agents import dynaqplus
import time
import datetime
import gym
import gym_taco_environments
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys

ARGUMENT_COUNT = 3

if __name__ == "__main__":
    if len(sys.argv) != ARGUMENT_COUNT:
        print("invalid number of arguments", len(sys.argv), "of", ARGUMENT_COUNT)
        sys.exit(-1)
    
    environment = sys.argv[1]
    episodes = int(sys.argv[2])
    render_mode = None
    delay = 0
    # episodes = int(sys.argv[3])
    
    start = time.time()
    alpha = 1.0
    gamma = 0.95
    epsilon = 0.1
    kappa = 0.0001
    terminated_count = 0
    terminated_plus_count = 0
    truncated_count = 0
    truncated_plus_count = 0

    steps_episodes = np.zeros(episodes)
    steps_episodes_plus = np.zeros(episodes)
    
    env1, env2 = gym.make(environment, render_mode=render_mode, delay=delay), gym.make(environment, render_mode=render_mode, delay=delay)
    dynagent = dynaq.DYNAQ(
        env1.observation_space.n, env1.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon
    )
    dynagentplus = dynaqplus.DYNAQPlus(
        env2.observation_space.n, env2.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon, kappa=kappa
    )
    
    dynaq_learned, dynaq_plus_learned = None, None
    try:
        dynaq_learned = open('dynaq.json', 'r', newline='')
        dynaq_plus_learned = open('dynaq_plus.json', 'r', newline='')
    except:
        pass
    
    if dynaq_learned and dynaq_plus_learned:
        with dynaq_learned and dynaq_plus_learned:
            dynagent.load(dynaq_learned.read())
            dynagentplus.load(dynaq_plus_learned.read())
    
    # Train
    episodes_step, _, _, terminated_count, truncated_count = experiment.run(
        env1, dynagent, "epsilon-greedy", episodes)
    env1.close()

    episodes_plus, _, _, terminated_plus_count, truncated_plus_count = experiment.run(
        env2, dynagentplus, "epsilon-greedy", episodes)
    env2.close()
    
    steps_episodes = np.add(steps_episodes, episodes_step)
    steps_episodes_plus = np.add(steps_episodes_plus, episodes_plus)

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')
    print(f'Succes: dynaq - {terminated_count} / {episodes}, {truncated_count} / {episodes}, dynaqplus - {terminated_plus_count} / {episodes}, {truncated_plus_count} / {episodes}')

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
            
    dynaq_learned = open('dynaq.json', 'w', newline='')
    dynaq_plus_learned = open('dynaq_plus.json', 'w', newline='')
    with dynaq_learned and dynaq_plus_learned:
        dynaq_learned.write(dynagent.serialize())
        dynaq_plus_learned.write(dynagentplus.serialize())
    
    