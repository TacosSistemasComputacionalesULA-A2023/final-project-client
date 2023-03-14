import experiment
import dynaq
import dynaqplus
import time
import datetime
import gym
import gym_environments
import matplotlib.pyplot as plt
import csv

if __name__ == "__main__":
    start = time.time()
    environment = "Blocks-v0"
    episodes = 3500

    env1, env2 = gym.make(environment), gym.make(environment)

    dynagent = dynaq.DYNAQ(
        env1.observation_space.n, env1.action_space.n, alpha=1, gamma=0.95, epsilon=0.1
    )
    dynagentplus = dynaqplus.DYNAQPlus(
        env1.observation_space.n, env1.action_space.n, alpha=1, gamma=0.95, epsilon=0.1, kappa=0.001
    )

    # Train
    steps_episodes, _ , _ = experiment.run(env1, dynagent, "epsilon-greedy", episodes)
    env1.close()
    
    steps_episodes_plus, _ , _ = experiment.run(env1, dynagentplus, "epsilon-greedy", episodes)
    env2.close()

    # Play
    env2 = gym.make(environment)
    _, terminated, _ = experiment.run(env1, dynagentplus, "greedy", 2)
    dynagentplus.render()
    env2.close()
    print(terminated)

    env1 = gym.make(environment)
    _, terminated, _ = experiment.run(env1, dynagent, "greedy", 2)
    dynagent.render()
    env1.close()
    print(terminated)

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')
    
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['agent', 'episode', 'steps'])
        for i in range(episodes):
            writer.writerow(['dynaq', i, steps_episodes[i]])
            writer.writerow(['dynaq+', i, steps_episodes_plus[i]])


    # Plot
    plt.plot([i for i in range(episodes)], steps_episodes, label="DYNA-Q")
    plt.plot([i for i in range(episodes)], steps_episodes_plus, label="DYNA-QPlus")

    plt.xlabel("Episodes")
    plt.ylabel("Steps Per Episode")
    plt.title("Steps Per Episode vs Episodes")
    plt.legend()
    plt.show()
    