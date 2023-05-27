from src import experiment
from src.agents import dynaq
import time
import datetime
import gym
import gym_taco_environments
import csv
import numpy as np
import sys
import argparse
 
parser = argparse.ArgumentParser(prog='experiments')
parser.add_argument("--knowledge", help="Reads from the knowledge base of over 200000 episodes", default=False, action='store_true')
parser.add_argument("episodes", help="Episodes the experiment will run", type=int)
parser.add_argument("render_mode", help="Mode in which it will be rendered", type=str)
parser.add_argument("-d", "--delay", help="Animation delay", type=float, default=0.5)


ARGUMENT_COUNT = 4

if __name__ == "__main__":
    args = parser.parse_args()

    episodes = args.episodes
    render_mode = args.render_mode if args.render_mode != 'None' else None
    delay = args.delay

    environment = "BlockyRocks-v0"
    start = time.time()
    alpha = 1.0
    gamma = 0.95
    epsilon = 0.1
    kappa = 0.0001
    terminated_count = 0
    truncated_count = 0

    steps_per_episodes = np.zeros(episodes)

    env1 = gym.make(environment, render_mode=render_mode, delay=delay)
    dynagent = dynaq.DYNAQ(
        env1.observation_space.n,
        env1.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    if args.knowledge:
        dynaq_learned = None
        try:
            dynaq_learned = open("dynaq.json", "r", newline="")
        except:
            pass

        if dynaq_learned:
            with dynaq_learned:
                dynagent.load(dynaq_learned.read())

    # Train
    episodes_steps, _, _, terminated_count, truncated_count, episodes_results = experiment.run(
        env1, dynagent, "epsilon-greedy", episodes
    )
    env1.close()

    steps_per_episodes = np.add(steps_per_episodes, episodes_steps)

    print(f"Time taken: {datetime.timedelta(seconds=time.time() - start)}")
    print(f"Success: {terminated_count} / {episodes}")
    print(f"Failure: {truncated_count} / {episodes}")

    counter = 0
    checked1, checked50, checked100, checked150 = False, False, False, False
    for episode, success in episodes_results.items():
        if success:
            counter += 1
        else:
            counter = 0

        if counter == 1 and not checked1:
            print(f"Success streak 1: Episode Start: {episode - 1}")
            checked1 = True
        elif counter == 50 and not checked50:
            print(f"Success streak 50: Episode Start: {episode - 50}")
            checked50 = True
        elif counter == 100 and not checked100:
            print(f"Success streak 100: Episode Start: {episode - 100}")
            checked100 = True
        elif counter == 150 and not checked150:
            print(f"Success streak 150: Episode Start: {episode - 150}")
            checked150 = True

    print(f"Last success streak of: {counter} wins")

    dynaqfile = open("dynaq.csv", "w", newline="")
    success_rate = open("success_rate.csv", "w", newline="")
    with dynaqfile and success_rate:
        dynaqwriter = csv.writer(dynaqfile)
        dynaqwriter.writerow(["episode", "steps"])
        for i in range(episodes):
            dynaqwriter.writerow([i, steps_per_episodes[i]])

        success_rate_writer = csv.writer(success_rate)
        success_rate_writer.writerow(["episode", "success"])
        for i in range(episodes):
            success_rate_writer.writerow([i, int(episodes_results[i])])

    if args.knowledge:
        dynaq_learned = open("dynaq.json", "w", newline="")
        with dynaq_learned:
            dynaq_learned.write(dynagent.serialize())
