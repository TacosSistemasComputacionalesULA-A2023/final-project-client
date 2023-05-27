import sys
import numpy as np
import time
import datetime

def run(env, agent, selection_method, episodes):
    steps_episode = []
    episodes_results = {}
    terminated, truncated = False, False
    terminated_count, truncated_count = 0, 0
    for episode in range(episodes):
        observation, _ = env.reset()
        agent.start_episode()
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated):
            action = agent.get_action(observation, selection_method)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, next_observation, reward)
            observation = next_observation
            steps += 1
        steps_episode.append(steps)

        result=False
        if terminated:
            terminated_count += 1
            result=True
        if truncated:
            truncated_count += 1

        episodes_results[episode] = result

        if selection_method == "epsilon-greedy" and not truncated:
            for _ in range(100):
                state = np.random.choice(list(agent.visited_states.keys()))
                action = np.random.choice(agent.visited_states[state])
                reward, next_state = agent.model[(state, action)]
                agent.update(state, action, next_state, reward)
    
    return steps_episode, terminated, truncated, terminated_count, truncated_count, episodes_results
                

