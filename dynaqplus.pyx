import numpy as np


class DYNAQPlus:
    def __init__(self, states_n, actions_n, alpha, gamma, epsilon, kappa):
        self.states_n = states_n
        self.actions_n = actions_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.kappa = kappa
        self.reset()

    def reset(self):
        self.episode = 0
        self.step = 0
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.reward = 0
        self.q_table = np.zeros((self.states_n, self.actions_n))
        self.model = {}
        self.visited_states = {}
        self.visited_at = {}

    def start_episode(self):
        self.episode += 1
        self.step = 0
        self.visited_at = {}

    def update(self, state, action, next_state, reward):
        self._update(state, action, next_state, reward)
        key = (state, next_state)
        last_visit = self.visited_at.get(key)
        if last_visit is None:
            last_visit = 0
        
        tau = self.step - last_visit
        self.visited_at[key] = self.step

        reward = reward + self.kappa * np.sqrt(tau)
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[state, action]
        )
        self.model[(state, action)] = (reward, next_state)
        if state in self.visited_states:
            if action not in self.visited_states[state]:
                self.visited_states[state].append(action)
        else:
            self.visited_states[state] = [action]

    def _update(self, state, action, next_state, reward):
        self.step += 1
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def get_action(self, state, mode):
        if mode == "random":
            return np.random.choice(self.actions_n)
        elif mode == "greedy":
            return np.argmax(self.q_table[state])
        elif mode == "epsilon-greedy":
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.actions_n)
            else:
                return np.argmax(self.q_table[state])

    def render(self, mode="step"):
        if mode == "step":
            print(
                f"Episode: {self.episode}, Step: {self.step}, State: {self.state}, "
            )
            print(
                f"Action: {self.action}, Next state: {self.next_state}, Reward: {self.reward}"
            )

        elif mode == "values":
            print(f"Q-Table: {self.q_table}")
