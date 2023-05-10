import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class DYNAQ_Dumpable:
    def __init__(self, q_table, model, visited_states):
        self.q_table = q_table.tolist()
        self.model =   {",".join([str(k[0]), str(k[1])]): v for k, v in model.items()}
        self.visited_states = {str(k): v for k, v in visited_states.items()}

    def serialize(self):
        return json.dumps({
            "q_table": self.q_table,
            "model": self.model,
            "visited_states": self.visited_states
        }, cls=NpEncoder)

class DYNAQ:
    def __init__(self, states_n, actions_n, alpha, gamma, epsilon):
        self.states_n = states_n
        self.actions_n = actions_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
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

    def start_episode(self):
        self.episode += 1
        self.step = 0

    def update(self, state, action, next_state, reward):
        self._update(state, action, next_state, reward)
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

    def serialize(self):
        return DYNAQ_Dumpable(self.q_table, self.model, self.visited_states).serialize()
    
    def load(self, data):
        data = json.loads(data)
        self.q_table = np.array(data["q_table"])
        self.model = {(np.int64(k.split(",")[0]), np.int64(k.split(",")[1])): v for k, v in data["model"].items()}
        self.visited_states = {np.int64(k): v for k, v in data["visited_states"].items()}