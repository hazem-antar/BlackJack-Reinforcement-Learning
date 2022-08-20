import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 1
        self.alpha = 0.08

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        Qs = self.Q[state]
        if np.all(Qs == Qs[0]): 
            return np.random.choice(self.nA)
        else:
            max_value = np.max(Qs)
            max_count = np.count_nonzero(Qs == max_value)
            indices = np.where(Qs == max_value)
            probs = np.ones(self.nA) * (self.eps / self.nA)
            probs[indices] += (1 - self.eps) / max_count
            return np.random.choice(self.nA, p = probs)

    def expected_reward(self, state):
        """ Given the state, calculate the expected reward.

         Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - reward: a number, expected final reward from all possible actions
        """
        rewards = self.Q[state]
        if np.all(rewards == rewards[0]):
            return rewards[0]
        else:
            max_value = np.max(rewards)
            max_count = np.count_nonzero(rewards == max_value)
            indices = np.where(rewards == max_value)
            probs = np.ones(self.nA) * (self.eps / self.nA)
            probs[indices] += (1 - self.eps) / max_count
            return np.dot(probs, rewards)
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            self.Q[state][action] += self.alpha * (reward + self.expected_reward(next_state) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            