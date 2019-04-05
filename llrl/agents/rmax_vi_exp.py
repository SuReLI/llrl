from llrl.agents.rmax_vi import RMaxVI
from llrl.utils.utils import csv_write


class RMaxVIExp(RMaxVI):
    """
    Copy of RMaxVI for experiments:
    - Record number of time steps to convergence
    """

    def __init__(self, actions, gamma=0.9, count_threshold=1, epsilon=0.1, name="RMax-Exp", path="output.csv"):
        RMaxVI.__init__(self, actions=actions, gamma=gamma, count_threshold=count_threshold, epsilon=epsilon, name=name)

        self.n_time_steps = 0  # nb of time steps
        self.n_time_steps_cv = 0  # nb of time steps before convergence with high probability

        self.path = path

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        RMaxVI.reset(self)

        csv_write([self.n_time_steps, self.n_time_steps_cv], self.path, 'a')

        self.n_time_steps = 0
        self.n_time_steps_cv = 0

    def act(self, s, r):
        """
        Acting method called online during learning.

        :param s: int current state of the agent
        :param r: float received reward for the previous transition
        :return: return the greedy action wrt the current learned model.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a = self.greedy_action(s)

        self.prev_a = a
        self.prev_s = s

        self.n_time_steps += 1  # INCREASE COUNTER BY 1

        return a

    def update(self, s, a, r, s_p):
        """
        Updates transition and reward dictionaries with the input transition
        tuple if the corresponding state-action pair is not known enough.
        :param s: int state
        :param a: int action
        :param r: float reward
        :param s_p: int next state
        :return: None
        """
        if s is not None and a is not None:
            if self.counter[s][a] < self.count_threshold:
                self.counter[s][a] += 1
                self.R[s][a] += [r]
                self.T[s][a][s_p] += 1
                if self.counter[s][a] == self.count_threshold:
                    self.n_time_steps_cv = self.n_time_steps  # RECORD LAST TIME A PAIR WAS UPDATED
                    self.update_upper_bound()