"""
Implementation of an R-Max agent [Brafman and Tennenholtz 2003]
Use Value Iteration to compute the R-Max upper-bound following [Strehl et al 2009].

Changes compared to Dave's original RMax:
- Use of Value Iteration for upper-bound computation with provable precision;
- Only use learned transition model when state-action pair is known;
- Single visit counter (less memory);
- end_of_episode method implemented avoiding using the initial state as the resulting state from the last transition.
"""

import random
import numpy as np
from collections import defaultdict

from simple_rl.agents.AgentClass import Agent


class RMaxVI(Agent):
    """
    Implementation of an R-Max agent [Brafman and Tennenholtz 2003]
    Use Value Iteration to compute the R-Max upper-bound following [Strehl et al 2009].
    """

    def __init__(self, actions, gamma=0.9, count_threshold=1, epsilon=0.1, name="RMax-VI-e"):
        name = name + str(epsilon) if name[-2:] == "-e" else name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.nA = len(self.actions)
        self.r_max = 1.0
        self.count_threshold = count_threshold
        self.vi_n_iter = int(np.log(1. / (epsilon * (1. - self.gamma))) / (1. - self.gamma))  # Nb of value iterations

        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

    def reset(self):
        """
        Reset the attributes to initial state.
        Save the previous model.
        :return: None
        """
        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

    def end_of_episode(self):
        """
        Reset between episodes within the same MDP.
        :return: None
        """
        self.prev_s = None
        self.prev_a = None

    def empty_memory_structure(self):
        """
        Empty memory structure:
        U[s][a] (float): upper-bound on the Q-value
        R[s][a] (list): list of collected rewards
        T[s][a][s'] (int): number of times the transition has been observed
        counter[s][a] (int): number of times the state action pair has been sampled
        :return: R, T, counter
        """
        return defaultdict(lambda: defaultdict(lambda: self.r_max / (1.0 - self.gamma))), \
               defaultdict(lambda: defaultdict(list)), \
               defaultdict(lambda: defaultdict(lambda: defaultdict(int))), \
               defaultdict(lambda: defaultdict(int))

    def display(self):
        """
        Display info about the attributes.
        """
        print('Displaying R-MAX-VI agent :')
        print('Action space           :', self.actions)
        print('Number of actions      :', self.nA)
        print('Gamma                  :', self.gamma)
        print('Count threshold        :', self.count_threshold)

    def is_known(self, s, a):
        return self.counter[s][a] >= self.count_threshold

    def get_nb_known_sa(self):
        return sum([self.is_known(s, a) for s, a in self.counter.keys()])

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
                    self.update_upper_bound()

    def greedy_action(self, s):
        """
        Compute the greedy action wrt the current upper bound.
        :param s: state
        :return: return the greedy action.
        """
        a_star = random.choice(self.actions)
        u_star = self.U[s][a_star]
        for a in self.actions:
            u_s_a = self.U[s][a]
            if u_s_a > u_star:
                u_star = u_s_a
                a_star = a
        return a_star

    def update_upper_bound(self):
        """
        Update the upper bound on the Q-value function.
        Called when a new state-action pair is known.
        :return: None
        """
        for i in range(self.vi_n_iter):
            for s in self.R:
                for a in self.R[s]:
                    n_s_a = float(self.counter[s][a])
                    r_s_a = sum(self.R[s][a]) / n_s_a

                    s_p_dict = self.T[s][a]
                    weighted_next_upper_bound = 0.
                    for s_p in s_p_dict:
                        weighted_next_upper_bound += self.U[s_p][self.greedy_action(s_p)] * s_p_dict[s_p] / n_s_a

                    self.U[s][a] = r_s_a + self.gamma * weighted_next_upper_bound
