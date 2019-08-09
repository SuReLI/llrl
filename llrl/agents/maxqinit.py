"""
Implementation of an MaxQInit [Abel et al 2018]
"""

import copy
import numpy as np

from llrl.agents.rmax import RMax


class MaxQInit(RMax):
    def __init__(
            self,
            actions,
            gamma=0.9,
            count_threshold=None,
            epsilon_q=0.1,
            epsilon_m=None,
            delta=None,
            n_states=None,
            min_sampling_probability=0.1,
            name="MaxQInit"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param count_threshold: (int) count after which a state-action pair is considered known
        :param epsilon_q: (float) precision of value iteration algorithm for Q-value computation
        :param epsilon_m: (float) precision of the learned models in L1 norm
        :param delta: (float) uncertainty degree
        :param n_states: (int) number of states
        :param min_sampling_probability: (float) minimum sampling probability of an environment
        :param name: (str)
        """
        RMax.__init__(self, actions=actions, gamma=gamma, count_threshold=count_threshold, epsilon_q=epsilon_q,
                      epsilon_m=epsilon_m, delta=delta, n_states=n_states, name=name)

        self.SA_memory = defaultdict(lambda: defaultdict(lambda: False))
        self.U_memory = []  # Upper-bounds on the Q-values of previous MDPs
        self.n_required_tasks = np.log(delta) / np.log(1. - min_sampling_probability)

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        :return: None
        """
        for s in self.R:
            for a in self.R[s]:
                if self.is_known(s, a):
                    self.SA_memory[s][a] = True
        self.U_memory.append(copy.deepcopy(self.U))

        RMax.reset(self)

        if len(self.U_memory) > self.n_required_tasks:
            self.update_max_q_init_upper_bound()

    def update_max_q_init_upper_bound(self):
        """
        Update the bound on the Q-value with the MaxQInit method.
        :return: None
        """
        for s in self.SA_memory:
            for a in self.SA_memory[s]:
                self.U[s][a] = max([u[s][a] for u in self.U_memory])
