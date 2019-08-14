"""
Implementation of an MaxQInit [Abel et al 2018]
"""

import numpy as np
from copy import deepcopy
from collections import defaultdict

from llrl.agents.rmax import RMax


def number_of_tasks_for_high_confidence_upper_bound(delta, min_sampling_probability):
    """
    Compute the required number of task for valid upper-bounds on the
    Q-value function with probability at least 1 - delta.
    :param delta: (float) uncertainty degree
    :param min_sampling_probability: (float) minimum sampling probability of an environment
    :return: (int)
    """
    return np.log(delta) / np.log(1. - min_sampling_probability)


class MaxQInit(RMax):
    def __init__(
            self,
            actions,
            gamma=0.9,
            r_max=1.,
            v_max=None,
            deduce_v_max=True,
            n_known=None,
            epsilon_q=0.1,
            epsilon_m=None,
            delta=None,
            n_states=None,
            deduce_n_known=True,
            min_sampling_probability=0.1,
            name="MaxQInit"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param r_max: (float) known upper-bound on the reward function
        :param v_max: (float) known upper-bound on the value function
        :param deduce_v_max: (bool) set to True to deduce v_max from r_max
        :param n_known: (int) count after which a state-action pair is considered known
        (only set n_known if delta and epsilon are not defined)
        :param epsilon_q: (float) precision of value iteration algorithm for Q-value computation
        :param epsilon_m: (float) precision of the learned models in L1 norm
        :param delta: (float) models are learned epsilon_m-closely with probability at least 1 - delta
        :param n_states: (int) number of states
        :param deduce_n_known: (bool) set to True to deduce n_known from (delta, n_states, epsilon_m)

        :param min_sampling_probability: (float) minimum sampling probability of an environment
        :param name: (str)
        """
        RMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                      n_known=n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                      deduce_n_known=deduce_n_known, name=name)

        self.min_sampling_probability = min_sampling_probability
        self.SA_memory = defaultdict(lambda: defaultdict(lambda: False))
        self.U_memory = []  # Upper-bounds on the Q-values of previous MDPs
        self.n_required_tasks = number_of_tasks_for_high_confidence_upper_bound(delta, min_sampling_probability)

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(actions=self.actions, gamma=self.gamma, r_max=self.r_max, v_max=self.v_max,
                      deduce_v_max=self.deduce_v_max, n_known=self.n_known, epsilon_q=self.epsilon_q,
                      epsilon_m=self.epsilon_m, delta=self.delta, n_states=self.n_states,
                      deduce_n_known=self.deduce_n_known, min_sampling_probability=self.min_sampling_probability,
                      name=self.name)

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        :return: None
        """
        self.update_memory()

        RMax.reset(self)

        if len(self.U_memory) > self.n_required_tasks:
            self.update_max_q_init_upper_bound()

    def update_memory(self):
        """
        Update the memory i.e. the set of known state-action pairs and the upper-bound on the Q-value function.
        :return: None
        """
        for s in self.R:
            for a in self.R[s]:
                if self.is_known(s, a):
                    self.SA_memory[s][a] = True

        self.U_memory.append(deepcopy(self.U))

    def update_max_q_init_upper_bound(self):
        """
        Update the bound on the Q-value with the MaxQInit method.
        :return: None
        """
        for s in self.SA_memory:
            for a in self.SA_memory[s]:
                self.U[s][a] = min(self.U[s][a], max([u[s][a] for u in self.U_memory]))
