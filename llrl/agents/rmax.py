"""
Implementation of an R-Max Agent [Brafman and Tennenholtz 2003]
"""

import copy
import random
from collections import defaultdict

import llrl.utils.utils as utils
import llrl.spaces.discrete as discrete
from simple_rl.agents.AgentClass import Agent


class RMax(Agent):
    """
    Implementation of an R-Max Agent [Brafman and Tennenholtz 2003]
    """

    def __init__(self, actions, gamma=0.9, horizon=3, count_threshold=1, name="RMax-h"):
        name = name + str(horizon) if name[-2:] == "-h" else name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.nA = len(self.actions)
        self.r_max = 1.0
        self.horizon = horizon
        self.count_threshold = count_threshold
        self.reset()

    def reset(self):
        """
        Reset the attributes to initial state.
        :return: None
        """
        self.Q_init = defaultdict(lambda: defaultdict(lambda: self.r_max / (1.0 - self.gamma)))  # heuristic
        self.R = defaultdict(lambda: defaultdict(list))  # collected rewards [s][a][r_1, ...]
        self.T = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # collected transitions [s][a][s'][count]
        self.counter = defaultdict(lambda: defaultdict(int))  # counter [s][a][count]
        self.prev_s = None
        self.prev_a = None

    def set(self, p=None):
        """
        Set the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p is None:
            self.__init__(self.actions)
        else:
            utils.assert_types(p, [discrete.Discrete, float, int, int])
            self.__init__(p[0], p[1], p[2], p[3])

    def display(self):
        """
        Display info about the attributes.
        """
        print('Displaying R-MAX agent :')
        print('Action space           :', self.actions)
        print('Number of actions      :', self.nA)
        print('Gamma                  :', self.gamma)
        print('Horizon                :', self.horizon)
        print('Count threshold        :', self.count_threshold)

    def set_initial_q_function(self, q):
        """
        Set the initial heuristic function.

        :param q: given heuristic.
        :return: None
        """
        self.Q_init = copy.deepcopy(q)

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

        _, a = self._compute_max_q_value_action_pair(s)

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
            if self.counter[s][a] <= self.count_threshold:
                self.R[s][a] += [r]
                self.counter[s][a] += 1
                self.T[s][a][s_p] += 1

    def _compute_max_q_value_action_pair(self, s, horizon=None):
        """
        Compute the greedy action wrt the current Q-value function.

        :param s: int state
        :param horizon: int horizon
        :return: return the tuple (q_star, a_star) with q_star the maximum Q-value and a_star the maximizing action.
        """
        if horizon is None:
            horizon = self.horizon
        a_star = random.choice(self.actions)
        q_star = self.compute_q_value(s, a_star, horizon)
        for a in self.actions:
            q_s_a = self.compute_q_value(s, a, horizon)
            if q_s_a > q_star:
                q_star = q_s_a
                a_star = a
        return q_star, a_star

    def compute_q_value(self, s, a, horizon=None):
        """
        Compute the learned Q-value at (s, a).

        :param s: int state
        :param a: int action
        :param horizon: int number of steps ahead
        :return: return Q(s, a)
        """
        if horizon is None:
            horizon = self.horizon

        r_s_a = self._get_reward(s, a)
        if horizon <= 0 or s.is_terminal():
            return r_s_a
        else:
            expected_future_return = self.gamma * self._compute_expected_future_return(s, a, horizon)
            q_val = r_s_a + expected_future_return
            return q_val

    def _compute_expected_future_return(self, s, a, horizon=None):
        """
        Compute the expected return 1 step ahead from (s, a)

        :param s: int state
        :param a: int action
        :param horizon: int number of steps ahead
        :return: return the expected return 1 step ahead from (s, a)
        """
        if horizon is None:
            horizon = self.horizon

        s_p_dictionary = self.T[s][a]

        denominator = float(sum(s_p_dictionary.values()))

        s_p_weights = defaultdict(float)

        for s_p in s_p_dictionary.keys():
            count = s_p_dictionary[s_p]
            s_p_weights[s_p] = (count / denominator)

        weighted_future_returns = [self._compute_max_q_value_action_pair(s_p, horizon - 1)[0] * s_p_weights[s_p]
                                   for s_p in s_p_dictionary.keys()]
        return sum(weighted_future_returns)

    def _get_reward(self, s, a):
        """
        Return the learned expected reward function at (s, a) if known.
        Else return the used heuristic.

        :param s: int state
        :param a: int action
        :return: return R(s, a)
        """
        if self.counter[s][a] >= self.count_threshold:
            collected_rewards = self.R[s][a]
            return float(sum(collected_rewards)) / float(len(collected_rewards))
        else:
            return self.r_max
