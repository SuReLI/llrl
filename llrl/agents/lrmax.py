import copy
import random
from collections import defaultdict

import llrl.utils.utils as utils
import llrl.spaces.discrete as discrete
from simple_rl.agents.AgentClass import Agent


class LRMax(Agent):
    """
    Lipschitz R-Max agent.
    """

    def __init__(self, actions, gamma=0.9, horizon=3, count_threshold=1, name="LRMax-h"):
        name = name + str(horizon) if name[-2:] == "-h" else name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.nA = len(self.actions)
        self.r_max = 1.0
        self.horizon = horizon
        self.count_threshold = count_threshold
        self.prev_s = None
        self.prev_a = None

        # Learned model
        self.R, self.T, self.counter = self.empty_memory_structure()

        # Lifelong Learning memories
        self.U_memory = []
        self.R_memory = []
        self.T_memory = []

    def reset(self):
        """
        Reset the attributes to initial state.
        Save the previous model.

        TODO check whether reset is only applied when sampling new task in lifelong setting -> not between instances

        :return: None
        """
        if len(self.counter) > 0:  # Save previously learned model
            self.update_memory()

        self.R, self.T, self.counter = self.empty_memory_structure()

        self.prev_s = None
        self.prev_a = None

    def empty_memory_structure(self):
        """
        Empty memory structure:
        R[s][a] (list): list of collected rewards
        T[s][a][s'] (int): number of times the transition has been observed
        counter[s][a] (int): number of times the state action pair has been sampled
        :return: R, T, counter
        """
        return defaultdict(lambda: defaultdict(list)), \
               defaultdict(lambda: defaultdict(lambda: defaultdict(int))), \
               defaultdict(lambda: defaultdict(int))

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

    def update_memory(self):
        """
        Update the memory:
        1. Store the reward and transitions for the known state-action pairs in R_memory and T_memory
        2. Compute the final upper-bound and store it in U_memory
        :return: None
        """
        new_R = defaultdict(lambda: defaultdict(list))
        new_T = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        new_U = defaultdict(lambda: defaultdict(lambda: self.r_max / (1.0 - self.gamma)))

        # Store known rewards and transition
        for s in self.R:
            for a in self.R[s]:
                if self.is_known(s, a):
                    new_R[s][a] = self.R[s][a]
                    new_T[s][a] = self.T[s][a]

        # Compute and store upper-bounds
        for s in new_R:
            for a in new_R[s]:
                new_U[s][a] = self.compute_q_value(s, a)

        self.R_memory.append(new_R)
        self.T_memory.append(new_T)
        self.U_memory.append(new_U)

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
        if self.is_known(s, a):
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
        else:
            return self.r_max / (1. - self.gamma)

    def _get_reward(self, s, a):
        """
        Return the learned expected reward function at (s, a) if known.
        Else return the used heuristic.

        :param s: int state
        :param a: int action
        :return: return R(s, a)
        """
        if self.is_known(s, a):
            collected_rewards = self.R[s][a]
            return float(sum(collected_rewards)) / float(len(collected_rewards))
        else:
            return self.r_max
