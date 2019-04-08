import copy
from collections import defaultdict

from llrl.agents.lrmax import LRMax


class LRMaxCT(LRMax):
    """
    Lipschitz R-Max agent for constant transitions setting.

    Assume shared transition function across MDPs in Lifelong RL setting.
    """

    def __init__(
            self,
            actions,
            gamma=.9,
            count_threshold=1,
            epsilon=.1,
            max_memory_size=None,
            prior=1.,
            name="LRMax-CT"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param count_threshold: (int) count after which a state-action pair is considered known
        :param epsilon: (float) precision of value iteration algorithm
        :param max_memory_size: (int) maximum number of saved models (infinity if None)
        :param prior: (float) prior knowledge of maximum model's distance
        :param name: (str)
        """
        LRMax.__init__(
            self,
            actions=actions,
            gamma=gamma,
            count_threshold=count_threshold,
            epsilon=epsilon,
            max_memory_size=max_memory_size,
            prior=prior,
            name=name
        )

    def act(self, s, r):  # TODO remove - no override
        """
        Acting method called online during learning.
        :param s: int current state of the agent
        :param r: float received reward for the previous transition
        :return: return the greedy action wrt the current learned model.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a = self.greedy_action(s, self.min_upper_bound(s))

        # TEST
        # self.print_model()  # TODO remove
        self.test(s, a)  # TODO remove

        self.prev_a = a
        self.prev_s = s

        return a

    def update(self, s, a, r, s_p):  # TODO remove - no override
        """
        Updates transition and reward dictionaries with the input transition
        tuple if the corresponding state-action pair is not known enough.
        :param s: int state
        :param a: int action
        :param r: float reward
        :param s_p: int next state
        :return: None
        """
        print('## Call on update ({} {} {} {})'.format(str(s), a, r, str(s_p)))  # TODO remove
        if s is not None and a is not None:
            if self.counter[s][a] < self.count_threshold:
                print('## Updating ({} {} {} {})'.format(str(s), a, r, str(s_p)))  # TODO remove
                self.counter[s][a] += 1
                normalizer = 1. / float(self.counter[s][a])

                self.R[s][a] = self.R[s][a] + normalizer * (r - self.R[s][a])
                self.T[s][a][s_p] = self.T[s][a][s_p] + normalizer * (1. - self.T[s][a][s_p])
                for _s_p in self.T[s][a]:
                    if _s_p not in [s_p]:
                        self.T[s][a][_s_p] = self.T[s][a][_s_p] * (1 - normalizer)

                if self.counter[s][a] == self.count_threshold:
                    self.update_rmax_upper_bound()

    def test(self, s, a):  # TODO remove
        print(s, a)
        for _a in self.R[s]:
            print('    a = {}'.format(_a))
            print('        U_rmax(s,a) = {}'.format(self.U[s][_a]))
            if len(self.U_lip) > 0:
                print('        U_lip(s,a)  = {}'.format(self.U_lip[0][s][_a]))
            print('        R(s,a)      = {}'.format(self.R[s][_a]))
            for sp in self.T[s][_a]:
                print('        T({} | s,a) = {}'.format(str(sp), self.T[s][_a][sp]))

    def print_model(self):  # TODO remove
        print('    --- CURRENT MODEL ---')
        for s in self.R:
            for a in self.R[s]:
                print('    {} a: {:>10} counter: {:>10}'.format(str(s), a, self.counter[s][a]))
                print('        R(s,a) = {}'.format(self.R[s][a]))
                for sp in self.T[s][a]:
                    print('      T({}| s, a) = {}'.format(str(sp), self.T[s][a][sp]))
        print('    ---------------------')

    def compute_lipschitz_upper_bound(self, u_mem, r_mem, t_mem):
        ''' Note: different from LRMax '''
        # 1. Separate state-action pairs
        s_a_kk, s_a_ku, s_a_uk = self.separate_state_action_pairs(r_mem)

        # 2. Compute models distances upper-bounds
        distances_dict = self._models_distances(r_mem, s_a_kk, s_a_ku, s_a_uk)

        # 3. Compute the Q-values gap with dynamic programming
        gap = self._q_values_gap(distances_dict, t_mem, s_a_kk, s_a_ku, s_a_uk)

        # 4. Deduce upper-bound from u_mem
        return self.lipschitz_upper_bound(u_mem, gap)

    def models_distances(self, u_mem, r_mem, t_mem, s_a_kk, s_a_ku, s_a_uk):
        raise ValueError('Method models_distances not implemented in this class, see _models_distances method.')

    def _models_distances(self, r_mem, s_a_kk, s_a_ku, s_a_uk):
        # Initialize model's distances upper-bounds
        distances_dict = defaultdict(lambda: defaultdict(lambda: self.prior))

        # Compute model's distances upper-bounds for known-known (s, a)
        for s, a in s_a_kk:
            distances_dict[s][a] = abs(self.R[s][a] - r_mem[s][a])

        # Compute model's distances upper-bounds for known-unknown (s, a)
        for s, a in s_a_ku:
            distances_dict[s][a] = min(self.prior, max(self.r_max - self.R[s][a], self.R[s][a]))

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            distances_dict[s][a] = min(self.prior, max(self.r_max - r_mem[s][a], r_mem[s][a]))

        return distances_dict

    def q_values_gap(self, distances_dict, s_a_kk, s_a_ku, s_a_uk):
        raise ValueError('Method q_values_gap not implemented in this class, see _q_values_gap method.')

    def _q_values_gap(self, distances_dict, t_mem, s_a_kk, s_a_ku, s_a_uk):
        ''' Note: different from LRMax '''
        gap = defaultdict(lambda: defaultdict(lambda: self.prior / (1. - self.gamma)))

        for i in range(self.vi_n_iter):
            tmp = copy.deepcopy(gap)

            for s, a in s_a_uk:  # Unknown (s, a) in current MDP
                weighted_next_gap = 0.
                for s_p in t_mem[s][a]:
                    a_p = self.greedy_action(s_p, tmp)
                    weighted_next_gap += tmp[s_p][a_p] * t_mem[s][a][s_p]
                gap[s][a] = distances_dict[s][a] + self.gamma * weighted_next_gap

            for s, a in s_a_kk + s_a_ku:  # Known (s, a) in current MDP
                weighted_next_gap = 0.
                for s_p in self.T[s][a]:
                    a_p = self.greedy_action(s_p, tmp)
                    weighted_next_gap += tmp[s_p][a_p] * self.T[s][a][s_p]
                gap[s][a] = distances_dict[s][a] + self.gamma * weighted_next_gap

        return gap

    def lipschitz_upper_bound(self, u_mem, gap):
        ''' Note: different from LRMax '''
        u_lip = defaultdict(lambda: defaultdict(lambda: (self.prior + self.r_max) / (1. - self.gamma)))
        for s in gap:
            for a in gap[s]:
                u_lip[s][a] = u_mem[s][a] + gap[s][a]
        return u_lip
