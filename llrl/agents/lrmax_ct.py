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
            min_sampling_probability=0.1,
            delta=0.05,
            name="LRMax-CT"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param count_threshold: (int) count after which a state-action pair is considered known
        :param epsilon: (float) precision of value iteration algorithm
        :param max_memory_size: (int) maximum number of saved models (infinity if None)
        :param prior: (float) prior knowledge of maximum model's distance
        :param min_sampling_probability: (float) minimum sampling probability of an environment
        :param delta: (float) uncertainty degree on the maximum model's distance of a state-action pair
        :param name: (str)
        """
        name = name if prior is None else name + '-prior' + str(prior)
        LRMax.__init__(
            self,
            actions=actions,
            gamma=gamma,
            count_threshold=count_threshold,
            epsilon=epsilon,
            max_memory_size=max_memory_size,
            prior=prior,
            min_sampling_probability=min_sampling_probability,
            delta=delta,
            name=name
        )

    def compute_lipschitz_upper_bound(self, u_mem, r_mem, t_mem):
        """
        See parent class LRMax.
        """
        # 1. Separate state-action pairs
        s_a_kk, s_a_ku, s_a_uk = self.separate_state_action_pairs(r_mem)

        # 2. Compute models distances upper-bounds
        distances_dict = self._models_distances(r_mem, s_a_kk, s_a_ku, s_a_uk)
        if self.estimate_distances_online:
            distances_dict = self.integrate_distances_knowledge(distances_dict)

        # 3. Compute the Q-values gap with dynamic programming
        gap = self._env_local_dist(distances_dict, t_mem, s_a_kk, s_a_ku, s_a_uk)

        # 4. Deduce upper-bound from u_mem
        return self.lipschitz_upper_bound(u_mem, gap)

    def models_distances(self, u_mem, r_mem, t_mem, s_a_kk, s_a_ku, s_a_uk):
        raise ValueError('Method models_distances not implemented in this class, see _models_distances method.')

    def model_upper_bound(self, i, j, s, a):
        """
        See parent class LRMax.
        """
        return abs(self.R_memory[i][s][a] - self.R_memory[j][s][a])

    def _models_distances(self, r_mem, s_a_kk, s_a_ku, s_a_uk):
        """
        See parent class LRMax.
        """
        distances_dict = defaultdict(lambda: defaultdict(lambda: self.prior))

        # Compute model's distances upper-bounds for known-known (s, a)
        for s, a in s_a_kk:
            distances_dict[s][a] = min(self.prior, abs(self.R[s][a] - r_mem[s][a]))

        # Compute model's distances upper-bounds for known-unknown (s, a)
        for s, a in s_a_ku:
            distances_dict[s][a] = min(self.prior, max(self.r_max - self.R[s][a], self.R[s][a]))

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            distances_dict[s][a] = min(self.prior, max(self.r_max - r_mem[s][a], r_mem[s][a]))

        return distances_dict

    def env_local_dist(self, distances_cur, distances_mem, t_mem, s_a_kk, s_a_ku, s_a_uk):
        raise ValueError('Method q_values_gap not implemented in this class, see _q_values_gap method.')

    def _env_local_dist(self, distances_dict, t_mem, s_a_kk, s_a_ku, s_a_uk):
        """
        See parent class LRMax.
        """
        '''
        gap = defaultdict(lambda: defaultdict(lambda: self.prior / (1. - self.gamma)))

        for i in range(self.vi_n_iter):
            tmp = copy.deepcopy(gap)

            for s, a in s_a_uk:  # Unknown (s, a) in current MDP
                gap_p = 0.
                for s_p in t_mem[s][a]:
                    gap_p += max([tmp[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]
                gap[s][a] = distances_dict[s][a] + self.gamma * gap_p

            for s, a in s_a_kk + s_a_ku:  # Known (s, a) in current MDP
                gap_p = 0.
                for s_p in self.T[s][a]:
                    gap_p += max([tmp[s_p][a] for a in self.actions]) * self.T[s][a][s_p]
                gap[s][a] = distances_dict[s][a] + self.gamma * gap_p
        '''
        return LRMax.env_local_dist(self, distances_dict, distances_dict, t_mem, s_a_kk, s_a_ku, s_a_uk)
