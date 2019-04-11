import numpy as np
from collections import defaultdict

from llrl.agents.lrmax import LRMax


class LRMaxExp(LRMax):
    """
    Lipschitz R-Max agent for experiment.
    """

    def __init__(
            self,
            actions,
            gamma=0.9,
            count_threshold=1,
            epsilon=0.1,
            max_memory_size=None,
            prior=np.Inf,
            name="LRMax-prior"
    ):
        """
        See LRMax class.
        """
        name = name + str(prior) if name[-6:] == "-prior" else name
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

        self.time_step = 0
        self.time_step_counter = []
        self.prior_use_counter = [
            [0, 0]  # ['n_computation', 'n_prior_use']
        ]

    def _update_counters(self, is_prior_used):
        self.prior_use_counter[-1][0] += 1
        if is_prior_used:
            self.prior_use_counter[-1][1] += 1

    def _set_distance(self, dsa):
        if dsa > self.prior:
            self._update_counters(True)
            return self.prior
        else:
            self._update_counters(False)
            return dsa

    def get_results(self):
        result = []
        for i in range(len(self.prior_use_counter) - 1):
            recorded_time_step = self.time_step_counter[i]
            prior_use_ratio = round(100. * float(self.prior_use_counter[i][1]) / float(self.prior_use_counter[i][0]), 2)
            result.append([recorded_time_step, prior_use_ratio])
        return result

    def reset(self):
        self.time_step = 0
        self.time_step_counter = []
        LRMax.reset(self)

    def act(self, s, r):
        self.time_step += 1
        LRMax.act(self, s, r)

    def models_distances(self, u_mem, r_mem, t_mem, s_a_kk, s_a_ku, s_a_uk):
        """
        See LRMax class.
        Overrides method of LRMax for prior use counter.
        """

        distances_dict = defaultdict(lambda: defaultdict(lambda: self.prior))

        if len(self.U_memory) > 1:  # No computation after the second environment
            return distances_dict

        # Compute model's distances upper-bounds for known-known (s, a)
        for s, a in s_a_kk:
            weighted_sum = 0.
            for s_p in self.T[s][a]:
                weighted_sum += u_mem[s_p][self.greedy_action(s_p, u_mem)] * abs(self.T[s][a][s_p] - t_mem[s][a][s_p])
            for s_p in t_mem[s][a]:
                if s_p not in self.T[s][a]:
                    weighted_sum += u_mem[s_p][self.greedy_action(s_p, u_mem)] * t_mem[s][a][s_p]

            distances_dict[s][a] = min(
                abs(self.R[s][a] - r_mem[s][a]) + self.gamma * weighted_sum,
                self.prior
            )

            dsa = abs(self.R[s][a] - r_mem[s][a]) + self.gamma * weighted_sum
            distances_dict[s][a] = self._set_distance(dsa)

        # Compute model's distances upper-bounds for known-unknown (s, a)
        for s, a in s_a_ku:
            weighted_sum = 0.
            for s_p in self.T[s][a]:
                weighted_sum += u_mem[s_p][self.greedy_action(s_p, u_mem)] * self.T[s][a][s_p]

            dsa = max(self.r_max - self.R[s][a], self.R[s][a]) + \
                  self.gamma * weighted_sum + \
                  self.gamma * self.r_max / (1. - self.gamma)
            distances_dict[s][a] = self._set_distance(dsa)

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            weighted_sum = 0.
            for s_p in t_mem[s][a]:
                weighted_sum += u_mem[s_p][self.greedy_action(s_p, u_mem)] * t_mem[s][a][s_p]

            dsa = max(self.r_max - r_mem[s][a], r_mem[s][a]) + \
                  self.gamma * weighted_sum + \
                  self.gamma * self.r_max / (1. - self.gamma)
            distances_dict[s][a] = self._set_distance(dsa)

        self.time_step_counter.append(self.time_step)
        self.prior_use_counter.append([0, 0])  # Add a new counter for the next computation

        return distances_dict
