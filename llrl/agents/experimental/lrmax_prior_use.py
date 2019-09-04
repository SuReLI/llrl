import numpy as np
from collections import defaultdict

from llrl.agents.lrmax import LRMax


class ExpLRMax(LRMax):
    """
    Lipschitz R-Max agent for experiments.
    """

    def __init__(
            self,
            actions,
            gamma=.9,
            r_max=1.,
            v_max=None,
            deduce_v_max=True,
            n_known=None,
            deduce_n_known=True,
            epsilon_q=0.1,
            epsilon_m=None,
            delta=None,
            n_states=None,
            max_memory_size=None,
            prior=None,
            estimate_distances_online=True,
            min_sampling_probability=.1,
            name="ExpLRMax"
    ):
        """
        See LRMax class.
        """
        LRMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                       n_known=n_known, deduce_n_known=deduce_n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m,
                       delta=delta, n_states=n_states, max_memory_size=max_memory_size, prior=prior,
                       estimate_distances_online=estimate_distances_online,
                       min_sampling_probability=min_sampling_probability, name=name)

        self.time_step = 0
        self.time_step_counter = []

        self.data = {'n_computation': [0], 'n_prior_use': [0]}

    def _set_distance(self, dsa):
        self.data['n_computation'][-1] += 1
        if dsa > self.prior:
            self.data['n_prior_use'][-1] += 1
            return self.prior
        else:
            return dsa

    def get_results(self):
        assert len(self.data['n_computation']) == len(self.data['n_prior_use'])
        result = []
        for i in range(len(self.data['n_computation']) - 1):
            recorded_time_step = self.time_step_counter[i]
            prior_use_ratio = round(100. * float(self.data['n_prior_use'][i]) / float(self.data['n_computation'][i]), 2)
            result.append([recorded_time_step, prior_use_ratio])
        return result

    def reset(self):
        self.time_step = 0
        self.time_step_counter = []
        LRMax.reset(self)

    def act(self, s, r):
        self.time_step += 1
        return LRMax.act(self, s, r)

    def models_distances(self, u_mem, r_mem, t_mem, s_a_kk, s_a_ku, s_a_uk):
        """
        See LRMax class.
        Overrides method of LRMax for prior use counter.
        """
        distances_cur = defaultdict(lambda: defaultdict(lambda: self.prior))  # distances computed wrt current MDP
        distances_mem = defaultdict(lambda: defaultdict(lambda: self.prior))  # distances computed wrt memory MDP

        if len(self.U_memory) > 1:  # NEW: no computation after the second environment
            return distances_cur, distances_mem

        # Compute model's distances upper-bounds for known-known (s, a)
        for s, a in s_a_kk:
            weighted_sum_wrt_cur = 0.
            weighted_sum_wrt_mem = 0.
            for s_p in self.T[s][a]:
                dt = abs(self.T[s][a][s_p] - t_mem[s][a][s_p])
                weighted_sum_wrt_cur += max([self.U[s_p][a] for a in self.actions]) * dt
                weighted_sum_wrt_mem += max([u_mem[s_p][a] for a in self.actions]) * dt
            for s_p in t_mem[s][a]:
                if s_p not in self.T[s][a]:
                    weighted_sum_wrt_cur += max([self.U[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]
                    weighted_sum_wrt_mem += max([u_mem[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]

            dr = abs(self.R[s][a] - r_mem[s][a])
            # distances_cur[s][a] = min(dr + self.gamma * weighted_sum_wrt_cur + 2. * self.b, self.prior)  # PREV
            # distances_mem[s][a] = min(dr + self.gamma * weighted_sum_wrt_mem + 2. * self.b, self.prior)  # PREV
            distances_cur[s][a] = self._set_distance(dr + self.gamma * weighted_sum_wrt_cur + 2. * self.b)  # NEW
            distances_mem[s][a] = self._set_distance(dr + self.gamma * weighted_sum_wrt_mem + 2. * self.b)  # NEW

        ma = self.gamma * self.v_max + self.b

        # Compute model's distances upper-bounds for known-unknown (s, a)
        for s, a in s_a_ku:
            weighted_sum_wrt_cur = 0.
            weighted_sum_wrt_mem = 0.
            for s_p in self.T[s][a]:
                weighted_sum_wrt_cur += max([self.U[s_p][a] for a in self.actions]) * self.T[s][a][s_p]
                weighted_sum_wrt_mem += max([u_mem[s_p][a] for a in self.actions]) * self.T[s][a][s_p]

            dr = max(self.r_max - self.R[s][a], self.R[s][a])
            # distances_cur[s][a] = min(dr + self.gamma * weighted_sum_wrt_cur + ma, self.prior)  # PREV
            # distances_mem[s][a] = min(dr + self.gamma * weighted_sum_wrt_mem + ma, self.prior)  # PREV
            distances_cur[s][a] = self._set_distance(dr + self.gamma * weighted_sum_wrt_cur + ma)  # NEW
            distances_mem[s][a] = self._set_distance(dr + self.gamma * weighted_sum_wrt_mem + ma)  # NEW

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            weighted_sum_wrt_cur = 0.
            weighted_sum_wrt_mem = 0.
            for s_p in t_mem[s][a]:
                weighted_sum_wrt_cur += max([self.U[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]
                weighted_sum_wrt_mem += max([u_mem[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]

            dr = max(self.r_max - r_mem[s][a], r_mem[s][a])
            # distances_cur[s][a] = min(dr + self.gamma * weighted_sum_wrt_cur + ma, self.prior)  # PREV
            # distances_mem[s][a] = min(dr + self.gamma * weighted_sum_wrt_mem + ma, self.prior)  # PREV
            distances_cur[s][a] = self._set_distance(dr + self.gamma * weighted_sum_wrt_cur + ma)  # NEW
            distances_mem[s][a] = self._set_distance(dr + self.gamma * weighted_sum_wrt_mem + ma)  # NEW

        self.time_step_counter.append(self.time_step)  # NEW: record the number of time steps
        self.data['n_computation'].append(0)  # NEW: add a new counter for the next computation
        self.data['n_prior_use'].append(0)  # NEW: idem

        return distances_cur, distances_mem
