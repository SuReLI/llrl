from collections import defaultdict

from llrl.agents.lrmax import LRMax
from llrl.utils.save import csv_write
from llrl.utils.utils import avg_last_elts


class ExpLRMax(LRMax):
    """
    Copy of LRMax agent for experiments, listed below:
    - Record the number of use of the Lipschitz bound and the R-Max bound
    - Save this result at each call to the reset function
    - Record number of time steps to convergence
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
            name="ExpLRMax",
            path='results/'
    ):
        LRMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                       n_known=n_known, deduce_n_known=deduce_n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m,
                       delta=delta, n_states=n_states, max_memory_size=max_memory_size, prior=prior,
                       estimate_distances_online=estimate_distances_online,
                       min_sampling_probability=min_sampling_probability, name=name)

        # Counters used for experiments (not useful to the algorithm)
        self.n_rmax = 0  # number of times the rmax bound is used for all the updates of 1 task
        self.n_lip = 0  # number of times the lipschitz bound is used for all the updates of 1 task

        # Counter for prior use
        self.n_prior_use = 0  # number of times the prior is used for each update of 1 task
        self.n_dista_use = 0  # number of times the distance is used for each update of 1 task

        # Recorded variables
        self.discounted_return = 0.
        self.total_return = 0.
        self.n_time_steps = 0  # number of time steps

        self.path = path
        self.write_data = False  # Enable data writing
        self.instance_number = 0
        self.run_number = 0

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(actions=self.actions, gamma=self.gamma, r_max=self.r_max, v_max=self.v_max,
                      deduce_v_max=self.deduce_v_max, n_known=self.n_known, deduce_n_known=self.deduce_n_known,
                      epsilon_q=self.epsilon_q, epsilon_m=self.epsilon_m, delta=self.delta, n_states=self.n_states,
                      max_memory_size=self.max_memory_size, prior=self.prior,
                      estimate_distances_online=self.estimate_distances_online,
                      min_sampling_probability=self.min_sampling_probability, name=self.name, path=self.path)

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        LRMax.reset(self)

        '''
        n_bound_use = self.n_rmax + self.n_lip
        if n_bound_use > 0:
            if self.write_data:
                self.write(init=False)
        '''
        # Reset counters
        self.reset_counters()

        # Reset recorded variables between MDPs
        self.discounted_return = 0.
        self.total_return = 0.
        self.n_time_steps = 0

    def reset_counters(self):
        self.n_rmax = 0
        self.n_lip = 0
        self.n_prior_use = 0
        self.n_dista_use = 0

    def write(self, init=True):
        if init:
            col = [
                'instance_number',
                'run_number',
                'time_step',
                'prior',
                'n_prior_use',
                'n_dista_use',
                'n_rmax',
                'n_lip',
                'ratio_rmax_bound_use',
                'ratio_lip_bound_use',
                'n_time_steps',
                'n_time_steps_cv',
                'avg_ts_l10',
                'avg_ts_l50',
                'discounted_return',
                'total_return'
            ]
            csv_write(col, self.path, 'w')
        else:
            n_bound_use = self.n_rmax + self.n_lip
            ratio_rmax_bound_use = self.n_rmax / n_bound_use if n_bound_use > 0 else 0.
            ratio_lip_bound_use = self.n_lip / n_bound_use if n_bound_use > 0 else 0.
            val = [
                self.instance_number,
                self.run_number,
                self.n_time_steps,
                self.prior,
                self.n_prior_use,
                self.n_dista_use,
                self.n_rmax,
                self.n_lip,
                ratio_rmax_bound_use,
                ratio_lip_bound_use,
                self.n_time_steps,
                self.discounted_return,
                self.total_return
            ]
            csv_write(val, self.path, 'a')

    def initialize_upper_bound(self):
        """
        Initialization of the total upper-bound on the Q-value function.
        Called before applying the value iteration algorithm.
        :return: None
        """
        self.U = defaultdict(lambda: defaultdict(lambda: self.v_max))
        for u_lip in self.U_lip:
            for s in u_lip:
                for a in u_lip[s]:
                    self.U[s][a] = min(self.U[s][a], u_lip[s][a])
                    if self.v_max < u_lip[s][a]:  #
                        self.n_rmax += 1
                    else:
                        self.n_lip += 1

    def act(self, s, r):
        """
        Acting method called online during learning.
        :param s: int current state of the agent
        :param r: float received reward for the previous transition
        :return: return the greedy action wrt the current learned model.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a_star = self.greedy_action(s, self.U)

        self.prev_a = a_star
        self.prev_s = s

        self.discounted_return += (r * self.gamma ** float(self.n_time_steps))  # UPDATE
        self.total_return += r  # UPDATE
        self.n_time_steps += 1  # INCREMENT TIME STEPS COUNTER

        return a_star

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
            if self.counter[s][a] < self.n_known:
                self.counter[s][a] += 1
                normalizer = 1. / float(self.counter[s][a])

                self.R[s][a] = self.R[s][a] + normalizer * (r - self.R[s][a])
                self.T[s][a][s_p] = self.T[s][a][s_p] + normalizer * (1. - self.T[s][a][s_p])
                for _s_p in self.T[s][a]:
                    if _s_p not in [s_p]:
                        self.T[s][a][_s_p] = self.T[s][a][_s_p] * (1 - normalizer)

                if self.counter[s][a] == self.n_known:
                    self.update_upper_bound()

    def _set_distance(self, dsa):
        if dsa > self.prior:
            self.n_prior_use += 1
            return self.prior
        else:
            self.n_dista_use += 1
            return dsa

    def models_distances(self, u_mem, r_mem, t_mem, s_a_kk, s_a_ku, s_a_uk):
        """
        Compute the model's local pseudo-distances between the current MDP and the input memory unit.
        :param u_mem: (dictionary) upper-bound on the Q-value function of the previous MDP.
        :param r_mem: (dictionary) learned expected reward function of the previous MDP.
        :param t_mem: (dictionary) learned transition function of the previous MDP.
        :param s_a_kk: (list) state-actions pairs known in both MDPs
        :param s_a_ku: (list) state-actions pairs known in the current MDP - unknown in the previous MDP
        :param s_a_uk: (list) state-actions pairs unknown in the current MDP - known in the previous MDP
        :return: (dictionary) model's local distances
        """
        distances_cur = defaultdict(lambda: defaultdict(lambda: self.prior))  # distances computed wrt current MDP
        distances_mem = defaultdict(lambda: defaultdict(lambda: self.prior))  # distances computed wrt memory MDP

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

            dr = max(self.r_max - r_mem[s][a], r_mem[s][a])
            distances_cur[s][a] = min(dr + self.gamma * weighted_sum_wrt_cur + ma, self.prior)
            distances_mem[s][a] = min(dr + self.gamma * weighted_sum_wrt_mem + ma, self.prior)

        if self.write_data:
            self.write(init=False)
        self.reset_counters()

        return distances_cur, distances_mem
