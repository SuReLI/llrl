from copy import deepcopy
from collections import defaultdict
from itertools import permutations

from llrl.agents.rmax import RMax


def probability_of_success(n_samples, p_min):
    """
    Compute a lower bound on the probability of successful distance estimation.
    :param n_samples: (int) number of samples
    :param p_min: (float) minimum sampling probability of an environment
    :return: (float) the probability of successful estimation in [0, 1]
    """
    return 1. - 2. * (1. - p_min) ** float(n_samples) + (1. - 2. * p_min) ** float(n_samples)


def compute_n_samples_high_confidence(p_min, delta):
    """
    Compute the number of samples required for an accurate estimate
    of the model pseudo-distance with high probability.
    :param p_min: (float) minimum sampling probability
    :param delta: (float) uncertainty degree on the maximum model's distance of a state-action pair
    :return: (int) the number of samples
    """
    hc = 1. - delta
    n_max = int(1e6)
    for i in range(n_max):
        if probability_of_success(i, p_min) >= hc:
            return i
    raise ValueError(
        'Could not compute the required number of samples for accurate estimates with high probability. ' +
        'Reached end of loop for {} samples.'.format(n_max)
    )


class LRMax(RMax):
    """
    Lipschitz R-Max agent.
    Leverage the pseudo-Lipschitz continuity of the optimal Q-value function in the MDP space to perform value transfer.
    """

    def __init__(
            self,
            actions,
            gamma=.9,
            r_max=1.,
            v_max=None,
            deduce_v_max=True,
            n_known=None,
            epsilon_q=0.1,
            epsilon_m=None,
            delta=None,
            n_states=None,
            deduce_n_known=True,  #
            max_memory_size=None,
            prior=None,
            estimate_distances_online=True,
            min_sampling_probability=.1,
            name="LRMax"
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

        :param max_memory_size: (int) maximum number of saved models (infinity if None)
        :param prior: (float) prior knowledge of maximum model's distance
        :param estimate_distances_online: (bool) set to True for online estimation of a tighter upper-bound for the
        model pseudo-distances. The estimation is valid with high probability.
        :param min_sampling_probability: (float) minimum sampling probability of an environment
        :param name: (str)
        """
        name = name if prior is None else name + '-prior' + str(prior)
        RMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                      n_known=n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                      deduce_n_known=deduce_n_known, name=name)

        # Lifelong Learning memories
        self.max_memory_size = max_memory_size
        self.U_memory = []
        self.R_memory = []
        self.T_memory = []
        self.SA_memory = defaultdict(lambda: defaultdict(lambda: False))

        self.U_lip = []
        self.b = self.epsilon_m * (1. + self.gamma * self.v_max)

        # Prior knowledge on maximum model distance
        prior_max = (1. + gamma) / (1. - gamma)
        self.prior = prior_max if prior is None else min(prior, prior_max)

        # Online distances estimation
        self.estimate_distances_online = estimate_distances_online
        self.min_sampling_probability = min_sampling_probability
        self.D = defaultdict(lambda: defaultdict(lambda: prior_max))  # Dictionary of distances (high probability)
        self.n_samples_high_confidence = compute_n_samples_high_confidence(min_sampling_probability, delta)

        self.update_upper_bound()

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(actions=self.actions, gamma=self.gamma, r_max=self.r_max, v_max=self.v_max,
                      deduce_v_max=self.deduce_v_max, n_known=self.n_known, epsilon_q=self.epsilon_q,
                      epsilon_m=self.epsilon_m, delta=self.delta, n_states=self.n_states,
                      deduce_n_known=self.deduce_n_known, max_memory_size=self.max_memory_size, prior=self.prior,
                      estimate_distances_online=self.estimate_distances_online,
                      min_sampling_probability=self.min_sampling_probability, name=self.name)

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        # Save previously learned model
        if len(self.counter) > 0 and (self.max_memory_size is None or len(self.U_lip) < self.max_memory_size):
            self.update_memory()

        RMax.reset(self)

        if self.estimate_distances_online:
            self.update_max_distances()
        self.update_upper_bound()

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

        return a_star

    def update_memory(self):
        """
        Update the memory (called between each MDP change i.e. when the reset method is called).
        Store the rewards, transitions and upper-bounds for the known state-action pairs
        respectively in R_memory, T_memory and U_memory.
        All the data corresponding to partially known state-action pairs are discarded.
        Consequently, the saved state-action pairs only refer to known pairs.
        :return: None
        """
        for s in self.R:
            for a in self.R[s]:
                if self.is_known(s, a):
                    self.SA_memory[s][a] = True

        self.U_memory.append(deepcopy(self.U))
        self.R_memory.append(deepcopy(self.R))
        self.T_memory.append(deepcopy(self.T))

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

    def update_upper_bound(self):
        """
        Update the total upper bound on the Q-value function.
        Called at initialization and when a new state-action pair is known.
        :return: None
        """
        self.update_lipschitz_upper_bounds()
        self.initialize_upper_bound()
        RMax.update_upper_bound(self)

    def update_lipschitz_upper_bounds(self):
        """
        Update the Lipschitz upper-bound for each instance of the memory.
        Called at initialization and when a new state-action pair is known.
        :return: None
        """
        n_prev_tasks = len(self.U_memory)
        if n_prev_tasks > 0:
            self.U_lip = []
            for i in range(n_prev_tasks):
                self.U_lip.append(self.compute_lipschitz_upper_bound(self.U_memory[i], self.R_memory[i],
                                                                     self.T_memory[i]))

    def model_upper_bound(self, i, j, s, a):
        """
        Compute the distance between memory models at (s, a)
        :param i: (int) index of the first model, whose Q-value upper-bound is used
        :param j: (int) index of the second model
        :param s: state
        :param a: action
        :return: Return the distance
        """
        dt = 0.
        for s_p in self.T_memory[i][s][a]:
            v_p = max([self.U_memory[i][s_p][a_p] for a_p in self.actions])
            dt += v_p * abs(self.T_memory[i][s][a][s_p] - self.T_memory[j][s][a][s_p])
        for s_p in self.T_memory[j][s][a]:
            if s_p not in self.T_memory[i][s][a]:
                v_p = max([self.U_memory[i][s_p][a_p] for a_p in self.actions])
                dt += v_p * self.T_memory[j][s][a][s_p]
        return abs(self.R_memory[i][s][a] - self.R_memory[j][s][a]) + self.gamma * dt

    def update_max_distances(self):
        """
        Update the maximum model's distance for each state-action pair.
        Called after each interaction with an environment.
        :return: None
        """
        n_prev_tasks = len(self.U_memory)
        if n_prev_tasks >= self.n_samples_high_confidence:
            for s in self.SA_memory:
                for a in self.SA_memory[s]:
                    indices = []  # indices of the tasks where (s, a) is known
                    for i in range(n_prev_tasks):
                        if s in self.R_memory[i] and a in self.R_memory[i][s]:  # s, a is known in ith
                            indices.append(i)
                    if len(indices) >= self.n_samples_high_confidence:
                        distances = []
                        for p in permutations(indices, 2):
                            distances.append(self.model_upper_bound(p[0], p[1], s, a))
                        self.D[s][a] = max(distances)

    def integrate_distances_knowledge(self, distances):
        """
        Integrate the knowledge on the learned distances to the distance estimates.
        :param distances: (dictionary) distance estimates
        :return: (dictionary) distance estimates with integrated knowledge
        """
        for s in distances:
            for a in distances[s]:
                distances[s][a] = min(distances[s][a], self.D[s][a])
        for s in self.D:
            for a in self.D[s]:
                distances[s][a] = min(distances[s][a], self.D[s][a])
        return distances

    def compute_lipschitz_upper_bound(self, u_mem, r_mem, t_mem):
        """
        Compute the Lipschitz upper-bound from a previous MDP given an upper-bound on its Q-value function and
        a (possibly partial) model of its reward and transition functions.
        :param u_mem: (dictionary) upper-bound on the Q-value function of the previous MDP.
        :param r_mem: (dictionary) learned expected reward function of the previous MDP.
        :param t_mem: (dictionary) learned transition function of the previous MDP.
        :return: (dictionary) Lipschitz upper-bound
        """
        # 1. Separate state-action pairs
        s_a_kk, s_a_ku, s_a_uk = self.separate_state_action_pairs(r_mem)

        # 2. Compute models distances upper-bounds
        distances_cur, distances_mem = self.models_distances(u_mem, r_mem, t_mem, s_a_kk, s_a_ku, s_a_uk)
        if self.estimate_distances_online:
            distances_cur = self.integrate_distances_knowledge(distances_cur)
            distances_mem = self.integrate_distances_knowledge(distances_mem)

        # 3. Compute the Q-values distances with dynamic programming
        d = self.env_local_dist(distances_cur, distances_mem, t_mem, s_a_kk, s_a_ku, s_a_uk)

        # 4. Deduce upper-bound from u_mem
        return self.lipschitz_upper_bound(u_mem, d)

    def separate_state_action_pairs(self, r_mem):
        """
        Create 3 lists of state-action pairs corresponding to:
        - pairs known in the current MDP and the considered previous one;
        - known only in the current MDP;
        - known only in the previous MDP.
        :param r_mem: Reward memory of the previous MDP
        :return: the 3 lists as a tuple
        """
        # Define different state-action pairs container:
        s_a_kk = []  # Known in both MDPs
        s_a_ku = []  # Known in current MDP - Unknown in previous MDP
        s_a_uk = []  # Unknown in current MDP - Known in previous MDP

        # Fill containers
        for s in self.R:
            for a in self.actions:
                if self.is_known(s, a):
                    if s in r_mem and a in r_mem[s]:  # (s, a) known for both MDPs
                        s_a_kk.append((s, a))
                    else:  # (s, a) only known in current MDP
                        s_a_ku.append((s, a))
        for s in r_mem:
            for a in r_mem[s]:
                if not self.is_known(s, a):  # (s, a) only known in previous MDP
                    s_a_uk.append((s, a))

        return s_a_kk, s_a_ku, s_a_uk

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
            distances_cur[s][a] = min(dr + self.gamma * weighted_sum_wrt_cur + 2. * self.b, self.prior)
            distances_mem[s][a] = min(dr + self.gamma * weighted_sum_wrt_mem + 2. * self.b, self.prior)

        ma = self.gamma * self.v_max + self.b

        # Compute model's distances upper-bounds for known-unknown (s, a)
        for s, a in s_a_ku:
            weighted_sum_wrt_cur = 0.
            weighted_sum_wrt_mem = 0.
            for s_p in self.T[s][a]:
                weighted_sum_wrt_cur += max([self.U[s_p][a] for a in self.actions]) * self.T[s][a][s_p]
                weighted_sum_wrt_mem += max([u_mem[s_p][a] for a in self.actions]) * self.T[s][a][s_p]

            dr = max(self.r_max - self.R[s][a], self.R[s][a])
            distances_cur[s][a] = min(dr + self.gamma * weighted_sum_wrt_cur + ma, self.prior)
            distances_mem[s][a] = min(dr + self.gamma * weighted_sum_wrt_mem + ma, self.prior)

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            weighted_sum_wrt_cur = 0.
            weighted_sum_wrt_mem = 0.
            for s_p in t_mem[s][a]:
                weighted_sum_wrt_cur += max([self.U[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]
                weighted_sum_wrt_mem += max([u_mem[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]

            dr = max(self.r_max - r_mem[s][a], r_mem[s][a])
            distances_cur[s][a] = min(dr + self.gamma * weighted_sum_wrt_cur + ma, self.prior)
            distances_mem[s][a] = min(dr + self.gamma * weighted_sum_wrt_mem + ma, self.prior)

        return distances_cur, distances_mem

    def env_local_dist(self, distances_cur, distances_mem, t_mem, s_a_kk, s_a_ku, s_a_uk):
        """
        Compute the environment's local distances based on model's distances between two MDPs.
        :param distances_cur: distances dictionary computed wrt current MDP
        :param distances_mem: distances dictionary computed wrt memory MDP
        :param t_mem: (dictionary) learned transition function of the previous MDP.
        :param s_a_kk: (list) state-actions pairs known in both MDPs
        :param s_a_ku: (list) state-actions pairs known in the current MDP - unknown in the previous MDP
        :param s_a_uk: (list) state-actions pairs unknown in the current MDP - known in the previous MDP
        :return: (dictionary) computed Q-values distances
        """
        d_max = self.prior / (1. - self.gamma)
        gamma_d_max = self.gamma * d_max
        d_mem = defaultdict(lambda: defaultdict(lambda: d_max))
        d_cur = defaultdict(lambda: defaultdict(lambda: d_max))

        for s, a in s_a_uk:  # Unknown (s, a) in current MDP
            d_mem[s][a] = distances_mem[s][a] + gamma_d_max
        for i in range(self.vi_n_iter):
            for s, a in s_a_kk + s_a_ku:  # Known (s, a) in current MDP
                d_p = 0.
                for s_p in self.T[s][a]:
                    d_p += max([d_mem[s_p][a] for a in self.actions]) * self.T[s][a][s_p]
                d_mem[s][a] = distances_mem[s][a] + self.gamma * d_p + self.epsilon_m * gamma_d_max

        for s, a in s_a_ku:  # Unknown (s, a) in memory MDP
            d_cur[s][a] = distances_cur[s][a] + gamma_d_max
        for i in range(self.vi_n_iter):
            for s, a in s_a_kk + s_a_uk:  # Known (s, a) in memory MDP
                d_p = 0.
                for s_p in t_mem[s][a]:
                    d_p += max([d_cur[s_p][a] for a in self.actions]) * t_mem[s][a][s_p]
                d_cur[s][a] = distances_mem[s][a] + self.gamma * d_p + self.epsilon_m * gamma_d_max

        d = defaultdict(lambda: defaultdict(lambda: d_max))
        for s in d_mem:
            for a in d_mem[s]:
                d[s][a] = min(d_mem[s][a], d_cur[s][a])

        return d

    def lipschitz_upper_bound(self, u_mem, d):
        """
        Compute the Lipschitz upper-bound based on:
        1) The upper-bound on the Q-value of the previous MDP;
        2) The computed Q-values distances based on model's distances between the two MDPs.
        :param u_mem: (dictionary) upper-bound on the Q-value of the previous MDP
        :param d: (dictionary) computed Q-values distances
        :return: (dictionary) Lipschitz upper-bound
        """
        u_lip = defaultdict(lambda: defaultdict(lambda: (self.prior + self.r_max) / (1. - self.gamma)))
        for s in d:
            for a in d[s]:
                u_lip[s][a] = u_mem[s][a] + d[s][a]
        return u_lip
