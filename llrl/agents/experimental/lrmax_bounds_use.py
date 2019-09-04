import pandas as pd
from collections import defaultdict

from llrl.agents.lrmax import LRMax


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
            name="ExpLRMax"
    ):
        LRMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                       n_known=n_known, deduce_n_known=deduce_n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m,
                       delta=delta, n_states=n_states, max_memory_size=max_memory_size, prior=prior,
                       estimate_distances_online=estimate_distances_online,
                       min_sampling_probability=min_sampling_probability, name=name)

        # Counters used for experiments (not useful to the algorithm)
        self.cnt_rmax = 0  # number of times the rmax bound is used
        self.cnt_lip = 0  # number of times the lipschitz bound is used

        self.cnt_time_steps = 0  # number of time steps
        self.cnt_time_steps_cv = 0  # number of time steps before convergence with high probability

        self.write_data = True  # Enable data writing
        self.data = pd.DataFrame(columns=['prior', 'ratio_rmax_bound_use', 'ratio_lip_bound_use', 'cnt_time_steps',
                                          'cnt_time_steps_cv'])

    '''
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
                      min_sampling_probability=self.min_sampling_probability, path=self.path, name=self.name)
    '''

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        LRMax.reset(self)

        cnt_bound_use = self.cnt_rmax + self.cnt_lip
        if cnt_bound_use > 0:
            # Save ratio
            ratio_rmax_bound_use = self.cnt_rmax / cnt_bound_use
            ratio_lip_bound_use = self.cnt_lip / cnt_bound_use
            if self.write_data:
                self.write(ratio_rmax_bound_use, ratio_lip_bound_use)

            # Reset
            self.cnt_rmax = 0
            self.cnt_lip = 0
        self.cnt_time_steps = 0
        self.cnt_time_steps_cv = 0

    def write(self, ratio_rmax_bound_use, ratio_lip_bound_use):
        self.data = self.data.append({'prior': self.prior, 'ratio_rmax_bound_use': ratio_rmax_bound_use,
                                     'ratio_lip_bound_use': ratio_lip_bound_use, 'cnt_time_steps': self.cnt_time_steps,
                                      'cnt_time_steps_cv': self.cnt_time_steps_cv},
                                     ignore_index=True)

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
                        self.cnt_rmax += 1
                    else:
                        self.cnt_lip += 1

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

        self.cnt_time_steps += 1  # INCREMENT TIME STEPS COUNTER

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
                    self.cnt_time_steps_cv = self.cnt_time_steps  # RECORD LAST TIME A PAIR WAS UPDATED
                    self.update_upper_bound()
