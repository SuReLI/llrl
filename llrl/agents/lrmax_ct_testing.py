from collections import defaultdict

from llrl.agents.lrmax_ct import LRMaxCT
from llrl.utils.utils import csv_write


class LRMaxCTTesting(LRMaxCT):
    """
    Copy of LRMaxCT agent with a few modifications used for experiments.
    - Record the number of use of the Lipschitz bound and the R-Max bound
    - Save this result at each call to the reset function
    - Record number of time steps to convergence
    """

    def __init__(self, actions, gamma=.9, count_threshold=1, epsilon=.1, delta_r=1., name="LRMaxCTTesting", path="output.csv"):
        LRMaxCT.__init__(
            self, actions, gamma=gamma, count_threshold=count_threshold, epsilon=epsilon, delta_r=delta_r, name=name
        )

        # Counters used for experiments (not useful to the algorithm)
        self.n_rmax = 0  # nb of times the rmax bound is used (smaller)
        self.n_lip = 0  # nb of times the lip bound is used (smaller)

        self.n_time_steps = 0  # nb of time steps
        self.n_time_steps_cv = 0  # nb of time steps before convergence with high probability

        self.path = path

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        LRMaxCT.reset(self)

        n_bound_use = self.n_rmax + self.n_lip
        if n_bound_use > 0:
            # Save ratio
            ratio_rmax_bound_use = self.n_rmax / n_bound_use
            ratio_lip_bound_use = self.n_lip / n_bound_use
            csv_write(
                [self.delta_r, ratio_rmax_bound_use, ratio_lip_bound_use, self.n_time_steps, self.n_time_steps_cv],
                self.path,
                'a'
            )

            # Reset
            self.n_rmax = 0
            self.n_lip = 0
            self.n_time_steps = 0
            self.n_time_steps_cv = 0

    def min_upper_bound(self, s):
        """
        Choose the minimum local upper-bound between the one provided by R-Max and
        those provided by each Lipschitz bound.
        :param s: input state for which the bound is derived
        :return: return the minimum upper-bound.
        """
        u_min = defaultdict(lambda: defaultdict(lambda: self.r_max / (1. - self.gamma)))
        for a in self.actions:
            u_min[s][a] = self.U[s][a]
            if len(self.U_lip) > 0:
                for u in self.U_lip:
                    if u[s][a] < self.U[s][a]:
                        self.n_lip += 1
                    else:
                        self.n_rmax += 1
                    if u[s][a] < u_min[s][a]:
                        u_min[s][a] = u[s][a]
        return u_min

    def act(self, s, r):
        """
        Acting method called online during learning.
        :param s: int current state of the agent
        :param r: float received reward for the previous transition
        :return: return the greedy action wrt the current learned model.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a = self.greedy_action(s, self.min_upper_bound(s))

        self.prev_a = a
        self.prev_s = s

        self.n_time_steps += 1  # INCREASE COUNTER BY 1

        return a

    def update(self, s, a, r, s_p):
        """
        Updates transition and reward dictionaries with the input transition
        tuple if the corresponding state-action pair is not known enough.
        If a new state-action pair is known, update both the R-Max upper-bound
        and the Lipschitz upper-bounds.
        :param s: state
        :param a: action
        :param r: reward
        :param s_p: next state
        :return: None
        """
        if s is not None and a is not None:
            if self.counter[s][a] < self.count_threshold:
                self.counter[s][a] += 1
                self.R[s][a] += [r]
                self.T[s][a][s_p] += 1
                if self.counter[s][a] == self.count_threshold:
                    self.n_time_steps_cv = self.n_time_steps  # RECORD LAST TIME A PAIR WAS UPDATED
                    self.update_rmax_upper_bound()
                    self.update_lipschitz_upper_bounds()
