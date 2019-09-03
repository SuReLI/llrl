import pandas as pd

from llrl.agents.rmax import RMax


class ExpRMax(RMax):
    """
    Copy of RMax for experiments:
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
            name="ExpRMax"
    ):
        RMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                      n_known=n_known, deduce_n_known=deduce_n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m,
                      delta=delta, n_states=n_states, name=name)

        self.cnt_time_steps = 0  # nb of time steps
        self.cnt_time_steps_cv = 0  # nb of time steps before convergence with high probability

        self.data = pd.DataFrame(columns=['cnt_time_steps', 'cnt_time_steps_cv'])

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        RMax.reset(self)

        self.write_data(self.cnt_time_steps, self.cnt_time_steps_cv)
        self.cnt_time_steps = 0
        self.cnt_time_steps_cv = 0

    def write_data(self, cnt_time_steps, cnt_time_steps_cv):
        self.data = self.data.append([cnt_time_steps, cnt_time_steps_cv])

    def act(self, s, r):
        """
        Acting method called online during learning.

        :param s: int current state of the agent
        :param r: float received reward for the previous transition
        :return: return the greedy action wrt the current learned model.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a = self.greedy_action(s, self.U)

        self.prev_a = a
        self.prev_s = s

        self.cnt_time_steps += 1  # INCREMENT TIME STEPS COUNTER

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