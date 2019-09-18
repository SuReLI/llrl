from llrl.agents.rmax import RMax
from llrl.utils.save import csv_write
from llrl.utils.utils import avg_last_elts


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
            name="ExpRMax",
            path='results/'
    ):
        RMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                      n_known=n_known, deduce_n_known=deduce_n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m,
                      delta=delta, n_states=n_states, name=name)

        # Recorded variables
        self.discounted_return = 0.
        self.total_return = 0.
        self.n_time_steps = 0  # nb of time steps
        self.update_time_steps = []  # time steps where a model update occurred

        self.path = path
        self.instance_number = 0
        self.run_number = 0

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(actions=self.actions, gamma=self.gamma, r_max=self.r_max, v_max=self.v_max,
                      deduce_v_max=self.deduce_v_max, n_known=self.n_known, epsilon_q=self.epsilon_q,
                      epsilon_m=self.epsilon_m, delta=self.delta, n_states=self.n_states,
                      deduce_n_known=self.deduce_n_known, name=self.name, path=self.path)

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        RMax.reset(self)

        self.write(init=False)

        # Reset recorded variables between MDPs
        self.discounted_return = 0.
        self.total_return = 0.
        self.n_time_steps = 0
        self.update_time_steps = []

    def write(self, init=False):
        if init:
            col = [
                'instance_number',
                'run_number',
                'n_time_steps',
                'n_time_steps_cv',
                'avg_ts_l2',
                'avg_ts_l5',
                'avg_ts_l10',
                'avg_ts_l50',
                'discounted_return',
                'total_return'
            ]
            csv_write(col, self.path, 'w')
        else:
            assert self.n_time_steps is not None
            val = [
                self.instance_number,
                self.run_number,
                self.n_time_steps,
                self.update_time_steps[-1],
                avg_last_elts(self.update_time_steps, 2),
                avg_last_elts(self.update_time_steps, 5),
                avg_last_elts(self.update_time_steps, 10),
                avg_last_elts(self.update_time_steps, 50),
                self.discounted_return,
                self.total_return
            ]
            csv_write(val, self.path, 'a')

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

        self.discounted_return += (r * self.gamma ** float(self.n_time_steps))  # UPDATE
        self.total_return += r  # UPDATE
        self.n_time_steps += 1  # INCREMENT TIME STEPS COUNTER

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
                    self.update_time_steps.append(self.n_time_steps)  # RECORD
                    self.update_upper_bound()
