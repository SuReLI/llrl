import random
from collections import defaultdict

from simple_rl.agents.AgentClass import Agent


class LRMax(Agent):
    """
    Lipschitz R-Max agent
    """

    def __init__(self, actions, gamma=0.9, count_threshold=1, name="Lipschitz-RMax"):
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.nA = len(self.actions)
        self.r_max = 1.0
        self.count_threshold = count_threshold

        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

        # Lifelong Learning memories
        self.U_memory = []
        self.R_memory = []
        self.T_memory = []

    def reset(self):
        """
        Reset the attributes to initial state.
        Save the previous model.

        TODO check whether reset is only applied when sampling new task in lifelong setting -> not between instances

        :return: None
        """
        if len(self.counter) > 0:  # Save previously learned model
            self.update_memory()

        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

    def empty_memory_structure(self):
        """
        Empty memory structure:
        U[s][a] (float): upper-bound on the Q-value
        R[s][a] (list): list of collected rewards
        T[s][a][s'] (int): number of times the transition has been observed
        counter[s][a] (int): number of times the state action pair has been sampled
        :return: R, T, counter
        """
        return defaultdict(lambda: defaultdict(lambda: self.r_max / (1.0 - self.gamma))), \
               defaultdict(lambda: defaultdict(list)), \
               defaultdict(lambda: defaultdict(lambda: defaultdict(int))), \
               defaultdict(lambda: defaultdict(int))

    def display(self):
        """
        Display info about the attributes.
        """
        print('Displaying R-MAX agent :')
        print('Action space           :', self.actions)
        print('Number of actions      :', self.nA)
        print('Gamma                  :', self.gamma)
        print('Count threshold        :', self.count_threshold)

    def is_known(self, s, a):
        return self.counter[s][a] >= self.count_threshold

    def get_nb_known_sa(self):
        return sum([self.is_known(s, a) for s, a in self.counter.keys()])

    def act(self, s, r):
        """
        Acting method called online during learning.

        :param s: int current state of the agent
        :param r: float received reward for the previous transition
        :return: return the greedy action wrt the current learned model.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a = self.greedy_action(s)

        self.prev_a = a
        self.prev_s = s

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
            if self.counter[s][a] < self.count_threshold:
                self.counter[s][a] += 1
                self.R[s][a] += [r]
                self.T[s][a][s_p] += 1
                if self.counter[s][a] == self.count_threshold:
                    self.update_upper_bound()

    def update_memory(self):
        """
        Update the memory:
        1. Store the reward and transitions for the known state-action pairs in R_memory and T_memory
        2. Compute the final upper-bound and store it in U_memory
        :return: None
        """
        new_R = defaultdict(lambda: defaultdict(list))
        new_T = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        new_U = defaultdict(lambda: defaultdict(lambda: self.r_max / (1.0 - self.gamma)))

        # Store known rewards and transition
        for s in self.R:
            for a in self.R[s]:
                if self.is_known(s, a):
                    new_R[s][a] = self.R[s][a]
                    new_T[s][a] = self.T[s][a]

        # Compute and store upper-bounds
        for s in new_R:
            for a in new_R[s]:
                new_U[s][a] = self.compute_q_value(s, a)

        self.R_memory.append(new_R)
        self.T_memory.append(new_T)
        self.U_memory.append(new_U)

    def compute_min_upper_bound(self, s, horizon=None):
        #TODO
        return 0  # TODO remove

    def greedy_action(self, s):
        """
        Compute the greedy action wrt the current upper bound.
        :param s: state
        :return: return the greedy action.
        """
        a_star = random.choice(self.actions)
        u_star = self.U[s][a_star]
        for a in self.actions:
            u_s_a = self.U[s][a]
            if u_s_a > u_star:
                u_star = u_s_a
                a_star = a
        return a_star

    def update_upper_bound(self, epsilon=0.1):
        """
        Update the upper bound on the Q-value function.
        Called when a new state-action pair is known.
        :param epsilon: maximum gap between the estimated Q-value and the optimal one.
        :return: None
        """
        n_iter = int(np.log(1. / (epsilon * (1. - self.gamma))) / (1. - self.gamma))
        for i in range(n_iter):
            for s in self.R:
                for a in self.R[s]:
                    n_s_a = float(self.counter[s][a])
                    r_s_a = sum(self.R[s][a]) / n_s_a

                    s_p_dict = self.T[s][a]
                    weighted_next_upper_bound = 0.
                    for s_p in s_p_dict:
                        weighted_next_upper_bound += self.U[s_p][self.greedy_action(s_p)] * s_p_dict[s_p] / n_s_a

                    self.U[s][a] = r_s_a + self.gamma * weighted_next_upper_bound
