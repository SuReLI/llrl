from llrl.agents.lrmax import LRMax
import llrl.agents.maxqinit as mqi
from collections import defaultdict


class LRMaxQInit(LRMax):
    """
    Lipschitz R-Max agent.
    Leverage the pseudo-Lipschitz continuity of the optimal Q-value function in the MDP space to perform value transfer.
    Benefits also from the MaxQInit initialization [Abel et al 2018]
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
            deduce_n_known=True,
            max_memory_size=None,
            prior=None,
            estimate_distances_online=True,
            min_sampling_probability=.1,
            name="LRMaxQInit"
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
        name = name if prior is None else name + '(Dmax =' + str(prior) + ')'
        self.n_required_tasks = mqi.number_of_tasks_for_high_confidence_upper_bound(delta, min_sampling_probability)

        LRMax.__init__(self, actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                       n_known=n_known, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                       deduce_n_known=deduce_n_known, max_memory_size=max_memory_size, prior=prior,
                       estimate_distances_online=estimate_distances_online,
                       min_sampling_probability=min_sampling_probability, name=name)

    def initialize_upper_bound(self):
        """
        Initialization of the total upper-bound on the Q-value function.
        Called before applying the value iteration algorithm.
        :return: None
        """
        self.U = defaultdict(lambda: defaultdict(lambda: self.v_max))

        if len(self.U_memory) > self.n_required_tasks:
            for s in self.SA_memory:
                for a in self.SA_memory[s]:
                    self.U[s][a] = min(self.U[s][a], max([u[s][a] for u in self.U_memory]))

        for u_lip in self.U_lip:
            for s in u_lip:
                for a in u_lip[s]:
                    self.U[s][a] = min(self.U[s][a], u_lip[s][a])
