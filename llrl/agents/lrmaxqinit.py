from llrl.agents.lrmax import LRMax


class LRMaxQInit(LRMax):
    """
    Lipschitz R-Max agent.
    Leverage the pseudo-Lipschitz continuity of the optimal Q-value function in the MDP space to perform value transfer.
    Benefits also from the MaxQInit initialization [Abel et al 2018]
    """

    def __init__(
            self,
            actions,
            gamma=0.9,
            count_threshold=None,
            epsilon_q=0.1,
            epsilon_m=None,
            delta=None,
            n_states=None,
            v_max=None,
            max_memory_size=None,
            prior=None,
            min_sampling_probability=0.1,
            name="LRMaxQInit"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param count_threshold: (int) count after which a state-action pair is considered known
        (only set count_threshold if delta and epsilon are not defined)
        :param epsilon_q: (float) precision of value iteration algorithm for Q-value computation
        :param epsilon_m: (float) precision of the learned models in L1 norm
        :param delta: (float) models are learned epsilon_m-closely with probability at least 1 - delta
        :param n_states: (int) number of states
        :param v_max: (float) known upper-bound on the value function
        :param max_memory_size: (int) maximum number of saved models (infinity if None)
        :param prior: (float) prior knowledge of maximum model's distance
        :param min_sampling_probability: (float) minimum sampling probability of an environment
        :param name: (str)
        """
        name = name if prior is None else name + '-prior' + str(prior)
        LRMax.__init__(
            self,
            actions=actions,
            gamma=gamma,
            count_threshold=count_threshold,
            epsilon_q=epsilon_q,
            epsilon_m=epsilon_m,
            delta=delta,
            n_states=n_states,
            v_max=v_max,
            max_memory_size=max_memory_size,
            prior=prior,
            min_sampling_probability=min_sampling_probability,
            name=name
        )
