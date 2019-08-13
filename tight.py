"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.agents.maxqinit import MaxQInit
from llrl.agents.lrmaxqinit import LRMaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments import run_agents_lifelong


def experiment():
    # Parameters
    gamma = .9
    env_distribution = make_env_distribution(env_class='tight', env_name='tight', gamma=gamma)
    actions = env_distribution.get_actions()
    p_min = 1. / 7.  # There are seven possible MDPs
    epsilon_q = .1
    epsilon_m = .01
    delta = .1
    n_states = 4
    max_mem = 10

    # Agents
    rmax = RMax(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states)
    lrmax = LRMax(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                  n_states=n_states, max_memory_size=max_mem, prior=19.)
    maxqinit = MaxQInit(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                        n_states=n_states, min_sampling_probability=p_min)
    lrmaxqinit = LRMaxQInit(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                            n_states=n_states, max_memory_size=max_mem, prior=19.)
    agents_pool = [rmax]

    # Run
    run_agents_lifelong(agents_pool, env_distribution, n_instances=1, n_tasks=50, n_episodes=50, n_steps=2,
                        reset_at_terminal=False)


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
