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
    n_env = 5
    n_states = 3
    env_distribution = make_env_distribution(env_class='corridor', n_env=n_env, gamma=gamma, w=n_states, h=1)
    actions = env_distribution.get_actions()
    p_min = 1. / float(n_env)
    r_max = 1.
    v_max = 1.
    epsilon_q = .1
    epsilon_m = .1
    delta = .1
    max_mem = 2

    # Agents
    rmax = RMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, epsilon_q=epsilon_q,
                n_known=5, deduce_n_known=False)
    lrmax = LRMax(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                  n_states=n_states, max_memory_size=max_mem, prior=19.)
    maxqinit = MaxQInit(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                        n_states=n_states, min_sampling_probability=p_min)
    lrmaxqinit = LRMaxQInit(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                            n_states=n_states, max_memory_size=max_mem, prior=19.)
    agents_pool = [rmax]

    # Run
    run_agents_lifelong(agents_pool, env_distribution, name_identifier=None, n_instances=1, n_tasks=50, n_episodes=50,
                        n_steps=2, reset_at_terminal=False)


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
