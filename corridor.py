"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.agents.maxqinit import MaxQInit
from llrl.agents.lrmaxqinit import LRMaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


def experiment():
    gamma = .9
    n_env = 5
    n_states = 20
    env_distribution = make_env_distribution(env_class='corridor', n_env=n_env, gamma=gamma, w=n_states, h=1)
    actions = env_distribution.get_actions()
    p_min = 1. / float(n_env)
    epsilon_q = .1
    epsilon_m = 1.
    delta = .1
    max_mem = 2

    # Agents
    rmax = RMax(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states)
    lrmax = LRMax(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                  n_states=n_states, max_memory_size=max_mem, prior=19.)
    maxqinit = MaxQInit(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                        n_states=n_states, min_sampling_probability=p_min)
    lrmaxqinit = LRMaxQInit(actions=actions, gamma=gamma, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                            n_states=n_states, max_memory_size=max_mem, prior=19.)
    agents_pool = [rmax, maxqinit]

    # Run
    run_agents_lifelong(agents_pool, env_distribution, samples=20, episodes=20, steps=10, reset_at_terminal=False,
                        open_plot=True, cumulative_plot=False, is_tracked_value_discounted=False, plot_only=False,
                        plot_title=False)


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
