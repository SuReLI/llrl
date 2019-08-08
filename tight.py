"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.agents.maxqinit import MaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .9


def experiment():
    env_distribution = make_env_distribution(env_class='tight', env_name='tight', gamma=GAMMA)
    actions = env_distribution.get_actions()
    p_min = 1. / 7.  # There are seven possible MDPs
    epsilon = .001
    delta = .1

    m = 100
    max_mem = 10
    rmax = RMax(actions=actions, gamma=GAMMA, epsilon=epsilon, count_threshold=m)
    lrmax = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=1.)

    agents_pool = [rmax, lrmax]

    run_agents_lifelong(
        agents_pool, env_distribution, samples=20, episodes=100, steps=1000, reset_at_terminal=False,
        open_plot=True, cumulative_plot=False, is_tracked_value_discounted=True, plot_only=False, plot_title=False
    )


if __name__ == '__main__':
    np.random.seed(199311)
    experiment()
