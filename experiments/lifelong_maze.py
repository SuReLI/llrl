"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .9


def experiment():
    env_distribution = make_env_distribution(env_class='maze', n_env=10, gamma=GAMMA, w=5, h=5)
    actions = env_distribution.get_actions()

    m = 100
    max_mem = 4
    rmax = RMax(actions=actions, gamma=GAMMA, count_threshold=m)
    lrmax1 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=1.)
    lrmax030 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.30)
    lrmax020 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.20)
    lrmax018 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.18)

    # agents_pool = [rmax, lrmax1, lrmax030, lrmax018]
    agents_pool = [rmax, lrmax020]  # TODO remove

    run_agents_lifelong(
        agents_pool, env_distribution, samples=50, episodes=50, steps=1000, reset_at_terminal=False,
        open_plot=True, cumulative_plot=False, is_tracked_value_discounted=True, plot_only=False
    )


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
