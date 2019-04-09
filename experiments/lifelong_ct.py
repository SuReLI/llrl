"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax_ct import LRMaxCT
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .99


def experiment():
    # Create environments distribution
    # env_distribution = make_env_distribution(env_class='grid-world', n_env=10, gamma=GAMMA, w=3, h=3)
    env_distribution = make_env_distribution(env_class='corridor', n_env=10, gamma=GAMMA, w=50, h=1)
    actions = env_distribution.get_actions()

    m = 1
    max_mem = 3
    rmax = RMax(actions=actions, gamma=GAMMA, count_threshold=m)
    lrmax0_2 = LRMaxCT(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.2)
    lrmax0_7 = LRMaxCT(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.7)
    lrmax1_0 = LRMaxCT(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=1.0)

    agents_pool = [rmax, lrmax0_2]

    run_agents_lifelong(
        agents_pool, env_distribution, samples=10, episodes=100, steps=10,
        reset_at_terminal=False, open_plot=True, cumulative_plot=False
    )


if __name__ == '__main__':
    np.random.seed(1995)
    experiment()
