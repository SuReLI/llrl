"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax_ct import LRMaxCT
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .9
WIDTH = 3
HEIGHT = 3


def experiment():
    # Create environments distribution
    env_distribution = make_env_distribution(env_class='grid-world', n_env=10, gamma=GAMMA, w=WIDTH, h=HEIGHT)

    rmax = RMax(actions=env_distribution.get_actions(), gamma=GAMMA, count_threshold=1)
    lrmax02 = LRMaxCT(actions=env_distribution.get_actions(), gamma=GAMMA, count_threshold=1, prior=.2)
    lrmax07 = LRMaxCT(actions=env_distribution.get_actions(), gamma=GAMMA, count_threshold=1, prior=.7)
    lrmax1 = LRMaxCT(actions=env_distribution.get_actions(), gamma=GAMMA, count_threshold=1, prior=1.0)

    agents_pool = [rmax, lrmax02]

    run_agents_lifelong(
        agents_pool, env_distribution, samples=5, episodes=15, steps=100,
        reset_at_terminal=False, open_plot=True, cumulative_plot=False
    )


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
