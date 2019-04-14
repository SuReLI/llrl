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

    epsilon = .1
    delta = .05
    n_states = float(24)
    m_r = np.log(2. / delta) / (2. * epsilon ** 2)
    m_t = 2. * (np.log(2**(n_states - 2.)) - np.log(delta)) / (epsilon**2)
    m = int(max(m_r, m_t))
    m = 100  # TODO remove

    max_mem = 4
    rmax = RMax(actions=actions, gamma=GAMMA, count_threshold=m)
    lrmax19 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=19.)
    lrmax13 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=13.)
    lrmax6 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=6.)
    lrmax018 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.18)

    agents_pool = [lrmax018, rmax]

    run_agents_lifelong(
        agents_pool, env_distribution, samples=10, episodes=100, steps=1000,
        reset_at_terminal=False, open_plot=True, cumulative_plot=False, is_tracked_value_discounted=True
    )


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
