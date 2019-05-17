"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.agents.rmax_maxqinit import RMaxMaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .9


def experiment():
    n_env = 5
    env_distribution = make_env_distribution(env_class='maze-multi-walls', env_name='maze-multi-walls', n_env=n_env, gamma=GAMMA)
    actions = env_distribution.get_actions()
    p_min = 1. / float(n_env)
    delta = .1

    m = 100
    max_mem = 10
    rmax = RMax(actions=actions, gamma=GAMMA, count_threshold=m)
    rmax_q = RMaxMaxQInit(actions=actions, gamma=GAMMA, count_threshold=m, min_sampling_probability=p_min, delta=delta)
    lrmax10 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=10.)
    lrmax5 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=5.)
    lrmax2 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=2.)
    lrmax1 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=1.)
    lrmax05 = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.5)
    lrmax_learn = LRMax(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=None)

    agents_pool = [rmax, lrmax2, lrmax1, lrmax05, lrmax_learn, rmax_q]

    run_agents_lifelong(
        agents_pool, env_distribution, samples=20, episodes=100, steps=1000, reset_at_terminal=False,
        open_plot=True, cumulative_plot=False, is_tracked_value_discounted=True, plot_only=False
    )


if __name__ == '__main__':
    np.random.seed(199311)
    experiment()
