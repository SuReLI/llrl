"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path
sys.path.append('/home/sureli/Documents/erwan/git/llrl')
sys.path.append('/home/sureli/Documents/erwan/git/llrl/venv/lib/python3.5/site-packages')

from llrl.agents.rmax import RMax
from llrl.agents.lrmax_ct import LRMaxCT
from llrl.agents.rmax_maxqinit import RMaxMaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .9


def experiment():
    n_env = 5
    env_distribution = make_env_distribution(env_class='heat-map', n_env=n_env, gamma=GAMMA, w=20, h=20)
    actions = env_distribution.get_actions()
    p_min = 1. / float(n_env)
    delta = .1

    m = 1
    max_mem = 4
    rmax = RMax(actions=actions, gamma=GAMMA, count_threshold=m)
    rmax_q = RMaxMaxQInit(actions=actions, gamma=GAMMA, count_threshold=m, min_sampling_probability=p_min, delta=delta)
    lrmax0_2 = LRMaxCT(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.2)
    lrmax0_6 = LRMaxCT(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=0.6)
    lrmax1_0 = LRMaxCT(actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=1.0)
    lrmax_learn = LRMaxCT(
        actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem, prior=None,
        min_sampling_probability=p_min, delta=delta
    )

    agents_pool = [rmax, lrmax0_2, lrmax0_6, lrmax1_0, lrmax_learn, rmax_q]

    run_agents_lifelong(
        agents_pool, env_distribution, samples=30, episodes=30, steps=10, reset_at_terminal=False,
        open_plot=True, cumulative_plot=False, is_tracked_value_discounted=True, plot_only=False
    )


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()