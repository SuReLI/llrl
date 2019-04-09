"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge + speed-up.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import csv_write
from llrl.envs.gridworld import GridWorld
from llrl.agents.experimental.lrmax_prior_use import LRMaxExp
from simple_rl.run_experiments import run_agents_on_mdp

GAMMA = 0.9

N_EPISODES = 10
N_STEPS = 10

PRIOR_MIN = (1. + GAMMA) / (1. - GAMMA)
PRIOR_MAX = 0.
PRIORS = [round(p, 1) for p in np.linspace(start=PRIOR_MIN, stop=PRIOR_MAX, num=3)]


def prior_use_experiment():
    size = 2
    env1 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], slip_prob=0.1, goal_reward=0.9)
    env2 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], slip_prob=0.2, goal_reward=1.0)

    # Compute needed number of samples for L-R-MAX to achieve epsilon optimal behavior with probability (1 - delta)
    epsilon = .1
    delta = .05
    m_r = np.log(2. / delta) / (2. * epsilon ** 2)
    m_t = 2. * (np.log(2 ** (float(size * size)) - 2.) - np.log(delta)) / (epsilon ** 2)
    m = int(max(m_r, m_t))

    for p in PRIORS:
        lrmax = LRMaxExp(
            actions=env1.get_actions(),
            gamma=GAMMA,
            count_threshold=m,
            epsilon=epsilon,
            prior=p
        )

        # Run twice
        run_agents_on_mdp([lrmax], env1, instances=1, episodes=N_EPISODES, steps=N_STEPS,
                          reset_at_terminal=False, verbose=False, open_plot=False)

        run_agents_on_mdp([lrmax], env2, instances=1, episodes=N_EPISODES, steps=N_STEPS,
                          reset_at_terminal=False, verbose=False, open_plot=False)


if __name__ == '__main__':
    np.random.seed(1993)
    prior_use_experiment()
