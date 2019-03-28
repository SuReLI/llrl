"""
Generic functions for environment management.
"""

import numpy as np

from simple_rl.mdp import MDPDistribution
from llrl.envs.gridworld import GridWorld


def make_env_distribution(env_class='grid-world', n_env=10, gamma=.9, horizon=0, w=5, h=5, verbose=True):
    """
    Create a distribution over environments.
    This function is specialized to the included environments.
    :param env_class: (str) name of the environment class
    :param n_env: (int) number of environments in the distribution
    :param gamma: (float) discount factor
    :param horizon: (int)
    :param w: (int) width for grid-world
    :param h: (int) height for grid-world
    :param verbose: (bool) print info if True
    :return: (MDPDistribution)
    """
    if verbose:
        print('Creating', n_env, 'environments of class', env_class)

    sampling_probability = 1. / float(n_env)
    env_dist_dict = {}

    for _ in range(n_env):
        sampled_reward = np.random.uniform()
        new_env = GridWorld(
            width=w, height=h, init_loc=(1, 1), goal_locs=[(w, h)],
            gamma=gamma, slip_prob=0.0, goal_reward=sampled_reward, name="grid-world"
        )
        env_dist_dict[new_env] = sampling_probability

    return MDPDistribution(env_dist_dict, horizon=horizon)