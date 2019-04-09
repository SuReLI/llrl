"""
Generic functions for environment management.
"""

import numpy as np

from simple_rl.mdp import MDPDistribution
from llrl.envs.gridworld import GridWorld


def sample_grid_world(gamma, w, h, verbose=False):
    r_min = 0.8
    r_max = 1.0
    possible_goals = [(w, h)]  # [(1, h), (w, 1)]

    sampled_reward = np.random.uniform(r_min, r_max)
    sampled_goal = possible_goals[np.random.randint(0, len(possible_goals))]
    env = GridWorld(
        width=w, height=h, init_loc=(1, 1), goal_locs=[sampled_goal],
        gamma=gamma, slip_prob=0.0, goal_reward=sampled_reward, name="grid-world"
    )

    if verbose:
        print('Sampled grid-world - goal location:', sampled_goal, '- goal reward:', sampled_reward)

    return env


def sample_corridor(gamma, w, verbose=False):
    r_min = 0.8
    r_max = 1.0
    possible_goals = [(w, 1)]
    init_loc = (int(w / 2.), 1)

    sampled_reward = np.random.uniform(r_min, r_max)
    sampled_goal = possible_goals[np.random.randint(0, len(possible_goals))]
    env = GridWorld(
        width=w, height=1, init_loc=init_loc, goal_locs=[sampled_goal],
        gamma=gamma, slip_prob=0.0, goal_reward=sampled_reward, name="corridor"
    )

    env.actions = ['left', 'right']

    if verbose:
        print('Sampled corridor - goal location:', sampled_goal, '- goal reward:', sampled_reward)

    return env


def make_env_distribution(env_class='grid-world', n_env=10, gamma=.9, w=5, h=5, horizon=0, verbose=True):
    """
    Create a distribution over environments.
    This function is specialized to the included environments.
    :param env_class: (str) name of the environment class
    :param n_env: (int) number of environments in the distribution
    :param gamma: (float) discount factor
    :param w: (int) width for grid-world
    :param h: (int) height for grid-world
    :param horizon: (int)
    :param verbose: (bool) print info if True
    :return: (MDPDistribution)
    """
    if verbose:
        print('Creating', n_env, 'environments of class', env_class)

    sampling_probability = 1. / float(n_env)
    env_dist_dict = {}

    for _ in range(n_env):
        if env_class == 'grid-world':
            new_env = sample_grid_world(gamma, w, h, verbose)
        elif env_class == 'corridor':
            new_env = sample_corridor(gamma, w, verbose)
        else:
            raise ValueError('Environment class not implemented.')
        env_dist_dict[new_env] = sampling_probability

    return MDPDistribution(env_dist_dict, horizon=horizon)