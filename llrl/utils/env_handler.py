"""
Generic functions for environment management.
"""

import numpy as np

from simple_rl.mdp import MDPDistribution
from llrl.envs.gridworld import GridWorld
from llrl.envs.gridworld import coord_from_binary_list
from llrl.envs.heatmap import HeatMap


def sample_grid_world(gamma, env_name, w, h, verbose=False):
    if env_name is None:
        env_name = "grid-world"

    r_min = 0.9
    r_max = 1.0
    possible_goals = [(w, h), (w-1, h), (w, h-1), (w-2, h)]

    sampled_reward = np.random.uniform(r_min, r_max)
    sampled_goal = possible_goals[np.random.randint(0, len(possible_goals))]
    env = GridWorld(
        width=w, height=h, init_loc=(1, 1), goal_locs=[sampled_goal],
        gamma=gamma, slip_prob=0.0, goal_reward=sampled_reward, name=env_name
    )

    if verbose:
        print('Sampled grid-world - goal location:', sampled_goal, '- goal reward:', sampled_reward)

    return env


def sample_corridor(gamma, env_name, w, verbose=False):
    if env_name is None:
        env_name = "corridor"

    r_min = 0.8
    r_max = 1.0
    possible_goals = [(w, 1)]
    init_loc = (int(w / 2.), 1)

    sampled_reward = np.random.uniform(r_min, r_max)
    sampled_goal = possible_goals[np.random.randint(0, len(possible_goals))]
    env = GridWorld(
        width=w, height=1, init_loc=init_loc, goal_locs=[sampled_goal],
        gamma=gamma, slip_prob=0.0, goal_reward=sampled_reward, name=env_name
    )

    env.actions = ['left', 'right']

    if verbose:
        print('Sampled corridor - goal location:', sampled_goal, '- goal reward:', sampled_reward)

    return env


def sample_heat_map(gamma, env_name, verbose=False):
    if env_name is None:
        env_name = "heat-map"

    w = 11
    h = 11
    possible_goals = [(w - 1, h), (w, h - 1), (w, h)]

    sampled_reward = np.random.uniform(0.8, 1.0)
    sampled_span = np.random.uniform(0.5, 1.5)
    sampled_goal = possible_goals[np.random.randint(0, len(possible_goals))]

    env = HeatMap(
        width=w, height=h, init_loc=(5, 5), rand_init=False, goal_locs=[sampled_goal], lava_locs=[()], walls=[],
        is_goal_terminal=False, gamma=gamma, slip_prob=0.0, step_cost=0.0, lava_cost=0.01,
        goal_reward=sampled_reward, reward_span=sampled_span, name=env_name
    )

    if verbose:
        print(
            'Sampled heat-map - goal location:', sampled_goal, '- goal reward:', sampled_reward,
            '- reward span:', sampled_span
        )

    return env


def sample_test_environment(gamma):
    w, h = 5, 3

    init_loc = (3, 2)
    goals = [(5, 1), (5, 3)]
    slip_probabilities = [0., 1.]
    walls = []

    index = np.random.randint(0, len(goals))
    g = goals[index]
    s = 0.  # slip_probabilities[index]

    env = GridWorld(
        width=w, height=h, init_loc=init_loc, rand_init=False, goal_locs=[g], lava_locs=[()], walls=walls,
        is_goal_terminal=True, gamma=gamma, slip_prob=s, step_cost=0.0, lava_cost=0.01,
        goal_reward=1, name="test"
    )
    print('Goal:', g, 'slip probability:', s)
    return env


def sample_maze(gamma, env_name, verbose=False):
    if env_name is None:
        env_name = "maze"

    w, h = 11, 11
    goals = coord_from_binary_list(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    walls = [
        coord_from_binary_list(
            [
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            ]
        ),
        coord_from_binary_list(
            [
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            ]
        )
    ]

    index = np.random.randint(0, len(walls))
    sl = 0.
    wa = walls[index]

    env = GridWorld(
        width=w, height=h, init_loc=(6, 6), rand_init=False, goal_locs=goals, lava_locs=[()], walls=wa,
        is_goal_terminal=True, gamma=gamma, slip_prob=sl, step_cost=0.0, lava_cost=0.01,
        goal_reward=1, name=env_name
    )

    if verbose:
        print('Sampled maze - index:', index)

    return env


def make_env_distribution(env_class='grid-world', env_name=None, n_env=10, gamma=.9, w=5, h=5, horizon=0, verbose=True):
    """
    Create a distribution over environments.
    This function is specialized to the included environments.
    :param env_class: (str) name of the environment class
    :param env_name: (str) name of the environment for save path
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
            new_env = sample_grid_world(gamma, env_name, w, h, verbose)
        elif env_class == 'corridor':
            new_env = sample_corridor(gamma, env_name, w, verbose)
        elif env_class == 'heat-map':
            new_env = sample_heat_map(gamma, env_name, verbose)
        elif env_class == 'maze':
            new_env = sample_maze(gamma, env_name, verbose)
        elif env_class == 'test':
            new_env = sample_test_environment(gamma)
        else:
            raise ValueError('Environment class not implemented.')
        env_dist_dict[new_env] = sampling_probability

    return MDPDistribution(env_dist_dict, horizon=horizon)
