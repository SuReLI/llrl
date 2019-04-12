"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge + speed-up.
"""

import numpy as np

from llrl.utils.experiments import prior_use_utils as utils
from llrl.envs.gridworld import GridWorld
from llrl.envs.heatmap import HeatMap
from llrl.agents.experimental.lrmax_prior_use import LRMaxExp
from simple_rl.run_experiments import run_single_agent_on_mdp

ROOT_PATH = 'results/prior_use/'

GAMMA = 0.9

N_INSTANCES = 10
N_EPISODES = 1000
N_STEPS = 1000

PRIOR_MIN = (1. + GAMMA) / (1. - GAMMA)
PRIOR_MAX = 0.
# PRIORS = [round(p, 1) for p in np.linspace(start=PRIOR_MIN, stop=PRIOR_MAX, num=5)]
PRIORS = [19.0, 17.0, 15.0, 10.0, 0.0]


def prior_use_experiment(run_experiment=True, open_plot=True, verbose=True):
    """
    Prior use experiment:
    Record the ratio of prior use during the model's distance computation in the simple setting of interacting
    sequentially with two different environments.
    :param run_experiment: (bool) set to False for plot only
    :param open_plot: (bool) set to False to disable plot (only saving)
    :param verbose: (bool)
    :return: None
    """
    w = 4
    h = 4
    walls = [(2, 2), (3, 2), (4, 2), (2, 4)]
    env1 = HeatMap(
        width=w, height=h, init_loc=(1, 1), goal_locs=[(w, h)], is_goal_terminal=False,
        walls=walls, slip_prob=0.1, goal_reward=1.0, reward_span=1.0
    )
    env2 = HeatMap(
        width=w, height=h, init_loc=(1, 1), goal_locs=[(w-1, h)], is_goal_terminal=False,
        walls=walls, slip_prob=0.05, goal_reward=0.6, reward_span=1.5
    )
    '''
    env1 = GridWorld(
        width=w, height=h, init_loc=(2, 1), goal_locs=[(w, h)],
        slip_prob=0.1, goal_reward=0.9, is_goal_terminal=False
    )
    env2 = GridWorld(
        width=w, height=h, init_loc=(2, 1), goal_locs=[(w, h)],
        slip_prob=0.2, goal_reward=1.0, is_goal_terminal=False
    )
    '''

    # Compute needed number of samples for L-R-MAX to achieve epsilon optimal behavior with probability (1 - delta)
    epsilon = .1
    delta = .05
    m_r = np.log(2. / delta) / (2. * epsilon ** 2)
    m_t = 2. * (np.log(2**(float(w * h) - float(len(walls))) - 2.) - np.log(delta)) / (epsilon ** 2)
    m = int(max(m_r, m_t))

    names = []

    for p in PRIORS:
        results = []
        name = 'default'
        for i in range(N_INSTANCES):
            agent = LRMaxExp(
                actions=env1.get_actions(),
                gamma=GAMMA,
                count_threshold=m,
                epsilon=epsilon,
                prior=p
            )
            name = agent.name

            if run_experiment:
                if verbose:
                    print('Running instance', i + 1, 'of', N_INSTANCES, 'for agent', name)

                run_single_agent_on_mdp(
                    agent, env1, episodes=N_EPISODES, steps=N_STEPS, experiment=None, verbose=False,
                    track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False
                )
                agent.reset()
                run_single_agent_on_mdp(
                    agent, env2, episodes=N_EPISODES, steps=N_STEPS, experiment=None, verbose=False,
                    track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False
                )

                results.append(agent.get_results())

        names.append(name)

        # Save results
        if run_experiment:
            utils.save_result(results, ROOT_PATH, name)

    # Plot
    utils.plot_computation_number_results(ROOT_PATH, names, open_plot)
    utils.plot_time_step_results(ROOT_PATH, names, open_plot)


if __name__ == '__main__':
    np.random.seed(1993)
    prior_use_experiment(run_experiment=True, open_plot=True, verbose=True)
