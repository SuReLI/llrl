"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge + speed-up.
"""

import numpy as np

from llrl.utils.experiments import prior_use_utils as utils
from llrl.envs.heatmap import HeatMap
from llrl.agents.experimental.lrmax_prior_use import ExpLRMax
from llrl.experiments import run_single_agent_on_mdp


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
    # Parameters
    path = 'results/'
    gamma = .9
    r_max = 1.
    v_max = None
    epsilon_q = .01
    epsilon_m = .01

    n_instances = 10
    n_episodes = 1000
    n_steps = 1000

    prior_min = (1. + gamma) / (1. - gamma)
    mrior_max = 0.
    priors = [19.0, 17.0, 15.0, 10.0, 0.0]

    # Environment
    w = 4
    h = 4
    walls = [(2, 2), (3, 2), (4, 2), (2, 4)]
    n_states = w * h - len(walls)
    print(n_states)
    exit()
    env1 = HeatMap(
        width=w, height=h, init_loc=(1, 1), goal_locs=[(w, h)], is_goal_terminal=False,
        walls=walls, slip_prob=0.1, goal_reward=1.0, reward_span=1.0
    )
    env2 = HeatMap(
        width=w, height=h, init_loc=(1, 1), goal_locs=[(w-1, h)], is_goal_terminal=False,
        walls=walls, slip_prob=0.05, goal_reward=0.6, reward_span=1.5
    )
    actions = env1.get_actions()

    # Compute needed number of samples for L-R-MAX to achieve epsilon optimal behavior with probability (1 - delta)
    epsilon = .1
    delta = .05
    m_r = np.log(2. / delta) / (2. * epsilon ** 2)
    m_t = 2. * (np.log(2**(float(w * h) - float(len(walls))) - 2.) - np.log(delta)) / (epsilon ** 2)
    m = int(max(m_r, m_t))

    names = []

    for p in priors:
        results = []
        name = 'default'
        for i in range(n_instances):
            name = 'ExpLRMax(' + str(p) + ')'
            agent = ExpLRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=True, n_known=None,
                             deduce_n_known=True, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                             n_states=n_states, max_memory_size=None, prior=p, estimate_distances_online=False,
                             min_sampling_probability=.1, name=name)

            if run_experiment:
                if verbose:
                    print('Running instance', i + 1, 'of', n_instances, 'for agent', name)

                run_single_agent_on_mdp(agent, env1, n_episodes, n_steps, experiment=None, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False, verbose=False)
                agent.reset()
                run_single_agent_on_mdp(agent, env2, n_episodes, n_steps, experiment=None, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False, verbose=False)

                results.append(agent.get_results())

        names.append(name)

        # Save results
        if run_experiment:
            utils.save_result(results, path, name)

    # Plot
    utils.plot_computation_number_results(path, names, open_plot)
    utils.plot_time_step_results(path, names, open_plot)


if __name__ == '__main__':
    np.random.seed(1993)
    prior_use_experiment(run_experiment=False, open_plot=True, verbose=True)
