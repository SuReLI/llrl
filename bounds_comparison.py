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
from llrl.agents.experimental.lrmax_bounds_use import ExpLRMax
from llrl.agents.experimental.rmax_bounds_use import ExpRMax
from llrl.experiments import run_agents_on_mdp


def plot_bound_use(path):
    # 1. Results in terms of bound-use
    df = pd.read_csv(path)
    prior = df.prior
    ratio = df.ratio_lip_bound_use_mean
    ratio_up = df.ratio_lip_bound_use_upper
    ratio_lo = df.ratio_lip_bound_use_lower
    speed_up = df.speed_up_mean
    speed_up_up = df.speed_up_upper
    speed_up_lo = df.speed_up_lower

    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    colors = ['steelblue', 'darkorange', 'green']
    plt.plot(prior, ratio, '-o', color=colors[0], label=r'\% use Lipschitz bound')
    plt.fill_between(prior, ratio_up, ratio_lo, color=colors[0], alpha=0.2)
    plt.plot(prior, speed_up, '-^', color=colors[1], label=r'\% time-steps gained (speed-up)')
    plt.fill_between(prior, speed_up_up, speed_up_lo, color=colors[1], alpha=0.2)

    plt.xlim(1., 0.)  # decreasing upper-bound
    plt.xlabel(r'Prior knowledge (known upper-bound on $\max_{s, a} = |R_s^a - \bar{R}_s^a|$)')
    plt.ylabel(r'\%')
    plt.legend(loc='best')
    # plt.title('')
    plt.grid(True, linestyle='--')
    plt.show()


def bounds_comparison_experiment(verbose=False):
    # Parameters
    gamma = 0.9
    n_instances = 2
    prior_min = (1. + gamma) / (1. - gamma)
    prior_max = 0.
    priors = [round(p, 1) for p in np.linspace(start=prior_min, stop=prior_max, num=5)]
    # priors = [0.1]  # TODO remove
    save_path = 'results/bounds_comparison_results.csv'

    # Environments
    sz = 2
    n_states = int(sz * sz)
    mdp1 = GridWorld(width=sz, height=sz, init_loc=(1, 1), goal_locs=[(sz, sz)], goal_reward=0.8)
    mdp2 = GridWorld(width=sz, height=sz, init_loc=(1, 1), goal_locs=[(sz, sz)], goal_reward=1.0)

    actions = mdp1.get_actions()
    r_max = 1.
    n_known = 1
    epsilon_q = .01
    epsilon_m = .01
    delta = .1

    n_episodes = 10  # 100
    n_steps = 10  # 30

    results = []

    for _ in range(n_instances):
        lrmax, rmax, df_lrmax, df_rmax = None, None, None, None
        for i in range(len(priors)):
            lrmax = ExpLRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=None, deduce_v_max=True, n_known=n_known,
                             deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                             n_states=n_states, max_memory_size=None, prior=priors[i], estimate_distances_online=True,
                             min_sampling_probability=.5, name="ExpLRMax")
            rmax = ExpRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=None, deduce_v_max=True, n_known=n_known,
                           deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                           n_states=n_states, name="ExpRMax")

            if i > 0:
                lrmax.data = df_lrmax
                rmax.data = df_rmax

            # Run twice
            lrmax.write_data = False
            run_agents_on_mdp([lrmax], mdp1, n_instances=1, n_episodes=n_episodes, n_steps=n_steps,
                              reset_at_terminal=False, verbose=False)

            lrmax.write_data = True
            run_agents_on_mdp([lrmax, rmax], mdp2, n_instances=1, n_episodes=n_episodes, n_steps=n_steps,
                              reset_at_terminal=False, verbose=False)

            # Retrieve data
            df_lrmax = lrmax.data
            df_rmax = rmax.data

        print(df_lrmax)
        print(df_rmax)

        exit()  # TODO here


        df = pd.concat([df_lrmax, df_rmax], axis=1, sort=False)
        result = []
        for index, row in df.iterrows():
            prior = row['prior']
            ratio_lip_bound_use = 100. * row['ratio_lip_bound_use']
            speed_up = 100. * (row['rmax_n_time_steps_cv'] - row['lrmax_n_time_steps_cv']) / row['rmax_n_time_steps_cv']
            result.append([prior, ratio_lip_bound_use, speed_up])
        results.append(result)

    # Gather results
    csv_write(
        ['prior',
         'ratio_lip_bound_use_mean',
         'ratio_lip_bound_use_upper',
         'ratio_lip_bound_use_lower',
         'speed_up_mean',
         'speed_up_upper',
         'speed_up_lower'],
        save_path, 'w'
    )
    for i in range(len(priors)):
        rlbu = []
        su = []
        for result in results:
            rlbu.append(result[i][1])
            su.append(result[i][2])
        rlbu_mci = mean_confidence_interval(rlbu)
        su_mci = mean_confidence_interval(su)
        csv_write([priors[i]] + list(rlbu_mci) + list(su_mci), save_path, 'a')
    if verbose:
        for result in results:
            print(result)


if __name__ == '__main__':
    bounds_comparison_experiment()
    # plot_bound_use(SAVE_PATH)
    # plot_bound_use('results/bounds-use-and-speed-up/gridworld_h-2_w-2/data.csv')
